"""Evaluate relational bidirectional probes on synthetic KG JSONL data.

The default score is span pseudo-log-likelihood: each candidate string is placed
at the [MASK] slot, all tokens belonging to that candidate span are replaced by
[MASK], and the model scores the original candidate tokens.  This avoids the
first-token collision problem that appears with byte-level BPE names.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

import torch

from unlearning_architecture.adapt import add_mask_token
from unlearning_architecture.eval import force_attention_mode, load_model_and_tokenizer


DEFAULT_SPLITS = [
    "eval_context",
    "eval_memory",
    "eval_logic",
    "eval_counterfactual",
]


@dataclass
class ProbeExample:
    row: dict
    input_ids: list[int]
    mask_pos: int
    target_token_id: int
    target_token_text: str
    candidate_token_ids: list[int]
    candidate_token_texts: list[str]
    has_candidate_collision: bool


@dataclass
class SpanCandidate:
    text: str
    input_ids: list[int]
    label_positions: list[int]
    label_ids: list[int]


@dataclass
class SpanProbeExample:
    row: dict
    target: str
    row_candidates: list[str]
    score_candidates: list[SpanCandidate]
    has_candidate_collision: bool


def parse_ckpts(items: list[str]) -> list[tuple[str, str]]:
    parsed = []
    for item in items:
        if "=" in item:
            label, ckpt = item.split("=", 1)
        else:
            ckpt = item
            label = Path(item).name or item
        parsed.append((label, ckpt))
    return parsed


def dtype_from_name(name: str, device: str) -> torch.dtype:
    if device == "cpu":
        return torch.float32
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def iter_jsonl(path: Path, max_rows: int | None = None) -> Iterable[dict]:
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            if line.strip():
                yield json.loads(line)


def load_rows(dataset: Path, splits: list[str], max_rows_per_split: int | None) -> list[dict]:
    rows = []
    for split in splits:
        path = dataset / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(iter_jsonl(path, max_rows_per_split))
    return rows


def first_token_id_in_context(tokenizer, prefix: str, candidate: str) -> int:
    """Return the first token id overlapping ``candidate`` in ``prefix+candidate``.

    This avoids mistakes with byte-level BPE tokenization when the prefix ends
    in whitespace and the tokenizer encodes " word" as one token.
    """
    text = prefix + candidate
    start = len(prefix)
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    for token_id, (s, e) in zip(encoded["input_ids"], encoded["offset_mapping"], strict=True):
        if e > start and s < len(text):
            return int(token_id)
    raise ValueError(f"Could not find candidate token in context: prefix={prefix!r} candidate={candidate!r}")


def build_example(row: dict, tokenizer, mask_id: int) -> ProbeExample:
    text = row["input"]
    if text.count("[MASK]") != 1:
        raise ValueError(f"Expected exactly one [MASK] in row {row.get('id')}: {text!r}")
    prefix = text.split("[MASK]", 1)[0]
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = [int(x) for x in encoded["input_ids"]]
    mask_positions = [i for i, token_id in enumerate(input_ids) if token_id == mask_id]
    if len(mask_positions) != 1:
        raise ValueError(
            f"Tokenizer produced {len(mask_positions)} mask tokens in row {row.get('id')}: {text!r}"
        )
    target_token_id = first_token_id_in_context(tokenizer, prefix, row["target"])
    candidate_token_ids = [
        first_token_id_in_context(tokenizer, prefix, candidate)
        for candidate in row.get("candidates", [row["target"]])
    ]
    candidate_token_texts = [tokenizer.decode([token_id]) for token_id in candidate_token_ids]
    return ProbeExample(
        row=row,
        input_ids=input_ids,
        mask_pos=mask_positions[0],
        target_token_id=target_token_id,
        target_token_text=tokenizer.decode([target_token_id]),
        candidate_token_ids=candidate_token_ids,
        candidate_token_texts=candidate_token_texts,
        has_candidate_collision=len(set(candidate_token_ids)) < len(candidate_token_ids),
    )


def prepare_examples(rows: list[dict], tokenizer, mask_id: int) -> list[ProbeExample]:
    examples = []
    failures = []
    for row in rows:
        try:
            examples.append(build_example(row, tokenizer, mask_id))
        except Exception as exc:  # noqa: BLE001 - keep row id context for dataset QA.
            failures.append((row.get("id"), str(exc)))
    if failures:
        preview = "\n".join(f"  {row_id}: {msg}" for row_id, msg in failures[:5])
        raise RuntimeError(f"Failed to prepare {len(failures)} examples. First failures:\n{preview}")
    return examples


def span_candidate_encoding(row: dict, candidate: str, tokenizer, mask_id: int) -> SpanCandidate:
    text = row["input"]
    if text.count("[MASK]") != 1:
        raise ValueError(f"Expected exactly one [MASK] in row {row.get('id')}: {text!r}")
    prefix, suffix = text.split("[MASK]", 1)
    full_text = prefix + candidate + suffix
    start = len(prefix)
    end = start + len(candidate)
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = [int(x) for x in encoded["input_ids"]]
    label_positions = [
        i
        for i, (s, e) in enumerate(encoded["offset_mapping"])
        if e > start and s < end
    ]
    if not label_positions:
        raise ValueError(f"Could not find candidate span in row {row.get('id')}: {candidate!r}")
    label_ids = [input_ids[i] for i in label_positions]
    corrupted = list(input_ids)
    for pos in label_positions:
        corrupted[pos] = mask_id
    return SpanCandidate(
        text=candidate,
        input_ids=corrupted,
        label_positions=label_positions,
        label_ids=label_ids,
    )


def prepare_span_examples(rows: list[dict], tokenizer, mask_id: int) -> list[SpanProbeExample]:
    group_targets: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group_id = row.get("group_id")
        if group_id and row["target"] not in group_targets[group_id]:
            group_targets[group_id].append(row["target"])

    examples = []
    failures = []
    for row in rows:
        try:
            row_candidates = list(dict.fromkeys(row.get("candidates", [row["target"]])))
            if row["target"] not in row_candidates:
                row_candidates.insert(0, row["target"])
            score_texts = list(row_candidates)
            for target in group_targets.get(row.get("group_id"), []):
                if target not in score_texts:
                    score_texts.append(target)
            score_candidates = [
                span_candidate_encoding(row, candidate, tokenizer, mask_id)
                for candidate in score_texts
            ]
            label_sequences = [tuple(candidate.label_ids) for candidate in score_candidates]
            examples.append(
                SpanProbeExample(
                    row=row,
                    target=row["target"],
                    row_candidates=row_candidates,
                    score_candidates=score_candidates,
                    has_candidate_collision=len(set(label_sequences)) < len(label_sequences),
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep row id context for dataset QA.
            failures.append((row.get("id"), str(exc)))
    if failures:
        preview = "\n".join(f"  {row_id}: {msg}" for row_id, msg in failures[:5])
        raise RuntimeError(f"Failed to prepare {len(failures)} span examples. First failures:\n{preview}")
    return examples


def unique_preserve_order(values: Iterable[int]) -> list[int]:
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def make_batch(examples: list[ProbeExample], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(ex.input_ids) for ex in examples)
    ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(examples), max_len), dtype=torch.long)
    for i, ex in enumerate(examples):
        cur = torch.tensor(ex.input_ids, dtype=torch.long)
        ids[i, : cur.numel()] = cur
        attn[i, : cur.numel()] = 1
    return ids.to(device), attn.to(device)


def make_span_batch(candidates: list[SpanCandidate], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(len(candidate.input_ids) for candidate in candidates)
    ids = torch.full((len(candidates), max_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(candidates), max_len), dtype=torch.long)
    for i, candidate in enumerate(candidates):
        cur = torch.tensor(candidate.input_ids, dtype=torch.long)
        ids[i, : cur.numel()] = cur
        attn[i, : cur.numel()] = 1
    return ids.to(device), attn.to(device)


def row_group_keys(row: dict) -> list[tuple[str, str]]:
    triple = row.get("triple") or {}
    keys = [
        ("overall", "all"),
        ("split", row.get("split", "unknown")),
        ("task", row.get("task", row.get("train_role", "unknown"))),
    ]
    for name in ("category", "relation", "schema"):
        value = triple.get(name)
        if value is not None:
            keys.append((name, value))
    keys.append(("requires_logic", str(bool(row.get("requires_logic", False))).lower()))
    keys.append(("requires_memory", str(bool(row.get("requires_memory", False))).lower()))
    keys.append(("requires_right_context", str(bool(row.get("requires_right_context", False))).lower()))
    return keys


def init_accumulator() -> dict:
    return {
        "n": 0,
        "candidate_top1": 0,
        "candidate_top5": 0,
        "vocab_top1": 0,
        "vocab_top5": 0,
        "collision": 0,
        "target_logprob_sum": 0.0,
        "candidate_rank_sum": 0.0,
        "vocab_rank_sum": 0.0,
        "candidate_margin_sum": 0.0,
    }


def init_span_accumulator() -> dict:
    return {
        "n": 0,
        "candidate_top1": 0,
        "candidate_top5": 0,
        "collision": 0,
        "target_score_sum": 0.0,
        "candidate_rank_sum": 0.0,
        "candidate_margin_sum": 0.0,
        "target_token_count_sum": 0,
    }


def update_accumulator(acc: dict, metric: dict) -> None:
    acc["n"] += 1
    acc["candidate_top1"] += int(metric["candidate_rank"] == 1)
    acc["candidate_top5"] += int(metric["candidate_rank"] <= 5)
    acc["vocab_top1"] += int(metric["vocab_rank"] == 1)
    acc["vocab_top5"] += int(metric["vocab_rank"] <= 5)
    acc["collision"] += int(metric["has_candidate_collision"])
    acc["target_logprob_sum"] += metric["target_logprob"]
    acc["candidate_rank_sum"] += metric["candidate_rank"]
    acc["vocab_rank_sum"] += metric["vocab_rank"]
    acc["candidate_margin_sum"] += metric["candidate_margin"]


def update_span_accumulator(acc: dict, metric: dict) -> None:
    acc["n"] += 1
    acc["candidate_top1"] += int(metric["candidate_rank"] == 1)
    acc["candidate_top5"] += int(metric["candidate_rank"] <= 5)
    acc["collision"] += int(metric["has_candidate_collision"])
    acc["target_score_sum"] += metric["target_score"]
    acc["candidate_rank_sum"] += metric["candidate_rank"]
    acc["candidate_margin_sum"] += metric["candidate_margin"]
    acc["target_token_count_sum"] += metric["target_token_count"]


def finalize_accumulator(acc: dict) -> dict:
    n = acc["n"]
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "candidate_top1": acc["candidate_top1"] / n,
        "candidate_top5": acc["candidate_top5"] / n,
        "vocab_top1": acc["vocab_top1"] / n,
        "vocab_top5": acc["vocab_top5"] / n,
        "candidate_collision_rate": acc["collision"] / n,
        "mean_target_logprob": acc["target_logprob_sum"] / n,
        "mean_candidate_rank": acc["candidate_rank_sum"] / n,
        "mean_vocab_rank": acc["vocab_rank_sum"] / n,
        "mean_candidate_margin": acc["candidate_margin_sum"] / n,
    }


def finalize_span_accumulator(acc: dict) -> dict:
    n = acc["n"]
    if n == 0:
        return {"n": 0}
    return {
        "n": n,
        "candidate_top1": acc["candidate_top1"] / n,
        "candidate_top5": acc["candidate_top5"] / n,
        "candidate_collision_rate": acc["collision"] / n,
        "mean_target_score": acc["target_score_sum"] / n,
        "mean_candidate_rank": acc["candidate_rank_sum"] / n,
        "mean_candidate_margin": acc["candidate_margin_sum"] / n,
        "mean_target_token_count": acc["target_token_count_sum"] / n,
    }


def candidate_metrics(logits: torch.Tensor, ex: ProbeExample) -> dict:
    target_score = logits[ex.target_token_id].item()
    target_logprob = torch.log_softmax(logits, dim=-1)[ex.target_token_id].item()
    vocab_rank = int((logits > logits[ex.target_token_id]).sum().item()) + 1

    unique_ids = unique_preserve_order(ex.candidate_token_ids)
    cand_scores = torch.tensor([logits[token_id].item() for token_id in unique_ids])
    target_unique_index = unique_ids.index(ex.target_token_id)
    candidate_rank = int((cand_scores > cand_scores[target_unique_index]).sum().item()) + 1
    wrong_scores = [
        score.item()
        for i, score in enumerate(cand_scores)
        if unique_ids[i] != ex.target_token_id
    ]
    best_wrong = max(wrong_scores) if wrong_scores else -math.inf

    return {
        "target_token_id": ex.target_token_id,
        "target_token_text": ex.target_token_text,
        "target_logprob": target_logprob,
        "vocab_rank": vocab_rank,
        "candidate_rank": candidate_rank,
        "candidate_margin": target_score - best_wrong,
        "has_candidate_collision": ex.has_candidate_collision,
    }


def span_candidate_metrics(ex: SpanProbeExample, scores: dict[str, float]) -> dict:
    row_scores = [scores[candidate] for candidate in ex.row_candidates]
    target_score = scores[ex.target]
    rank = int(sum(score > target_score for score in row_scores)) + 1
    wrong_scores = [
        score
        for candidate, score in zip(ex.row_candidates, row_scores, strict=True)
        if candidate != ex.target
    ]
    best_wrong = max(wrong_scores) if wrong_scores else -math.inf
    target_encoding = next(candidate for candidate in ex.score_candidates if candidate.text == ex.target)
    return {
        "target_score": target_score,
        "candidate_rank": rank,
        "candidate_margin": target_score - best_wrong,
        "target_token_count": len(target_encoding.label_ids),
        "has_candidate_collision": ex.has_candidate_collision,
    }


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    eps = 1e-12
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p / m).log()).sum().item() + 0.5 * (q * (q / m).log()).sum().item())


def counterfactual_summary(detail_rows: list[dict]) -> dict:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in detail_rows:
        group_id = row.get("group_id")
        if group_id:
            by_group[group_id].append(row)

    group_acc = []
    group_jsd = []
    for rows in by_group.values():
        if len(rows) < 2:
            continue
        target_ids = unique_preserve_order(row["target_token_id"] for row in rows)
        if len(target_ids) < 2:
            continue
        vectors = []
        correct = 0
        total = 0
        for row in rows:
            scores_by_id = {int(k): v for k, v in row["counterfactual_target_logits"].items()}
            scores = torch.tensor([scores_by_id[token_id] for token_id in target_ids], dtype=torch.float32)
            pred_id = target_ids[int(scores.argmax().item())]
            correct += int(pred_id == row["target_token_id"])
            total += 1
            vectors.append(torch.softmax(scores, dim=-1))
        group_acc.append(correct / total)
        pair_jsd = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pair_jsd.append(js_divergence(vectors[i], vectors[j]))
        if pair_jsd:
            group_jsd.append(mean(pair_jsd))

    if not group_acc:
        return {}
    return {
        "counterfactual_groups": len(group_acc),
        "counterfactual_group_target_acc": mean(group_acc),
        "counterfactual_group_target_jsd": mean(group_jsd) if group_jsd else 0.0,
    }


def counterfactual_span_summary(detail_rows: list[dict]) -> dict:
    by_group: dict[str, list[dict]] = defaultdict(list)
    for row in detail_rows:
        group_id = row.get("group_id")
        if group_id:
            by_group[group_id].append(row)

    group_acc = []
    group_jsd = []
    for rows in by_group.values():
        if len(rows) < 2:
            continue
        targets = []
        for row in rows:
            for target in row["counterfactual_target_scores"]:
                if target not in targets:
                    targets.append(target)
        if len(targets) < 2:
            continue
        vectors = []
        correct = 0
        total = 0
        for row in rows:
            scores = torch.tensor(
                [row["counterfactual_target_scores"][target] for target in targets],
                dtype=torch.float32,
            )
            pred = targets[int(scores.argmax().item())]
            correct += int(pred == row["target"])
            total += 1
            vectors.append(torch.softmax(scores, dim=-1))
        group_acc.append(correct / total)
        pair_jsd = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pair_jsd.append(js_divergence(vectors[i], vectors[j]))
        if pair_jsd:
            group_jsd.append(mean(pair_jsd))

    if not group_acc:
        return {}
    return {
        "counterfactual_groups": len(group_acc),
        "counterfactual_group_target_acc": mean(group_acc),
        "counterfactual_group_target_jsd": mean(group_jsd) if group_jsd else 0.0,
    }


@torch.no_grad()
def evaluate_mode(
    model,
    examples: list[ProbeExample],
    mode: str,
    batch_size: int,
    pad_id: int,
    device: str,
    save_details: bool,
) -> tuple[list[dict], list[dict]]:
    attention_patches = force_attention_mode(model, mode)
    if attention_patches == 0 and mode != "natural":
        raise RuntimeError(f"Attention mode switch to {mode!r} did not modify any module.")
    model.eval()
    accumulators: dict[tuple[str, str], dict] = defaultdict(init_accumulator)
    detail_rows: list[dict] = []

    cf_target_ids_by_group: dict[str, list[int]] = defaultdict(list)
    for ex in examples:
        group_id = ex.row.get("group_id")
        if group_id:
            cf_target_ids_by_group[group_id].append(ex.target_token_id)
    cf_target_ids_by_group = {
        group_id: unique_preserve_order(ids)
        for group_id, ids in cf_target_ids_by_group.items()
    }

    for start in range(0, len(examples), batch_size):
        batch_examples = examples[start : start + batch_size]
        ids, attn = make_batch(batch_examples, pad_id, device)
        logits = model(ids, attention_mask=attn).logits.float()
        for i, ex in enumerate(batch_examples):
            row_logits = logits[i, ex.mask_pos]
            metric = candidate_metrics(row_logits, ex)
            for key in row_group_keys(ex.row):
                update_accumulator(accumulators[key], metric)

            group_id = ex.row.get("group_id")
            detail = {
                "id": ex.row.get("id"),
                "split": ex.row.get("split"),
                "task": ex.row.get("task"),
                "group_id": group_id,
                "target": ex.row.get("target"),
                "target_token_id": ex.target_token_id,
                "target_token_text": ex.target_token_text,
                "candidate_rank": metric["candidate_rank"],
                "vocab_rank": metric["vocab_rank"],
                "target_logprob": metric["target_logprob"],
                "candidate_margin": metric["candidate_margin"],
                "has_candidate_collision": ex.has_candidate_collision,
            }
            if group_id:
                target_ids = cf_target_ids_by_group[group_id]
                detail["counterfactual_target_logits"] = {
                    str(token_id): row_logits[token_id].item()
                    for token_id in target_ids
                }
            if save_details or group_id:
                detail_rows.append(detail)

    summary_rows = []
    for (group, value), acc in sorted(accumulators.items()):
        row = {"mode": mode, "group": group, "value": value, "attention_patches": attention_patches}
        row.update(finalize_accumulator(acc))
        summary_rows.append(row)

    cf_summary = counterfactual_summary(detail_rows)
    if cf_summary:
        row = {
            "mode": mode,
            "group": "counterfactual",
            "value": "group_target_set",
            "attention_patches": attention_patches,
        }
        row.update(cf_summary)
        summary_rows.append(row)

    if not save_details:
        detail_rows = [row for row in detail_rows if row.get("group_id")]
    return summary_rows, detail_rows


@torch.no_grad()
def evaluate_span_mode(
    model,
    examples: list[SpanProbeExample],
    mode: str,
    batch_size: int,
    pad_id: int,
    device: str,
    save_details: bool,
    score_norm: str,
) -> tuple[list[dict], list[dict]]:
    attention_patches = force_attention_mode(model, mode)
    if attention_patches == 0 and mode != "natural":
        raise RuntimeError(f"Attention mode switch to {mode!r} did not modify any module.")
    model.eval()
    accumulators: dict[tuple[str, str], dict] = defaultdict(init_span_accumulator)
    detail_rows: list[dict] = []
    score_maps: list[dict[str, float]] = [dict() for _ in examples]
    group_targets_by_group: dict[str, list[str]] = defaultdict(list)
    for ex in examples:
        group_id = ex.row.get("group_id")
        if group_id and ex.target not in group_targets_by_group[group_id]:
            group_targets_by_group[group_id].append(ex.target)

    jobs = []
    for ex_i, ex in enumerate(examples):
        for candidate in ex.score_candidates:
            jobs.append((ex_i, candidate))

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        batch_candidates = [candidate for _, candidate in batch_jobs]
        ids, attn = make_span_batch(batch_candidates, pad_id, device)
        logits = model(ids, attention_mask=attn).logits.float()
        for i, (ex_i, candidate) in enumerate(batch_jobs):
            pos = torch.tensor(candidate.label_positions, dtype=torch.long, device=device)
            labels = torch.tensor(candidate.label_ids, dtype=torch.long, device=device)
            log_probs = torch.log_softmax(logits[i, pos], dim=-1)
            token_scores = log_probs[torch.arange(labels.numel(), device=device), labels]
            if score_norm == "sum":
                score = float(token_scores.sum().item())
            else:
                score = float(token_scores.mean().item())
            score_maps[ex_i][candidate.text] = score

    for ex, scores in zip(examples, score_maps, strict=True):
        metric = span_candidate_metrics(ex, scores)
        for key in row_group_keys(ex.row):
            update_span_accumulator(accumulators[key], metric)

        group_id = ex.row.get("group_id")
        detail = {
            "id": ex.row.get("id"),
            "split": ex.row.get("split"),
            "task": ex.row.get("task"),
            "group_id": group_id,
            "target": ex.target,
            "target_score": metric["target_score"],
            "candidate_rank": metric["candidate_rank"],
            "candidate_margin": metric["candidate_margin"],
            "target_token_count": metric["target_token_count"],
            "has_candidate_collision": ex.has_candidate_collision,
        }
        if group_id:
            detail["counterfactual_target_scores"] = {
                target: scores[target]
                for target in group_targets_by_group[group_id]
                if target in scores
            }
        if save_details or group_id:
            detail_rows.append(detail)

    summary_rows = []
    for (group, value), acc in sorted(accumulators.items()):
        row = {"mode": mode, "group": group, "value": value, "attention_patches": attention_patches}
        row.update(finalize_span_accumulator(acc))
        summary_rows.append(row)

    cf_summary = counterfactual_span_summary(detail_rows)
    if cf_summary:
        row = {
            "mode": mode,
            "group": "counterfactual",
            "value": "group_target_set",
            "attention_patches": attention_patches,
        }
        row.update(cf_summary)
        summary_rows.append(row)

    if not save_details:
        detail_rows = [row for row in detail_rows if row.get("group_id")]
    return summary_rows, detail_rows


def print_key_rows(label: str, rows: list[dict]) -> None:
    wanted = {
        ("split", "eval_context"),
        ("split", "eval_memory"),
        ("split", "eval_logic"),
        ("split", "eval_counterfactual"),
        ("counterfactual", "group_target_set"),
    }
    print(f"\n[relational-bidir] {label}")
    for row in rows:
        key = (row["group"], row["value"])
        if key not in wanted:
            continue
        if row["group"] == "counterfactual":
            print(
                f"  mode={row['mode']:<14} {row['value']:<18} "
                f"group_acc={row['counterfactual_group_target_acc']:.4f} "
                f"jsd={row['counterfactual_group_target_jsd']:.6f} "
                f"groups={row['counterfactual_groups']}",
                flush=True,
            )
        else:
            print(
                f"  mode={row['mode']:<14} {row['value']:<20} "
                f"cand@1={row['candidate_top1']:.4f} "
                f"cand@5={row['candidate_top5']:.4f} "
                f"rank={row['mean_candidate_rank']:.2f} "
                f"n={row['n']}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", action="append", required=True, help="label=ckpt or ckpt. Repeatable.")
    parser.add_argument("--dataset", type=Path, default=Path("data/relational_bidir_v1"))
    parser.add_argument("--split", action="append", choices=DEFAULT_SPLITS)
    parser.add_argument("--mode", action="append", choices=["causal", "bidirectional", "natural"])
    parser.add_argument("--out", type=Path, default=Path("results/relational_bidir_probe.jsonl"))
    parser.add_argument("--detail_out", type=Path)
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--max_rows_per_split", type=int)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--scoring", choices=["span_pll", "first_token"], default="span_pll")
    parser.add_argument("--span_score_norm", choices=["mean", "sum"], default="mean")
    parser.add_argument(
        "--native_dlm",
        action="store_true",
        help="Treat ckpts as native bidirectional DLMs (LLaDA, ...): apply compat "
             "patches at load, skip add_mask_token resize, default mode list to "
             "['natural'] (no causal/bidirectional toggle).",
    )
    args = parser.parse_args()

    splits = args.split or DEFAULT_SPLITS
    if args.native_dlm:
        modes = args.mode or ["natural"]
    else:
        modes = args.mode or ["causal", "bidirectional"]
    rows = load_rows(args.dataset, splits, args.max_rows_per_split)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.detail_out:
        args.detail_out.parent.mkdir(parents=True, exist_ok=True)

    all_summary_rows = []
    all_detail_rows = []
    for label, ckpt in parse_ckpts(args.ckpt):
        dtype = dtype_from_name(args.dtype, args.device)
        model, tok = load_model_and_tokenizer(ckpt, dtype, args.device, native_dlm=args.native_dlm)
        if args.native_dlm:
            # Native DLMs ship their own mask-token convention; do not resize.
            mask_id = tok.mask_token_id
            if mask_id is None:
                raise ValueError(
                    f"--native_dlm requires the tokenizer to define a mask token; "
                    f"{ckpt} reports tok.mask_token_id=None"
                )
        else:
            mask_id = add_mask_token(model, tok)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        if pad_id is None:
            pad_id = 0
        if args.scoring == "span_pll":
            examples = prepare_span_examples(rows, tok, mask_id)
        else:
            examples = prepare_examples(rows, tok, mask_id)

        ckpt_summary_rows = []
        for mode in modes:
            if args.scoring == "span_pll":
                summary_rows, detail_rows = evaluate_span_mode(
                    model=model,
                    examples=examples,
                    mode=mode,
                    batch_size=args.batch_size,
                    pad_id=pad_id,
                    device=args.device,
                    save_details=args.save_details,
                    score_norm=args.span_score_norm,
                )
            else:
                summary_rows, detail_rows = evaluate_mode(
                    model=model,
                    examples=examples,
                    mode=mode,
                    batch_size=args.batch_size,
                    pad_id=pad_id,
                    device=args.device,
                    save_details=args.save_details,
                )
            for row in summary_rows:
                row.update({
                    "label": label,
                    "ckpt": ckpt,
                    "dataset": str(args.dataset),
                    "scoring": args.scoring,
                    "span_score_norm": args.span_score_norm if args.scoring == "span_pll" else None,
                })
            for row in detail_rows:
                row.update({
                    "label": label,
                    "ckpt": ckpt,
                    "dataset": str(args.dataset),
                    "mode": mode,
                    "scoring": args.scoring,
                    "span_score_norm": args.span_score_norm if args.scoring == "span_pll" else None,
                })
            ckpt_summary_rows.extend(summary_rows)
            all_summary_rows.extend(summary_rows)
            all_detail_rows.extend(detail_rows)

        print_key_rows(label, ckpt_summary_rows)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with args.out.open("w", encoding="utf-8") as f:
        for row in all_summary_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"\nsaved summary: {args.out}")

    if args.detail_out:
        with args.detail_out.open("w", encoding="utf-8") as f:
            for row in all_detail_rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"saved details: {args.detail_out}")


if __name__ == "__main__":
    main()
