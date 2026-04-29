"""Phase 2.5: Force causal attention bias on LLaDA at inference.

LLaDA's ``LLaDAModelLM.forward`` accepts an ``attention_bias`` kwarg that, when
provided, overrides the default bidirectional bias built inside the model. We
inject the lower-triangular causal bias defined in ``modeling_llada`` itself so
the same fine-tuned ckpt is evaluated under AR-equivalent attention. No
re-training, no architecture change.

Used to obtain the same-architecture "forced AR" reference for Phase 3 v2.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from unlearning_architecture.eval import load_model_and_tokenizer
from scripts.relational_bidir_probe import (
    finalize_span_accumulator,
    init_span_accumulator,
    iter_jsonl,
    make_span_batch,
    prepare_span_examples,
    row_group_keys,
    span_candidate_metrics,
    update_span_accumulator,
)


def make_causal_bias(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal bias matching LLaDA's ``causal_attention_bias``.

    Shape ``(1, 1, seq, seq)``, dtype float; 0 on/below diagonal, -inf above.
    """
    bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    bias.masked_fill_(bias == 1, torch.finfo(bias.dtype).min)
    return bias.view(1, 1, seq_len, seq_len)


@torch.no_grad()
def evaluate(
    model,
    examples,
    batch_size: int,
    pad_id: int,
    device: str,
    mode: str,
):
    model.eval()
    accumulators: dict[tuple[str, str], dict] = defaultdict(init_span_accumulator)
    score_maps: list[dict[str, float]] = [dict() for _ in examples]

    jobs = []
    for ex_i, ex in enumerate(examples):
        for candidate in ex.score_candidates:
            jobs.append((ex_i, candidate))

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start : start + batch_size]
        batch_candidates = [c for _, c in batch_jobs]
        ids, attn = make_span_batch(batch_candidates, pad_id, device)
        seq_len = ids.shape[1]
        if mode == "causal":
            attention_bias = make_causal_bias(seq_len, ids.device)
            logits = model(
                ids, attention_mask=attn, attention_bias=attention_bias
            ).logits.float()
        else:
            logits = model(ids, attention_mask=attn).logits.float()
        for i, (ex_i, candidate) in enumerate(batch_jobs):
            pos = torch.tensor(candidate.label_positions, dtype=torch.long, device=device)
            labels = torch.tensor(candidate.label_ids, dtype=torch.long, device=device)
            log_probs = F.log_softmax(logits[i, pos], dim=-1)
            token_scores = log_probs[torch.arange(labels.numel(), device=device), labels]
            score = float(token_scores.mean().item())
            score_maps[ex_i][candidate.text] = score

    for ex, scores in zip(examples, score_maps, strict=True):
        metric = span_candidate_metrics(ex, scores)
        for key in row_group_keys(ex.row):
            update_span_accumulator(accumulators[key], metric)

    summary = []
    for (g, v), acc in sorted(accumulators.items()):
        row = {"mode": mode, "group": g, "value": v}
        row.update(finalize_span_accumulator(acc))
        summary.append(row)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", type=Path, default=Path("data/reversal_v1"))
    ap.add_argument("--splits", nargs="+", default=["eval_memory", "eval_logic"])
    ap.add_argument("--mode", nargs="+", default=["natural", "causal"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/phase25_llada_causal.jsonl"))
    args = ap.parse_args()

    rows = []
    for split in args.splits:
        path = args.dataset / f"{split}.jsonl"
        rows.extend(iter_jsonl(path))

    model, tok = load_model_and_tokenizer(
        args.ckpt, torch.bfloat16, args.device, native_dlm=True
    )
    mask_id = tok.mask_token_id
    if mask_id is None:
        raise ValueError(f"{args.ckpt}: tokenizer.mask_token_id is None")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    examples = prepare_span_examples(rows, tok, mask_id)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    all_summary = []
    for mode in args.mode:
        summary = evaluate(model, examples, args.batch_size, pad_id, args.device, mode)
        for row in summary:
            row["ckpt"] = args.ckpt
            row["dataset"] = str(args.dataset)
        all_summary.extend(summary)

        print(f"\n[phase2.5] mode={mode}")
        for row in summary:
            if row["group"] == "split" and row["value"] in ("eval_memory", "eval_logic"):
                print(
                    f"  {row['value']:<14} cand@1={row['candidate_top1']:.4f} "
                    f"rank={row['mean_candidate_rank']:.2f} n={row['n']}",
                    flush=True,
                )

    with args.out.open("w", encoding="utf-8") as f:
        for row in all_summary:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
