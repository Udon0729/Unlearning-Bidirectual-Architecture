"""Memorization-vs-attention test on reversal_v1 reverse queries.

For each reverse query "[MASK] is paired with Y." with target X, we score
cand@1 under three input ablations:

  full        : standard input — leading [MASK] (X slot), Y visible
  no_Y        : leading [MASK] kept, Y span replaced by mask tokens
  no_context  : only the leading [MASK]; everything to its right removed

If v3's reverse cand@1 (0.705) comes from attention to Y (right context):
  full         high
  no_Y         drops
  no_context   drops further (fall to prior)

If v3's cand@1 comes from weight-encoded entity association independent of
Y's surface form:
  full         high
  no_Y         stays high
  no_context   drops to prior

Scoring is first-token (predict the first token id of each candidate at the
leading [MASK] position). Mask is inserted as mask_id directly into
input_ids, bypassing the [MASK]-string-tokenization fragility of LLaDA's
custom tokenizer.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from unlearning_architecture.eval import load_model_and_tokenizer


def first_token_id_in_context(tokenizer, prefix: str, candidate: str) -> int:
    """Return the first token id of ``candidate`` when it follows ``prefix``."""
    text = prefix + candidate
    start = len(prefix)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    for token_id, (s, e) in zip(
        encoded["input_ids"], encoded["offset_mapping"], strict=True
    ):
        if e > start and s < len(text):
            return int(token_id)
    raise ValueError(f"could not locate candidate token: prefix={prefix!r} cand={candidate!r}")


def build_input_ids(row: dict, tokenizer, mask_id: int, regime: str) -> torch.Tensor:
    """Construct input_ids for one regime. Leading [MASK] is at position 0."""
    template = row["input"]              # "[MASK] is paired with Y."
    obj = row["object"]                  # Y entity string
    parts = template.split("[MASK]")
    assert len(parts) == 2, f"expected 1 [MASK] in {template!r}"
    suffix = parts[1]                    # " is paired with Y."

    y_pos = suffix.find(obj)
    if y_pos < 0:
        raise ValueError(f"Y={obj!r} not found in suffix={suffix!r}")
    pre_y = suffix[:y_pos]               # " is paired with "
    post_y = suffix[y_pos + len(obj):]   # "."

    if regime == "full":
        ids = [mask_id]
        ids += tokenizer.encode(pre_y, add_special_tokens=False)
        ids += tokenizer.encode(obj, add_special_tokens=False)
        ids += tokenizer.encode(post_y, add_special_tokens=False)
    elif regime == "no_Y":
        ids = [mask_id]
        ids += tokenizer.encode(pre_y, add_special_tokens=False)
        y_ids = tokenizer.encode(obj, add_special_tokens=False)
        ids += [mask_id] * len(y_ids)
        ids += tokenizer.encode(post_y, add_special_tokens=False)
    elif regime == "no_context":
        ids = [mask_id]
    else:
        raise ValueError(regime)

    return torch.tensor([ids], dtype=torch.long)


@torch.no_grad()
def score_row(
    model, tokenizer, mask_id: int, row: dict, regime: str, device: str
) -> tuple[bool, float]:
    """Return (is_top1_correct, log_p_target) for a single (row, regime)."""
    input_ids = build_input_ids(row, tokenizer, mask_id, regime).to(device)
    logits = model(input_ids).logits.float()
    log_probs = F.log_softmax(logits[0, 0], dim=-1)  # leading [MASK] at pos 0

    target_id = first_token_id_in_context(tokenizer, "", row["target"])
    cand_ids = [
        first_token_id_in_context(tokenizer, "", c) for c in row["candidates"]
    ]
    cand_scores = [log_probs[i].item() for i in cand_ids]
    pred_idx = max(range(len(cand_scores)), key=lambda j: cand_scores[j])
    is_correct = row["candidates"][pred_idx] == row["target"]
    return is_correct, log_probs[target_id].item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="label=ckpt")
    ap.add_argument(
        "--data", type=Path, default=Path("data/reversal_v1/eval_logic.jsonl")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("results/reversal_memorization_probe.jsonl")
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    rows = [
        json.loads(line)
        for line in args.data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    out_rows = []
    for item in args.ckpt:
        if "=" in item:
            label, ckpt = item.split("=", 1)
        else:
            label, ckpt = item, item
        print(f"\n[loading {label}]", flush=True)
        model, tok = load_model_and_tokenizer(
            ckpt, torch.bfloat16, args.device, native_dlm=True
        )
        if tok.mask_token_id is None:
            raise ValueError(f"{ckpt}: tokenizer.mask_token_id is None")
        model.eval()
        mask_id = tok.mask_token_id

        cand_correct: dict[str, int] = defaultdict(int)
        log_p_sum: dict[str, float] = defaultdict(float)
        n = 0
        for row in rows:
            for regime in ("full", "no_Y", "no_context"):
                ok, lp = score_row(model, tok, mask_id, row, regime, args.device)
                cand_correct[regime] += int(ok)
                log_p_sum[regime] += lp
            n += 1

        record = {"label": label, "ckpt": ckpt, "n_rows": n}
        for r in ("full", "no_Y", "no_context"):
            record[f"cand@1_{r}"] = cand_correct[r] / n
            record[f"log_p_target_{r}"] = log_p_sum[r] / n
        out_rows.append(record)

        print(f"[{label}]")
        for r in ("full", "no_Y", "no_context"):
            print(
                f"  {r:<11} cand@1={record[f'cand@1_{r}']:.4f} "
                f"log_p_target={record[f'log_p_target_{r}']:.4f}",
                flush=True,
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
