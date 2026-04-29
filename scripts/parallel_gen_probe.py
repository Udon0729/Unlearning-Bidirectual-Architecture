"""Position-conditional log-prob asymmetry test.

For each position i in a WikiText-103 sequence, we compute log p(token_i)
under three context-ablation regimes:
  full       : only position i is masked (bidirectional reference)
  left_only  : positions [i, seq) are all masked (AR-equivalent context)
  right_only : positions [0, i] are all masked (anti-AR context)

A pure DLM reads both directions; left_only ~ right_only ~ full.
A pure AR reads only left; left_only ~ full, right_only collapses to a
context-free prior. The (left_only - right_only) gap quantifies how
left-dependent the model's predictions are at each position.

Run on WikiText-103 validation (or any clean text), per ckpt, in LLaDA's
natural attention mode (no mask injection — we only manipulate inputs).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from unlearning_architecture.data import wikitext103_loader
from unlearning_architecture.eval import load_model_and_tokenizer


@torch.no_grad()
def score_position(
    model,
    input_ids: torch.Tensor,
    mask_id: int,
    i: int,
    regime: str,
) -> torch.Tensor:
    """Return log p(orig token at i) of shape (B,) for one regime."""
    bsz, seq = input_ids.shape
    masked = input_ids.clone()
    if regime == "full":
        masked[:, i] = mask_id
    elif regime == "left_only":
        masked[:, i:] = mask_id
    elif regime == "right_only":
        masked[:, : i + 1] = mask_id
    else:
        raise ValueError(regime)
    logits = model(masked).logits.float()  # (B, S, V)
    log_probs = F.log_softmax(logits[:, i, :], dim=-1)  # (B, V)
    target = input_ids[:, i]
    return log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def evaluate(model, loader, mask_id: int, positions: list[int], n_batches: int, device: str):
    sums = defaultdict(float)
    counts = defaultdict(int)
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        ids = batch["input_ids"].to(device)
        for regime in ("full", "left_only", "right_only"):
            for i in positions:
                lp = score_position(model, ids, mask_id, i, regime)
                sums[regime] += float(lp.sum().item())
                counts[regime] += int(lp.numel())
    return {k: sums[k] / counts[k] for k in sums}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="label=ckpt")
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=16)
    ap.add_argument(
        "--positions",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 80, 96, 112],
        help="positions to test (avoid extreme edges)",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/parallel_gen_probe.jsonl"))
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []

    for item in args.ckpt:
        if "=" in item:
            label, ckpt = item.split("=", 1)
        else:
            label, ckpt = ckpt, ckpt  # noqa
            label = ckpt
        model, tok = load_model_and_tokenizer(
            ckpt, torch.bfloat16, args.device, native_dlm=True
        )
        if tok.mask_token_id is None:
            raise ValueError(f"{ckpt}: tok.mask_token_id is None")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model.eval()

        loader = wikitext103_loader(tok, args.seq_len, args.batch_size, split="validation")
        means = evaluate(
            model, loader, tok.mask_token_id, args.positions, args.n_batches, args.device
        )
        gap = means["left_only"] - means["right_only"]
        retention = means["left_only"] - means["full"]
        row = {
            "label": label,
            "ckpt": ckpt,
            "n_positions": len(args.positions),
            "n_batches_eval": args.n_batches,
            "log_p_full": means["full"],
            "log_p_left_only": means["left_only"],
            "log_p_right_only": means["right_only"],
            "ar_gap": gap,
            "left_vs_full_drop": retention,
        }
        out_rows.append(row)
        print(
            f"\n[{label}]\n"
            f"  log_p_full       = {means['full']:.4f}\n"
            f"  log_p_left_only  = {means['left_only']:.4f}\n"
            f"  log_p_right_only = {means['right_only']:.4f}\n"
            f"  ar_gap (L−R)     = {gap:.4f}    (>0 = AR-like asymmetry)\n"
            f"  left vs full     = {retention:.4f} (≈0 = preserves AR mode)",
            flush=True,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with args.out.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"\nsaved: {args.out}")


if __name__ == "__main__":
    main()
