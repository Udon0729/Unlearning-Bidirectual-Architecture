"""Reverse-context prediction probe — the smoking-gun test for DLM-specific
bidirectional reasoning capability.

For each ckpt and each attention mode (causal / bidirectional), we:
  1. Stream WikiText-103 validation sequences of length ``seq_len``.
  2. Replace the first ``k_mask`` token positions with [MASK].
  3. Predict the original tokens at those positions.
  4. Report top-1 accuracy.

The first k positions can only be recovered using right-context information
(positions k..seq_len-1). Causal attention sees no right context; bidirectional
attention sees all of it. So:
  - AR baseline / DLM-adapted in causal mode: floor accuracy (no right ctx).
  - AR baseline in bidirectional mode: floor too — the model never trained on
    bidirectional inputs, so even with the architectural capability the weights
    cannot exploit right context.
  - DLM-adapted in bidirectional mode: must rise above the floor if the model
    actually learned to use right context.

A clear (causal vs bidirectional) gap on the adapted ckpt that does NOT appear
on the AR baseline is the cleanest evidence that DLM-specific capability was
acquired.
"""
from __future__ import annotations

import argparse

import torch
from transformers import AutoTokenizer  # noqa: F401  (kept for parity with eval.py imports)

from unlearning_architecture.adapt import add_mask_token
from unlearning_architecture.data import wikitext103_loader
from unlearning_architecture.eval import force_attention_mode, load_model_and_tokenizer


@torch.no_grad()
def reverse_accuracy(model, loader, n_batches: int, k_mask: int, device: str, mask_id: int) -> float:
    correct, total = 0, 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        gold = ids[:, :k_mask].clone()
        ids[:, :k_mask] = mask_id
        logits = model(ids, attention_mask=attn).logits
        preds = logits[:, :k_mask].argmax(dim=-1)
        correct += int((preds == gold).sum().item())
        total += int(gold.numel())
    return correct / total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=32)
    ap.add_argument("--k_mask", type=int, default=5)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.ckpt, torch.bfloat16, args.device)
    mask_id = add_mask_token(model, tok)
    model.eval()

    print(f"[reverse-probe] ckpt={args.ckpt} k_mask={args.k_mask} n_batches={args.n_batches}")
    for mode in ("causal", "bidirectional"):
        force_attention_mode(model, mode)
        loader = wikitext103_loader(tok, args.seq_len, args.batch_size, split="validation")
        acc = reverse_accuracy(model, loader, args.n_batches, args.k_mask, args.device, mask_id)
        print(f"  mode={mode:<14}  top-1 acc = {acc:.4f}")


if __name__ == "__main__":
    main()
