"""Evaluate AR-mode PPL and DLM-mode NELBO on WikiText-103 validation.

Both modes operate on the same model state; only the attention configuration
is toggled at inference (interpretation B from the project notes). This makes
the AR-base and DLM-adapted checkpoints comparable on equal footing.

Usage:
  uv run python -m unlearning_architecture.eval --ckpt EleutherAI/pythia-160m
  uv run python -m unlearning_architecture.eval --ckpt checkpoints/dlm-pythia160m-full
"""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_architecture.adapt import add_mask_token, set_config_causal
from unlearning_architecture.data import wikitext103_loader


def force_attention_mode(model: torch.nn.Module, mode: str) -> int:
    """Idempotently set every attention module to ``causal`` or ``bidirectional``,
    regardless of the current state. Reconstructs the bias buffer from scratch
    so it works correctly even after a load_pretrained where ``_orig_*`` stashes
    from :func:`unlearning_architecture.adapt.patch_bidirectional_attention`
    would not have been preserved.

    ``mode == "natural"`` is a no-op intended for native DLMs (LLaDA, ...) whose
    architecture has no causal-mask concept to toggle.
    """
    if mode == "natural":
        return 0
    assert mode in {"causal", "bidirectional"}, mode
    n = set_config_causal(model, mode == "causal")
    for m in model.modules():
        if hasattr(m, "is_causal") and isinstance(getattr(m, "is_causal"), bool):
            m.is_causal = mode == "causal"
            n += 1
        bias = getattr(m, "bias", None)
        if isinstance(bias, torch.Tensor) and bias.dtype == torch.bool and bias.dim() == 4:
            size = bias.shape[-1]
            if mode == "causal":
                tri = torch.tril(torch.ones(size, size, dtype=torch.bool, device=bias.device))
                m.bias = tri.view(1, 1, size, size)
            else:
                m.bias = torch.ones(1, 1, size, size, dtype=torch.bool, device=bias.device)
            n += 1
    return n


@torch.no_grad()
def ar_ppl(model, loader, n_batches: int, device: str) -> float:
    """Standard causal next-token PPL = exp(mean CE) over the held-out batches."""
    total_ce, total_tok = 0.0, 0
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        logits = model(ids, attention_mask=attn).logits.float()
        shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
        shift_labels = ids[:, 1:].reshape(-1)
        ce = F.cross_entropy(shift_logits, shift_labels, reduction="sum")
        total_ce += ce.item()
        total_tok += shift_labels.numel()
    return math.exp(total_ce / total_tok)


@torch.no_grad()
def dlm_nelbo(model, loader, n_batches: int, device: str, mask_id: int, n_t: int = 4) -> float:
    """Per-masked-token NELBO; averaged over ``n_t`` random t draws per batch."""
    total, count = 0.0, 0
    model.eval()
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        ids = batch["input_ids"].to(device)
        attn_long = batch["attention_mask"].to(device)
        attn_bool = attn_long.bool()
        for _ in range(n_t):
            t = torch.rand(ids.size(0), 1, device=device).clamp(min=1e-3)
            masked = (torch.rand_like(ids, dtype=torch.float) < t.expand_as(ids)) & attn_bool
            if not masked.any():
                masked[0, 0] = True
            corrupted = torch.where(masked, mask_id, ids)
            logits = model(corrupted, attention_mask=attn_long).logits.float()
            target = ids[masked]
            pred = logits[masked]
            ce = F.cross_entropy(pred, target, reduction="none")
            batch_idx = masked.nonzero(as_tuple=True)[0]
            weight = 1.0 / t.squeeze(1)[batch_idx]
            total += (ce * weight).sum().item()
            count += int(masked.sum().item())
    return total / count


def load_model_and_tokenizer(ckpt: str, dtype, device, native_dlm: bool = False):
    """Load HF or PEFT-LoRA checkpoint into a single regular ``CausalLM``.

    For LoRA: pulls ``base_model_name_or_path`` out of ``adapter_config.json``,
    re-applies the [MASK]-token resize on the base, attaches the adapter, and
    merges back into a plain HF model so downstream code is uniform.

    When ``native_dlm=True`` the ckpt is treated as a native bidirectional DLM
    (LLaDA, ...) — apply compat patches and skip the [MASK]-token resize, since
    such models ship their own mask-token convention.
    """
    if native_dlm:
        from unlearning_architecture.native_dlm import load_native_dlm

        return load_native_dlm(ckpt, dtype, device)

    import json
    from pathlib import Path

    tok = AutoTokenizer.from_pretrained(ckpt)
    adapter_cfg = Path(ckpt) / "adapter_config.json"
    if adapter_cfg.exists():
        from peft import PeftModel

        base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        add_mask_token(base, tok)
        model = PeftModel.from_pretrained(base, ckpt).merge_and_unload().to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(ckpt, dtype=dtype).to(device)
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF id or local path (full ckpt or PEFT-LoRA dir)")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--n_batches", type=int, default=64)
    ap.add_argument("--n_t", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, tok = load_model_and_tokenizer(args.ckpt, torch.bfloat16, args.device)

    # Ensure [MASK] exists (no-op if already present, e.g. reloaded full ckpt).
    mask_id = add_mask_token(model, tok)

    loader_ar = wikitext103_loader(tok, args.seq_len, args.batch_size, split="validation")
    force_attention_mode(model, "causal")
    ppl = ar_ppl(model, loader_ar, args.n_batches, args.device)

    loader_dlm = wikitext103_loader(tok, args.seq_len, args.batch_size, split="validation")
    force_attention_mode(model, "bidirectional")
    nelbo = dlm_nelbo(model, loader_dlm, args.n_batches, args.device, mask_id, args.n_t)

    print(f"[eval] ckpt={args.ckpt}")
    print(f"[eval] AR-mode PPL    = {ppl:.3f}    (lower = better AR retention)")
    print(f"[eval] DLM-mode NELBO = {nelbo:.3f}  (lower = better DLM acquisition)")


if __name__ == "__main__":
    main()
