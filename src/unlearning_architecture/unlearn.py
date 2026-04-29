"""Unlearn DLM behavior from a DLM-adapted ckpt.

Loss = alpha * NPO(forget) + (1 - alpha) * CE(retain).

  forget  : MDM-masked WikiText evaluated in bidirectional mode. NPO pushes
            the current model's probability of the correct masked-token
            prediction *below* the frozen reference (= the DLM-adapted ckpt
            itself at the start of unlearning). Stable cousin of gradient
            ascent (Zhang+ 2024).
  retain  : Standard causal-mode next-token CE on clean WikiText. Pulls the
            model back toward AR-style prediction.

A ``--selector`` flag controls which parameters receive gradient updates:

  all_params : load the adapted ckpt as a regular HF model (LoRA gets merged
               on the way in); every parameter is trainable. This is the
               "fully unrestricted" unlearning baseline.
  lora_only  : load the adapted ckpt as a PEFT-LoRA structure on top of the
               AR base; only LoRA adapter weights are trainable. Tests the
               hypothesis that the DLM-specific delta is structurally
               localized in the adapter and can be unlearned without
               disturbing the base.

Usage:
  uv run python -m unlearning_architecture.unlearn \
    --adapted_ckpt checkpoints/dlm-pythia160m-full \
    --selector all_params \
    --steps 1000 \
    --out checkpoints/unlearned-full-allparams \
    --wandb_project unlearning-architecture \
    --run_name unlearn-full-allparams-1k
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from unlearning_architecture.adapt import add_mask_token
from unlearning_architecture.data import relational_facts_loader, wikitext103_loader
from unlearning_architecture.eval import force_attention_mode, load_model_and_tokenizer


def load_for_unlearn(ckpt: str, selector: str, dtype, device: str, native_dlm: bool = False):
    """Load a target model whose trainable-parameter set is fixed by ``selector``.

    - ``all_params``  : load as a regular HF model (merging LoRA if any).
    - ``lora_only``   : require a PEFT-LoRA ckpt; preserve the adapter
                        structure so only the adapter is trainable.
    - ``delta_topk``  : load as a regular HF model; the gradient mask that
                        restricts updates to the top-k% of each parameter
                        by |adapted - ar_base| is installed *after* loading
                        via :func:`install_delta_topk_mask`.
    """
    tok = AutoTokenizer.from_pretrained(ckpt)
    adapter_cfg = Path(ckpt) / "adapter_config.json"
    is_lora = adapter_cfg.exists()

    if selector == "lora_only":
        if not is_lora:
            raise ValueError(f"selector='lora_only' requires a PEFT-LoRA ckpt; {ckpt} has none")
        from peft import PeftModel

        base_name = json.loads(adapter_cfg.read_text())["base_model_name_or_path"]
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        add_mask_token(base, tok)
        # is_trainable=True is required: PEFT's default is inference mode,
        # which freezes the adapter and yields an empty parameter list.
        model = PeftModel.from_pretrained(base, ckpt, is_trainable=True).to(device)
    elif selector in ("all_params", "delta_topk"):
        if native_dlm:
            from unlearning_architecture.native_dlm import load_native_dlm
            model, tok = load_native_dlm(ckpt, dtype, device)
        else:
            model, _ = load_model_and_tokenizer(ckpt, dtype, device)
        if selector == "all_params":
            for p in model.parameters():
                p.requires_grad_(True)
    else:
        raise ValueError(f"unknown selector: {selector!r}")
    return model, tok


def install_delta_topk_mask(
    model: torch.nn.Module,
    tokenizer,
    ar_base_name: str,
    top_pct: float,
    dtype,
    native_dlm: bool = False,
) -> tuple[int, int]:
    """Restrict gradient updates to the top-``top_pct`` fraction of each
    parameter, ranked by |adapted - ar_base|. The ar_base is loaded fresh,
    resized to match the adapted tokenizer's vocab, and used only to compute
    Δθ before being released. A backward hook on each parameter zeros the
    gradient outside the mask, so the optimizer's update reaches only
    high-Δθ positions.

    Returns (n_kept, n_total) for logging.
    """
    device = next(model.parameters()).device
    if native_dlm:
        from unlearning_architecture.native_dlm import load_native_dlm
        ar_base, _ = load_native_dlm(ar_base_name, dtype, str(device))
    else:
        ar_base = AutoModelForCausalLM.from_pretrained(ar_base_name, dtype=dtype)
        add_mask_token(ar_base, tokenizer)
        ar_base.to(device)
    ar_state = {n: p.data.clone() for n, p in ar_base.named_parameters()}
    del ar_base
    torch.cuda.empty_cache()

    n_kept, n_total = 0, 0
    for name, p in model.named_parameters():
        if name not in ar_state or ar_state[name].shape != p.shape:
            p.requires_grad_(False)
            n_total += p.numel()
            continue
        delta = (p.data - ar_state[name]).abs()
        flat = delta.flatten()
        k = max(1, int(flat.numel() * top_pct))
        if k >= flat.numel():
            mask = torch.ones_like(p)
        else:
            # Use exact top-k indices rather than thresholding by value.
            # LoRA-merged checkpoints have many exact-zero deltas; thresholding
            # would include every tied zero once the kth value is 0.
            idx = flat.topk(k, largest=True, sorted=False).indices
            mask_flat = torch.zeros_like(flat, dtype=p.dtype)
            mask_flat[idx] = 1
            mask = mask_flat.view_as(p)
        p.requires_grad_(True)
        p.register_hook(lambda g, m=mask: g * m)
        n_kept += int(mask.sum().item())
        n_total += mask.numel()
    del ar_state
    torch.cuda.empty_cache()
    return n_kept, n_total


def _mdm_masking(input_ids: torch.Tensor, attn_bool: torch.Tensor, mask_id: int):
    bsz, seq = input_ids.shape
    device = input_ids.device
    t = torch.rand(bsz, 1, device=device).clamp(min=1e-3)
    masked = (torch.rand(bsz, seq, device=device) < t.expand(-1, seq)) & attn_bool
    if not masked.any():
        masked[0, 0] = True
    corrupted = torch.where(masked, mask_id, input_ids)
    return corrupted, masked


def npo_forget_loss(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    batch: dict,
    mask_id: int,
    beta: float,
    mode: str = "bidirectional",
) -> torch.Tensor:
    """NPO on bidirectional MDM. Pushes p_theta(correct masked) below p_ref.

    ``mode='natural'`` skips the force_attention_mode call for native DLMs.
    """
    input_ids = batch["input_ids"]
    attn_long = batch["attention_mask"]
    corrupted, masked = _mdm_masking(input_ids, attn_long.bool(), mask_id)

    force_attention_mode(model, mode)
    force_attention_mode(ref_model, mode)

    logits = model(corrupted, attention_mask=attn_long).logits
    with torch.no_grad():
        ref_logits = ref_model(corrupted, attention_mask=attn_long).logits

    target = input_ids[masked]
    log_p_theta = -F.cross_entropy(logits[masked], target, reduction="none")
    log_p_ref = -F.cross_entropy(ref_logits[masked], target, reduction="none")
    diff = log_p_theta - log_p_ref
    return -(2.0 / beta) * F.logsigmoid(-beta * diff).mean()


def kl_to_ar_forget_loss(
    model: torch.nn.Module,
    ar_base_model: torch.nn.Module,
    batch: dict,
    mask_id: int,
    mode: str = "bidirectional",
) -> torch.Tensor:
    """Bounded forget via teacher-forcing KL on masked positions.

    PyTorch's kl_div(log_p_theta, p_ref) computes KL(p_ref || p_theta), so the
    AR-base bidirectional-mode distribution acts as a soft teacher. The forget
    term has its distribution-space optimum at AR-base behavior, though the
    full training objective also includes retain CE and finite-step optimizer
    effects.

    ``mode='natural'`` skips the force_attention_mode call for native DLMs.
    """
    input_ids = batch["input_ids"]
    attn_long = batch["attention_mask"]
    corrupted, masked = _mdm_masking(input_ids, attn_long.bool(), mask_id)

    force_attention_mode(model, mode)
    force_attention_mode(ar_base_model, mode)

    logits = model(corrupted, attention_mask=attn_long).logits
    with torch.no_grad():
        ref_logits = ar_base_model(corrupted, attention_mask=attn_long).logits

    log_p = torch.log_softmax(logits[masked].float(), dim=-1)
    p_ref = torch.softmax(ref_logits[masked].float(), dim=-1)
    return F.kl_div(log_p, p_ref, reduction="batchmean")


def retain_ce_loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    """Causal next-token CE on clean inputs. Pulls back toward AR behavior."""
    force_attention_mode(model, "causal")
    ids = batch["input_ids"]
    attn = batch["attention_mask"]
    logits = model(ids, attention_mask=attn).logits
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = ids[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels)


def retain_mdm_loss(model: torch.nn.Module, batch: dict, mask_id: int) -> torch.Tensor:
    """MDM-style retain for native DLMs that lack a causal mode.

    Same masking pattern as :func:`_mdm_masking` then plain CE on masked
    positions in the model's natural (bidirectional) attention mode.
    """
    input_ids = batch["input_ids"]
    attn_long = batch["attention_mask"]
    corrupted, masked = _mdm_masking(input_ids, attn_long.bool(), mask_id)
    logits = model(corrupted, attention_mask=attn_long).logits
    return F.cross_entropy(logits[masked], input_ids[masked])


def make_causal_attention_bias(seq_len: int, device: torch.device) -> torch.Tensor:
    """Lower-triangular causal bias for native DLM forward (LLaDA convention).

    Shape (1, 1, seq, seq), float; 0 on/below diagonal, -inf above.
    Mirrors modeling_llada.causal_attention_bias so it can be passed via the
    attention_bias kwarg of LLaDAModelLM.forward to override the default
    bidirectional bias for one call.
    """
    bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    bias.masked_fill_(bias == 1, torch.finfo(bias.dtype).min)
    return bias.view(1, 1, seq_len, seq_len)


def right_context_invariance_forget_loss(
    model: torch.nn.Module,
    batch: dict,
    mask_id: int,
    mode: str = "natural",
    layer_start: int = 0,
    n_positions: int = 1,
) -> torch.Tensor:
    """L_rs: residual-stream right-context invariance.

    Sample n_positions split positions i ~ U[1, L-1]. Compare hidden states
    at position i between the full input and a left-only version (right-of-i
    replaced by [MASK]). MSE on hidden_states[layer_start:], averaged over
    sampled positions and selected layers. At equilibrium the hidden state
    at position i no longer depends on the right context — the AR-equivalence
    condition in residual-stream space.

    The full forward is computed once (with grad) and reused; left-masked
    forwards are wrapped in torch.no_grad and detached so gradient flows only
    through the full-context branch (avoids the trivial zero-loss collapse).

    layer_start=0 keeps embedding + all 32 transformer layers (33 terms);
    layer_start=20 concentrates on deep content-routing layers.
    """
    input_ids = batch["input_ids"]
    attn_long = batch["attention_mask"]
    _, seq = input_ids.shape
    if seq < 2:
        return torch.zeros((), device=input_ids.device)

    k = min(n_positions, seq - 1)
    positions = (torch.randperm(seq - 1, device=input_ids.device)[:k] + 1).tolist()

    if mode != "natural":
        force_attention_mode(model, mode)

    out_full = model(
        input_ids, attention_mask=attn_long, output_hidden_states=True
    )
    hidden_full = out_full.hidden_states

    loss = torch.zeros((), device=input_ids.device)
    n_terms = 0
    for i in positions:
        left_masked = input_ids.clone()
        left_masked[:, i:] = mask_id
        with torch.no_grad():
            out_left = model(
                left_masked, attention_mask=attn_long, output_hidden_states=True
            )
        hidden_left = [h.detach() for h in out_left.hidden_states]
        for h_full, h_left in zip(
            hidden_full[layer_start:], hidden_left[layer_start:], strict=True
        ):
            diff = h_full[:, i, :] - h_left[:, i, :]
            loss = loss + (diff.float() ** 2).mean()
            n_terms += 1
    return loss / max(1, n_terms)


def retain_causal_ce_native_loss(
    model: torch.nn.Module,
    batch: dict,
) -> torch.Tensor:
    """Causal next-token CE for native DLMs via attention_bias injection.

    Pairs with right_context_invariance forget: pulls the model toward AR
    next-token prediction on the same data where forget pushes against
    bidirectional dependence.
    """
    input_ids = batch["input_ids"]
    attn_long = batch["attention_mask"]
    seq = input_ids.shape[1]
    causal_bias = make_causal_attention_bias(seq, input_ids.device)
    logits = model(
        input_ids, attention_mask=attn_long, attention_bias=causal_bias
    ).logits
    shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
    shift_labels = input_ids[:, 1:].reshape(-1)
    return F.cross_entropy(shift_logits, shift_labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapted_ckpt", required=True)
    ap.add_argument("--selector", choices=["all_params", "lora_only", "delta_topk"], default="all_params")
    ap.add_argument("--top_pct", type=float, default=0.10,
                    help="Fraction of each parameter kept trainable when selector=delta_topk")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--alpha", type=float, default=0.5, help="weight on forget loss")
    ap.add_argument("--beta", type=float, default=0.1, help="NPO temperature (forget_loss=npo only)")
    ap.add_argument(
        "--forget_loss",
        choices=["npo", "kl_to_ar", "right_context_invariance"],
        default="npo",
    )
    ap.add_argument(
        "--lrs_layer_start",
        type=int,
        default=0,
        help="L_rs: include only hidden_states[layer_start:]. Default 0 keeps "
             "embedding + all 32 transformer layers (v2 behavior). 20 concentrates "
             "on deep content-routing layers (LLaDA logit-lens emerges around L25).",
    )
    ap.add_argument(
        "--lrs_n_positions",
        type=int,
        default=1,
        help="L_rs: number of random split positions averaged per step. Each extra "
             "position adds one no_grad left-masked forward; the full forward is "
             "computed once and shared.",
    )
    ap.add_argument("--ar_base", default="EleutherAI/pythia-160m",
                    help="KL target ckpt when forget_loss=kl_to_ar. Usually the AR base "
                         "(forward direction). For reverse direction (AR→DLM), pass a "
                         "DLM-adapted ckpt (regular HF or PEFT-LoRA).")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save an intermediate ckpt every N steps under args.out/stepN/. "
             "0 disables intermediate saves; only the final ckpt at args.steps is written.",
    )
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--run_name", default="")
    ap.add_argument(
        "--data",
        choices=["wikitext", "relational"],
        default="wikitext",
        help="Training corpus for both forget (MDM/bidirectional) and retain "
             "(causal CE) sides. relational = data/relational_bidir_v1.",
    )
    ap.add_argument(
        "--relational_dataset",
        type=Path,
        default=Path("data/relational_bidir_v1"),
        help="Path to the relational dataset directory (used when --data=relational).",
    )
    ap.add_argument(
        "--native_dlm",
        action="store_true",
        help="Treat --adapted_ckpt and --ar_base as native bidirectional DLMs "
             "(LLaDA, ...). Skips force_attention_mode (no causal mode), uses "
             "MDM-style retain instead of causal-CE retain.",
    )
    ap.add_argument(
        "--use_8bit_optim",
        action="store_true",
        help="Use bitsandbytes AdamW8bit instead of torch.optim.AdamW. Saves ~3x "
             "optimizer-state VRAM at the cost of a small numeric drift. Required "
             "for 8B-class Full-FT Unlearning to fit on a single GPU.",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    model, tok = load_for_unlearn(
        args.adapted_ckpt, args.selector, torch.bfloat16, args.device,
        native_dlm=args.native_dlm,
    )
    if args.native_dlm:
        if tok.mask_token_id is None:
            raise ValueError("--native_dlm: tokenizer has no mask_token_id; cannot proceed")
    else:
        add_mask_token(model, tok)
    model.train()

    if args.selector == "delta_topk":
        n_kept, n_total = install_delta_topk_mask(
            model, tok, args.ar_base, args.top_pct, torch.bfloat16,
            native_dlm=args.native_dlm,
        )
        print(
            f"[unlearn] delta_topk mask: kept={n_kept:,}/{n_total:,} "
            f"({n_kept / n_total:.2%}, top_pct={args.top_pct})",
            flush=True,
        )

    # Reference model is loaded only when needed: NPO needs the frozen
    # adapted ckpt as ref; KL-to-AR-base needs the AR pretrained model as ref.
    # right_context_invariance is self-distillation (left-masked forward of the
    # same model) so it needs no ref. alpha=0 also skips loading.
    ref_model = None
    if args.alpha > 0 and args.forget_loss != "right_context_invariance":
        if args.forget_loss == "npo":
            ref_model, _ = load_for_unlearn(
                args.adapted_ckpt, args.selector, torch.bfloat16, args.device,
                native_dlm=args.native_dlm,
            )
            if not args.native_dlm:
                add_mask_token(ref_model, tok)
        else:  # kl_to_ar — use load_model_and_tokenizer so PEFT-LoRA targets work
            ref_model, _ = load_model_and_tokenizer(
                args.ar_base, torch.bfloat16, args.device,
                native_dlm=args.native_dlm,
            )
            if not args.native_dlm:
                add_mask_token(ref_model, tok)
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[unlearn] selector={args.selector} trainable_params={n_train:,}", flush=True)

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            config=vars(args) | {"phase": "unlearn", "trainable_params": n_train},
        )

    if args.data == "wikitext":
        loader = wikitext103_loader(tok, args.seq_len, args.batch_size, split="train")
    else:
        loader = relational_facts_loader(
            tok,
            args.relational_dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    if args.use_8bit_optim:
        import bitsandbytes as bnb

        optim = bnb.optim.AdamW8bit(trainable, lr=args.lr)
        print(f"[unlearn] using bitsandbytes AdamW8bit", flush=True)
    else:
        optim = torch.optim.AdamW(trainable, lr=args.lr)
    sched = get_cosine_schedule_with_warmup(optim, args.warmup, args.steps)

    args.out.mkdir(parents=True, exist_ok=True)
    iterator = iter(loader)
    step = 0
    while step < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = {k: v.to(args.device) for k, v in batch.items()}

        forget_mode = "natural" if args.native_dlm else "bidirectional"
        if args.alpha > 0:
            if args.forget_loss == "npo":
                l_forget = npo_forget_loss(model, ref_model, batch, tok.mask_token_id, args.beta, mode=forget_mode)
            elif args.forget_loss == "kl_to_ar":
                l_forget = kl_to_ar_forget_loss(model, ref_model, batch, tok.mask_token_id, mode=forget_mode)
            else:  # right_context_invariance
                l_forget = right_context_invariance_forget_loss(
                    model, batch, tok.mask_token_id, mode=forget_mode,
                    layer_start=args.lrs_layer_start,
                    n_positions=args.lrs_n_positions,
                )
        else:
            l_forget = torch.zeros((), device=args.device)
        if args.forget_loss == "right_context_invariance" and args.native_dlm:
            l_retain = retain_causal_ce_native_loss(model, batch)
        elif args.native_dlm:
            l_retain = retain_mdm_loss(model, batch, tok.mask_token_id)
        else:
            l_retain = retain_ce_loss(model, batch)
        loss = args.alpha * l_forget + (1 - args.alpha) * l_retain

        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            msg = (
                f"[unlearn] step={step} loss={loss.item():.4f} "
                f"forget={l_forget.item():.4f} retain={l_retain.item():.4f} "
                f"gnorm={gnorm.item():.2f}"
            )
            print(msg, flush=True)
            if use_wandb:
                wandb.log(
                    {
                        "loss": loss.item(),
                        "forget": l_forget.item(),
                        "retain": l_retain.item(),
                        "lr": sched.get_last_lr()[0],
                        "gnorm": gnorm.item(),
                    },
                    step=step,
                )
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            ckpt_dir = args.out / f"step{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"[unlearn] intermediate ckpt saved to {ckpt_dir}", flush=True)

    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    if use_wandb:
        wandb.finish()
    print(f"[unlearn] saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
