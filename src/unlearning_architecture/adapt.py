"""DiffuLLaMA-style MDM adaptation of an AR model.

Reference: Gong et al., "Scaling Diffusion Language Models via Adaptation
from Autoregressive Models" (arXiv:2410.17891).

Adaptation steps applied here:
- Continue training from an AR-pretrained checkpoint.
- Replace the causal attention bias / is_causal flag with bidirectional.
- Add a [MASK] token and resize the embedding matrix.
- Train with a masked-token objective. The default is the stable plain CE
  objective used in the current experiments; LLaDA-style 1/t-weighted NELBO
  variants are available for ablation.

The output of this script is the "DLM-adapted" checkpoint that downstream
unlearning experiments will operate on. Two variants can be produced:
- Full fine-tuning (default): all parameters update.
- LoRA (``--use_lora``): updates are confined to LoRA adapters; intended as a
  comparison baseline where the DLM-specific delta is structurally localized.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from unlearning_architecture.data import relational_facts_loader, wikitext103_loader


def set_config_causal(model: torch.nn.Module, is_causal: bool) -> int:
    """Set model-level causal-mask routing used by modern Transformers.

    Recent decoder implementations build the 4D attention mask from
    ``model.config.is_causal`` before individual attention modules run.  The
    per-module flags/buffers below are still needed for older backends, but
    they are not sufficient by themselves.
    """
    config = getattr(model, "config", None)
    if config is None:
        return 0
    config.is_causal = is_causal
    return 1


def patch_bidirectional_attention(model: torch.nn.Module) -> int:
    """Disable causal masking across all attention modules.

    Handles both legacy GPTNeoX (lower-triangular ``bias`` buffer) and modern
    HF attention (``is_causal`` attribute used with SDPA / flash backends).
    The original state is stashed on each module so :func:`unpatch_attention`
    can restore exact AR behavior. Idempotent: a second call is a no-op.
    Returns the number of modules touched.
    """
    config = getattr(model, "config", None)
    if config is not None and not hasattr(model, "_orig_config_has_is_causal"):
        model._orig_config_has_is_causal = hasattr(config, "is_causal")
        model._orig_config_is_causal = getattr(config, "is_causal", None)
    n = set_config_causal(model, False)
    for module in model.modules():
        if hasattr(module, "is_causal") and isinstance(getattr(module, "is_causal"), bool):
            if not hasattr(module, "_orig_is_causal"):
                module._orig_is_causal = module.is_causal
            module.is_causal = False
            n += 1
        bias = getattr(module, "bias", None)
        if isinstance(bias, torch.Tensor) and bias.dtype == torch.bool and bias.dim() == 4:
            if not hasattr(module, "_orig_attn_bias"):
                module._orig_attn_bias = bias.clone()
            module.bias = torch.ones_like(bias, dtype=torch.bool)
            n += 1
    return n


def unpatch_attention(model: torch.nn.Module) -> int:
    """Inverse of :func:`patch_bidirectional_attention`. Restores the original
    causal mask buffer / ``is_causal`` flag on every module that was patched.
    """
    n = 0
    config = getattr(model, "config", None)
    if config is not None and hasattr(model, "_orig_config_has_is_causal"):
        if model._orig_config_has_is_causal:
            config.is_causal = model._orig_config_is_causal
        elif hasattr(config, "is_causal"):
            delattr(config, "is_causal")
        del model._orig_config_has_is_causal
        del model._orig_config_is_causal
        n += 1
    for module in model.modules():
        if hasattr(module, "_orig_is_causal"):
            module.is_causal = module._orig_is_causal
            del module._orig_is_causal
            n += 1
        if hasattr(module, "_orig_attn_bias"):
            module.bias = module._orig_attn_bias
            del module._orig_attn_bias
            n += 1
    return n


def add_mask_token(model: torch.nn.Module, tokenizer) -> int:
    """Add a [MASK] token (if absent), resize input AND output embeddings to
    match ``len(tokenizer)``, and initialize the new row in each as the mean
    of the pre-existing rows. Idempotent in three cases:

    - tokenizer no mask, model not resized      -> add + resize + init
    - tokenizer has mask, model not yet resized -> resize + init only
                                                   (typical for LoRA reload
                                                    where the base model is
                                                    loaded fresh)
    - tokenizer has mask, model already resized -> no-op
    """
    if tokenizer.mask_token is None:
        tokenizer.add_special_tokens({"mask_token": "[MASK]"})
    target = len(tokenizer)
    if model.get_input_embeddings().weight.shape[0] != target:
        model.resize_token_embeddings(target)
        with torch.no_grad():
            emb_in = model.get_input_embeddings().weight
            emb_in[-1] = emb_in[:-1].mean(dim=0)
            out = model.get_output_embeddings()
            if out is not None and out.weight is not emb_in:
                out.weight[-1] = out.weight[:-1].mean(dim=0)
    return tokenizer.mask_token_id


def mdm_loss(
    model: torch.nn.Module,
    batch: dict,
    mask_token_id: int,
    objective: str = "plain_ce",
    t_min: float = 1e-3,
    nelbo_weight_clip: float | None = None,
) -> torch.Tensor:
    """One MDM step.

    Sample t ~ U[1e-3, 1] per sequence, mask each non-pad token i.i.d. with
    probability t, and predict the masked positions.

    objective:
      - plain_ce: stable BERT/RoBERTa-style CE averaged over masked positions.
      - llada_nelbo: LLaDA-style 1/t-weighted CE, normalized by token count.
      - llada_nelbo_clipped: same as llada_nelbo, but clips 1/t to reduce
        rare low-t gradient spikes.

    The 1/t weighting is mathematically aligned with masked diffusion NELBOs,
    but earlier runs showed unstable low-t gradients. Keeping plain_ce as the
    default preserves the current experimental baseline while enabling a direct
    objective ablation.
    """
    if objective not in {"plain_ce", "llada_nelbo", "llada_nelbo_clipped"}:
        raise ValueError(f"unknown MDM objective: {objective!r}")
    if objective == "llada_nelbo_clipped" and nelbo_weight_clip is None:
        raise ValueError("nelbo_weight_clip must be set for llada_nelbo_clipped")

    input_ids = batch["input_ids"]
    attn = batch["attention_mask"].bool()
    bsz, seq = input_ids.shape
    device = input_ids.device

    t = torch.rand(bsz, 1, device=device).clamp(min=t_min)
    masked = (torch.rand(bsz, seq, device=device) < t.expand(-1, seq)) & attn
    if not masked.any():
        masked[0, 0] = True

    corrupted = torch.where(masked, mask_token_id, input_ids)
    logits = model(corrupted, attention_mask=batch["attention_mask"]).logits
    ce = F.cross_entropy(logits[masked], input_ids[masked], reduction="none")

    if objective == "plain_ce":
        return ce.mean()

    batch_idx = masked.nonzero(as_tuple=True)[0]
    weight = 1.0 / t.squeeze(1)[batch_idx]
    if objective == "llada_nelbo_clipped":
        weight = weight.clamp(max=nelbo_weight_clip)
    denom = attn.sum().clamp_min(1)
    return (ce * weight).sum() / denom


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("checkpoints/dlm-pythia160m"))
    ap.add_argument(
        "--objective",
        choices=["plain_ce", "llada_nelbo", "llada_nelbo_clipped"],
        default="plain_ce",
        help="Masked-token training objective. plain_ce preserves existing runs; "
             "llada_nelbo adds 1/t weighting; llada_nelbo_clipped clips 1/t.",
    )
    ap.add_argument("--t_min", type=float, default=1e-3)
    ap.add_argument(
        "--nelbo_weight_clip",
        type=float,
        default=100.0,
        help="Max 1/t weight for --objective llada_nelbo_clipped.",
    )
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument(
        "--lora_target_modules", nargs="+",
        default=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        help="LoRA target module names. Default fits GPTNeoX/Pythia. "
             "For LLaMA-family use: q_proj k_proj v_proj o_proj gate_proj up_proj down_proj.",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable PyTorch deterministic algorithms (warn_only=True) plus "
             "cudnn.deterministic=True / benchmark=False. Set the env var "
             "CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) when launching, since "
             "it must be set before CUDA context init. Slower than default but "
             "produces reproducible loss trajectories across runs.",
    )
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument(
        "--save_every",
        type=int,
        default=0,
        help="Save an intermediate ckpt every N steps under args.out/stepN/. "
             "0 disables intermediate saves; only the final ckpt at args.steps is written.",
    )
    ap.add_argument("--wandb_project", default="", help="empty disables W&B")
    ap.add_argument("--run_name", default="")
    ap.add_argument(
        "--data",
        choices=["wikitext", "relational"],
        default="wikitext",
        help="Training corpus. wikitext = WikiText-103; "
             "relational = data/relational_bidir_v1 (inverse-problem-targeted).",
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
        help="Treat --model as a native bidirectional DLM (LLaDA, ...). Skips "
             "patch_bidirectional_attention (already bidirectional) and "
             "add_mask_token (uses tokenizer.mask_token_id directly).",
    )
    ap.add_argument(
        "--ar_mode",
        action="store_true",
        help="Plain AR (causal next-token) fine-tune. Skips bidirectional patch "
             "and [MASK] token; uses standard causal CE on the text. Intended "
             "for confirming reversal-curse baselines on AR models.",
    )
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Flash / mem-efficient SDP backends use atomicAdd in their backward
        # passes and are non-deterministic. Force the math backend instead.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.manual_seed(args.seed)

    if args.native_dlm and args.ar_mode:
        raise ValueError("--native_dlm and --ar_mode are mutually exclusive")
    if args.native_dlm:
        from unlearning_architecture.native_dlm import load_native_dlm

        model, tokenizer = load_native_dlm(args.model, torch.bfloat16, "cpu")
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError(
                f"--native_dlm requires the tokenizer (or model.config) to define "
                f"a mask token; {args.model} reports None"
            )
        print(f"[adapt] native_dlm mask_token_id={mask_id} (no bidirectional patch)", flush=True)
    elif args.ar_mode:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        mask_id = None
        print(f"[adapt] ar_mode (no bidirectional patch, no [MASK] token, causal CE)", flush=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
        mask_id = add_mask_token(model, tokenizer)
        n_patched = patch_bidirectional_attention(model)
        print(f"[adapt] mask_token_id={mask_id} patched_modules={n_patched}", flush=True)

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.run_name or None,
            config=vars(args) | {"mode": "lora" if args.use_lora else "full"},
        )

    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
        model.print_trainable_parameters()

    model.to(args.device)
    model.train()

    if args.data == "wikitext":
        loader = wikitext103_loader(
            tokenizer, seq_len=args.seq_len, batch_size=args.batch_size, split="train"
        )
    else:
        loader = relational_facts_loader(
            tokenizer,
            args.relational_dataset,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
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
        if args.ar_mode:
            ids = batch["input_ids"]
            attn = batch["attention_mask"]
            logits = model(ids, attention_mask=attn).logits
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels)
        else:
            loss = mdm_loss(
                model,
                batch,
                mask_id,
                objective=args.objective,
                t_min=args.t_min,
                nelbo_weight_clip=args.nelbo_weight_clip,
            )
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            print(f"[adapt] step={step} loss={loss.item():.4f} gnorm={gnorm.item():.2f}", flush=True)
            if use_wandb:
                wandb.log({"loss": loss.item(), "lr": sched.get_last_lr()[0], "gnorm": gnorm.item()}, step=step)
        if args.save_every and step % args.save_every == 0 and step < args.steps:
            ckpt_dir = args.out / f"step{step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[adapt] intermediate ckpt saved to {ckpt_dir}", flush=True)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"[adapt] saved to {args.out}", flush=True)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
