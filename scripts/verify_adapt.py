"""Rigorous correctness checks for AR→DLM adaptation primitives.

Each check returns True on success and False on failure. The script prints a
PASS/FAIL line per check and exits with non-zero status if any check fails.

Checks:
1. Causal model: prefix logits at position k are unaffected by tokens at
   positions > k. (Sanity: AR baseline is genuinely causal.)
2. Bidirectional patch: prefix logits at position k DEPEND on tokens at
   positions > k after :func:`patch_bidirectional_attention`.
3. Patch reversibility: bidirectional → causal round-trip yields identical
   logits to the original AR model. This is what enables interpretation B
   (mode switching at inference) downstream.
4. Mask token: tokenizer encodes ``[MASK]`` to a single id == mask_token_id.
5. Embedding resize: input AND output embeddings grow by exactly one row,
   and the new rows equal the mean of the pre-existing rows.
6. MDM training works: 50 AdamW steps on a fixed 2x32 batch reduce the loss
   by at least half, with consistent masking across steps.
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from unlearning_architecture.adapt import (
    add_mask_token,
    mdm_loss,
    patch_bidirectional_attention,
    unpatch_attention,
)

MODEL = "EleutherAI/pythia-160m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32


def fresh():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()
    return model, tok


def make_pair(vocab: int, seq: int, k: int, seed: int):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    a = torch.randint(0, vocab, (1, seq), device=DEVICE, generator=g)
    b = a.clone()
    b[:, k + 1 :] = torch.randint(0, vocab, (1, seq - k - 1), device=DEVICE, generator=g)
    return a, b


def check_1_causal_independence() -> bool:
    model, tok = fresh()
    a, b = make_pair(len(tok), seq=16, k=8, seed=0)
    with torch.no_grad():
        oa = model(a).logits[:, : 8 + 1]
        ob = model(b).logits[:, : 8 + 1]
    return torch.allclose(oa, ob, atol=1e-5)


def check_2_bidirectional_dependence() -> bool:
    model, tok = fresh()
    patch_bidirectional_attention(model)
    a, b = make_pair(len(tok), seq=16, k=8, seed=0)
    with torch.no_grad():
        oa = model(a).logits[:, : 8 + 1]
        ob = model(b).logits[:, : 8 + 1]
    return not torch.allclose(oa, ob, atol=1e-5)


def check_3_patch_reversible() -> bool:
    model, tok = fresh()
    a, _ = make_pair(len(tok), seq=16, k=8, seed=0)
    with torch.no_grad():
        before = model(a).logits.clone()
    patch_bidirectional_attention(model)
    unpatch_attention(model)
    with torch.no_grad():
        after = model(a).logits
    return torch.allclose(before, after, atol=1e-6)


def check_4_mask_token_roundtrip() -> bool:
    model, tok = fresh()
    mask_id = add_mask_token(model, tok)
    encoded = tok.encode("[MASK]", add_special_tokens=False)
    return len(encoded) == 1 and encoded[0] == mask_id == tok.mask_token_id


def check_5_embedding_resize() -> bool:
    """Contract: post-call, embedding rows == len(tokenizer); the row at
    mask_token_id equals the mean of the rows before it.

    Note: Pythia's pretrained embedding is over-allocated (50304 rows for a
    50277-vocab tokenizer, padded for GPU efficiency); resize_token_embeddings
    truncates to len(tokenizer), which is correct for our purposes. So we
    compare against len(tokenizer), not the pre-call embedding shape.
    """
    model, tok = fresh()
    n_tok_pre = len(tok)
    mask_id = add_mask_token(model, tok)
    n_tok_post = len(tok)
    in_w = model.get_input_embeddings().weight
    out_w = model.get_output_embeddings().weight
    if n_tok_post != n_tok_pre + 1:
        return False
    if in_w.shape[0] != n_tok_post or out_w.shape[0] != n_tok_post:
        return False
    if mask_id != n_tok_post - 1:
        return False
    new_in_match = torch.allclose(in_w[mask_id], in_w[:mask_id].mean(dim=0), atol=1e-3)
    new_out_match = torch.allclose(out_w[mask_id], out_w[:mask_id].mean(dim=0), atol=1e-3)
    return new_in_match and new_out_match


def check_6_mdm_loss_decreases() -> bool:
    model, tok = fresh()
    mask_id = add_mask_token(model, tok)
    patch_bidirectional_attention(model)
    model.train()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    bsz, seq = 2, 32
    g = torch.Generator(device=DEVICE).manual_seed(0)
    input_ids = torch.randint(0, len(tok) - 1, (bsz, seq), device=DEVICE, generator=g)
    attn = torch.ones(bsz, seq, dtype=torch.long, device=DEVICE)
    batch = {"input_ids": input_ids, "attention_mask": attn}

    losses = []
    for _ in range(50):
        torch.manual_seed(123)  # fix MDM masking pattern across steps
        loss = mdm_loss(model, batch, mask_id)
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(loss.item())
    print(f"      losses[0]={losses[0]:.3f} losses[-1]={losses[-1]:.3f}", flush=True)
    # plain CE starts ~log(vocab) ≈ 10.8; halving over 50 fixed-batch steps is
    # an easy bar that signals the loop is producing useful gradient updates.
    return losses[-1] < losses[0] * 0.5


CHECKS = [
    ("causal independence (prefix unaffected by suffix)", check_1_causal_independence),
    ("bidirectional dependence (prefix affected by suffix)", check_2_bidirectional_dependence),
    ("patch + unpatch reversibility (bit-equal logits)", check_3_patch_reversible),
    ("mask token round-trip ([MASK] -> single id)", check_4_mask_token_roundtrip),
    ("embedding resize + mean-of-rest init (input & output)", check_5_embedding_resize),
    ("MDM loss halves over 50 steps on fixed batch", check_6_mdm_loss_decreases),
]


def main() -> None:
    n_pass = 0
    for name, fn in CHECKS:
        try:
            ok = bool(fn())
            err = ""
        except Exception as e:
            ok = False
            err = f" (raised {type(e).__name__}: {e})"
        status = "PASS" if ok else "FAIL"
        n_pass += int(ok)
        print(f"[{status}] {name}{err}", flush=True)
    print(f"-- {n_pass}/{len(CHECKS)} passed", flush=True)
    if n_pass != len(CHECKS):
        sys.exit(1)


if __name__ == "__main__":
    main()
