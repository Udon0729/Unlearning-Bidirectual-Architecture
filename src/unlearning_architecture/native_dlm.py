"""Compatibility shims for native diffusion language models (LLaDA, Dream, ...).

Native DLMs are bidirectional by construction; their custom HF code often
predates current ``transformers`` API expectations (``all_tied_weights_keys``,
``tie_weights(missing_keys=...)``, ``config.use_cache``, ...). This module
applies minimal patches so the same probe / fine-tune / unlearn pipeline can
load them without forking ``adapt.py`` or ``unlearn.py`` per architecture.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.dynamic_module_utils import get_class_from_dynamic_module


def apply_llada_compat_patches(model_name: str = "GSAI-ML/LLaDA-8B-Base") -> None:
    """Inject the class attributes ``LLaDAModelLM`` is missing under modern
    transformers releases. Idempotent. Call before ``from_pretrained``.
    """
    cls = get_class_from_dynamic_module("modeling_llada.LLaDAModelLM", model_name)
    if not hasattr(cls, "all_tied_weights_keys"):
        cls.all_tied_weights_keys = {}
    # ``tie_weights`` is invoked by recent transformers with kwargs the LLaDA
    # implementation does not declare (``missing_keys``, ``recompute_mapping``,
    # ...). Wrap to drop them.
    if not getattr(cls.tie_weights, "_drops_kwargs", False):
        original = cls.tie_weights

        def patched(self, *args, **kwargs):
            return original(self)

        patched._drops_kwargs = True
        cls.tie_weights = patched


_CONFIG_DEFAULTS = {
    "use_cache": False,
    "output_attentions": False,
    "output_hidden_states": False,
    "return_dict": True,
}


def load_native_dlm(ckpt: str, dtype, device: str):
    """Load a native bidirectional DLM with the patches required by current
    transformers. Returns ``(model, tokenizer)``.

    For now only LLaDA-style ckpts are recognised; extend the dispatch when a
    second native architecture is added.

    LLaDA stores its mask-token id on ``model.config.mask_token_id`` (e.g. 126336
    for ``<|mdm_mask|>`` on LLaDA-8B-Base) but does not register the token under
    ``tokenizer.mask_token``. We propagate it onto the tokenizer so downstream
    code can use ``tokenizer.mask_token_id`` uniformly.
    """
    if "llada" in ckpt.lower():
        apply_llada_compat_patches(ckpt)
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, dtype=dtype, trust_remote_code=True
    ).to(device)
    for attr, default in _CONFIG_DEFAULTS.items():
        if not hasattr(model.config, attr):
            setattr(model.config, attr, default)
    cfg_mask_id = getattr(model.config, "mask_token_id", None)
    if tokenizer.mask_token_id is None and cfg_mask_id is not None:
        mask_str = tokenizer.convert_ids_to_tokens(cfg_mask_id)
        tokenizer.mask_token = mask_str
    return model, tokenizer
