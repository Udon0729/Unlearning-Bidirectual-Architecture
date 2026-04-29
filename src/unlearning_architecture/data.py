"""Tokenized streaming loader for license-clean small corpora.

Default: WikiText-103 (CC BY-SA 3.0). Documents are tokenized, EOS-joined, and
packed into fixed-length windows. A second loader reads the synthetic
relational-bidirectional dataset (`data/relational_bidir_v1/train_facts.jsonl`)
for inverse-problem-targeted DLM adaptation and Unlearning.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset


class _PackedTokenStream(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, seq_len: int):
        self.hf = hf_dataset
        self.tok = tokenizer
        self.seq_len = seq_len

    def __iter__(self):
        buf: list[int] = []
        eos = self.tok.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define eos_token_id for packing.")
        for ex in self.hf:
            text = ex["text"]
            if not text.strip():
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                }


def wikitext103_loader(tokenizer, seq_len: int = 512, batch_size: int = 8, split: str = "train"):
    hf = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split=split,
        streaming=True,
    )
    ds = _PackedTokenStream(hf, tokenizer, seq_len)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)


class _JsonlFactStream(IterableDataset):
    """Pack the ``text`` field of a relational-bidirectional jsonl into
    fixed-length windows. Rows are loaded once into memory and shuffled per
    epoch so adjacent ``graph_id`` rows do not cluster inside a window.
    """

    def __init__(self, path: Path, tokenizer, seq_len: int, shuffle: bool, seed: int):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.shuffle = shuffle
        self.rng = random.Random(seed)
        with path.open(encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f if line.strip()]

    def __iter__(self):
        eos = self.tok.eos_token_id
        if eos is None:
            raise ValueError("Tokenizer must define eos_token_id for packing.")
        rows = list(self.rows)
        if self.shuffle:
            self.rng.shuffle(rows)
        buf: list[int] = []
        for row in rows:
            text = row.get("text")
            if not text:
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            buf.append(eos)
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                }


def relational_facts_loader(
    tokenizer,
    dataset_dir: str | Path,
    seq_len: int = 512,
    batch_size: int = 8,
    file: str = "train_facts.jsonl",
    shuffle: bool = True,
    seed: int = 0,
):
    """Loader over the synthetic relational dataset's ``train_facts.jsonl``.

    Each row's ``text`` field is the complete factual sentence (no [MASK]).
    MDM masking is applied at training time by ``mdm_loss`` exactly as for the
    WikiText loader, so this is a drop-in replacement.
    """
    path = Path(dataset_dir) / file
    if not path.exists():
        raise FileNotFoundError(path)
    ds = _JsonlFactStream(path, tokenizer, seq_len, shuffle=shuffle, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=0)
