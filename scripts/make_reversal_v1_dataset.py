"""Reversal-curse-targeted fine-tune dataset.

Each unique pair (X, Y) appears in training as a *forward* fact only:
  "X is paired with Y."  (and 4 other paraphrases of the same forward direction)

Eval queries the same pairs in BOTH directions:
  eval_memory.jsonl: "X is paired with [MASK]." -> answer Y  (forward, memorization)
  eval_logic.jsonl:  "[MASK] is paired with Y." -> answer X  (reverse, curse test)

Reverse facts are NEVER in train. AR fine-tuned on this dataset memorizes the
forward direction but should fail the reverse direction (reversal curse,
Berglund et al. 2023). MDM training on the same data implicitly masks both
positions, so a DLM should learn both directions; that gap is the bidirectional
contribution we want to capture as Δθ.

Output format reuses the relational_bidir layout so ``relational_facts_loader``
and ``relational_bidir_probe.py`` work without changes (probe is invoked with
``--split eval_memory --split eval_logic``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import List, Tuple


# Fictional first names (chosen to avoid common public-figure overlap).
FIRST_NAMES: List[str] = [
    "Aelar", "Briarn", "Caedyn", "Daros", "Eilora",
    "Faeryn", "Gildan", "Halryn", "Iolas", "Jaeden",
    "Kaerys", "Lyrian", "Maelis", "Nevyrn", "Orsam",
    "Phaeton", "Quoryn", "Riven", "Sylvar", "Tarvos",
    "Ulryk", "Veylin", "Wraen", "Xaiver", "Yoros",
    "Zephyr", "Albren", "Darshun", "Eryndor", "Fenric",
    "Glaeryn", "Helvic", "Imryss", "Jorvyn", "Kethrin",
    "Lothar", "Myros", "Naelth", "Oryon", "Praxis",
]

# Fictional last names (compounded place-words, no real surname overlap).
LAST_NAMES: List[str] = [
    "Ashthorne", "Brackenwood", "Calderfen", "Drystone", "Emberlock",
    "Fallowmere", "Glasswain", "Hollowmoor", "Ironvale", "Jadespire",
    "Kelpforge", "Lichgate", "Marshhollow", "Northvein", "Oakshroud",
    "Pyrecrest", "Quillmoor", "Ravenhalt", "Slatebrook", "Thornglade",
    "Underbough", "Velvetfen", "Wyrmstone", "Yarrowdyke", "Zinnshade",
    "Alchwood", "Brokenfern", "Cinderhalt", "Duskmere", "Elmreach",
    "Frostkettle", "Greyvane", "Hailspire", "Inkwell", "Jutmere",
    "Kindlebrook", "Loamspire", "Mistwarden", "Nightroost", "Orchardfen",
]


# Forward-direction templates (subject precedes object).
FACT_TEMPLATES: List[str] = [
    "{x} is paired with {y}.",
    "{x} is associated with {y}.",
    "{x} is matched with {y}.",
    "{x} is connected to {y}.",
    "{x} is linked with {y}.",
]


MASK = "[MASK]"


def stable_int(seed: str, *parts: object, mod: int) -> int:
    """SHA-256 -> integer mod ``mod`` for deterministic seeded selection."""
    payload = "::".join([seed, *[str(p) for p in parts]])
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big") % mod


def make_entity_name(seed: str, salt: str) -> str:
    first = FIRST_NAMES[stable_int(seed, salt, "first", mod=len(FIRST_NAMES))]
    last = LAST_NAMES[stable_int(seed, salt, "last", mod=len(LAST_NAMES))]
    return f"{first} {last}"


def gen_unique_pairs(seed: str, n_pairs: int) -> List[Tuple[str, str]]:
    """Each pair (X, Y) uses two distinct entity names; no entity is reused
    across pairs. Drawn deterministically from the (first × last) pool with
    rejection if the next entity collides."""
    pairs = []
    used: set[str] = set()
    cursor = 0
    capacity = len(FIRST_NAMES) * len(LAST_NAMES)
    while len(pairs) < n_pairs:
        x = make_entity_name(seed, f"x-{cursor}")
        y = make_entity_name(seed, f"y-{cursor}")
        cursor += 1
        if x == y or x in used or y in used:
            if cursor > capacity * 4:
                raise RuntimeError(
                    f"could not generate {n_pairs} disjoint pairs from "
                    f"{capacity} entity names; reduce --n_pairs"
                )
            continue
        pairs.append((x, y))
        used.update([x, y])
    return pairs


def deterministic_sample(pool: List[str], k: int, seed: str, *parts: object) -> List[str]:
    """Pick ``k`` items from ``pool`` without replacement, deterministic per seed."""
    if k > len(pool):
        raise ValueError(f"sample size {k} > pool size {len(pool)}")
    chosen: List[str] = []
    remaining = list(pool)
    for i in range(k):
        idx = stable_int(seed, *parts, "sample", i, mod=len(remaining))
        chosen.append(remaining.pop(idx))
    return chosen


def shuffled(items: List[str], seed: str, *parts: object) -> List[str]:
    """Deterministic Fisher-Yates shuffle."""
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = stable_int(seed, *parts, "shuf", i, mod=i + 1)
        out[i], out[j] = out[j], out[i]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=500)
    parser.add_argument("--n_templates_per_pair", type=int, default=5)
    parser.add_argument("--n_eval_pairs", type=int, default=200)
    parser.add_argument("--n_candidates", type=int, default=8)
    parser.add_argument("--seed", default="reversal-v1")
    parser.add_argument("--out", type=Path, default=Path("data/reversal_v1"))
    args = parser.parse_args()

    if args.n_eval_pairs > args.n_pairs:
        raise ValueError("--n_eval_pairs must be <= --n_pairs")
    if args.n_candidates < 2:
        raise ValueError("--n_candidates must be >= 2")
    if args.n_templates_per_pair > len(FACT_TEMPLATES):
        raise ValueError(
            f"--n_templates_per_pair must be <= {len(FACT_TEMPLATES)} (the number "
            f"of FACT_TEMPLATES)"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    pairs = gen_unique_pairs(args.seed, args.n_pairs)

    # Train rows: forward facts, multiple templates per pair.
    train_rows = []
    for i, (x, y) in enumerate(pairs):
        for j in range(args.n_templates_per_pair):
            template = FACT_TEMPLATES[j]
            text = template.format(x=x, y=y)
            train_rows.append(
                {
                    "id": f"train_{i:05d}_t{j}",
                    "split": "train_facts",
                    "text": text,
                    "pair_idx": i,
                    "subject": x,
                    "object": y,
                    "template_idx": j,
                }
            )
    train_rows = shuffled([json.dumps(r) for r in train_rows], args.seed, "train-shuf")
    with (args.out / "train_facts.jsonl").open("w", encoding="utf-8") as f:
        for line in train_rows:
            f.write(line + "\n")

    # Eval pairs: subset of train pairs (deterministic).
    eval_pair_ids = sorted(
        deterministic_sample(
            [str(i) for i in range(len(pairs))], args.n_eval_pairs, args.seed, "eval-pick"
        ),
        key=int,
    )
    eval_pair_indices = [int(s) for s in eval_pair_ids]

    all_subjects = [p[0] for p in pairs]
    all_objects = [p[1] for p in pairs]

    forward_rows, reverse_rows = [], []
    for k, i in enumerate(eval_pair_indices):
        x, y = pairs[i]
        # Use a single template per eval row, rotating across templates.
        template = FACT_TEMPLATES[k % len(FACT_TEMPLATES)]

        # ---- forward eval (memorization, AR can do) ----
        forward_input = template.format(x=x, y=MASK)
        # Distractors: 7 other Y's from the dataset.
        candidate_pool_y = [p[1] for p in pairs if p[1] != y]
        distractors_y = deterministic_sample(
            candidate_pool_y, args.n_candidates - 1, args.seed, "fwd-dist", i
        )
        candidates_y = shuffled([y] + distractors_y, args.seed, "fwd-shuf", i)
        forward_rows.append(
            {
                "id": f"eval_forward_{i:05d}",
                "split": "eval_memory",
                "task": "forward_query",
                "input": forward_input,
                "query": forward_input,
                "target": y,
                "candidates": candidates_y,
                "target_index": candidates_y.index(y),
                "mask_position": "right",
                "pair_idx": i,
                "subject": x,
                "object": y,
                "template_idx": k % len(FACT_TEMPLATES),
            }
        )

        # ---- reverse eval (curse test, AR fails) ----
        reverse_input = template.format(x=MASK, y=y)
        candidate_pool_x = [p[0] for p in pairs if p[0] != x]
        distractors_x = deterministic_sample(
            candidate_pool_x, args.n_candidates - 1, args.seed, "rev-dist", i
        )
        candidates_x = shuffled([x] + distractors_x, args.seed, "rev-shuf", i)
        reverse_rows.append(
            {
                "id": f"eval_reverse_{i:05d}",
                "split": "eval_logic",
                "task": "reverse_query",
                "input": reverse_input,
                "query": reverse_input,
                "target": x,
                "candidates": candidates_x,
                "target_index": candidates_x.index(x),
                "mask_position": "left",
                "pair_idx": i,
                "subject": x,
                "object": y,
                "template_idx": k % len(FACT_TEMPLATES),
            }
        )

    with (args.out / "eval_memory.jsonl").open("w", encoding="utf-8") as f:
        for r in forward_rows:
            f.write(json.dumps(r) + "\n")
    with (args.out / "eval_logic.jsonl").open("w", encoding="utf-8") as f:
        for r in reverse_rows:
            f.write(json.dumps(r) + "\n")

    metadata = {
        "config": vars(args) | {"out": str(args.out)},
        "files": {
            "train_facts.jsonl": len(train_rows),
            "eval_memory.jsonl": len(forward_rows),
            "eval_logic.jsonl": len(reverse_rows),
        },
        "n_unique_entities": len({e for p in pairs for e in p}),
    }
    with (args.out / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"=== Generated reversal_v1 ===")
    print(f"  out: {args.out}")
    print(f"  pairs: {len(pairs)}")
    print(f"  unique entities: {metadata['n_unique_entities']}")
    print(f"  train rows: {len(train_rows)}")
    print(f"  eval forward (eval_memory): {len(forward_rows)}")
    print(f"  eval reverse (eval_logic): {len(reverse_rows)}")


if __name__ == "__main__":
    main()
