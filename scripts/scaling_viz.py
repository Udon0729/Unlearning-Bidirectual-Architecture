"""4-scale (160M / 410M / 1B / 8B) scaling result plot.

Each model is a discrete measurement; we did not interpolate between sizes.
Bar chart with grouped bars per model is the correct visualization — line
plots would falsely imply we have data at intermediate sizes (e.g. 200M, 700M).
A divider separates Pythia (same family) from LLaMA-3.1-8B (different family).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Each row: (label, family, baseline_ppl, dlm_ppl, unlearn_ppl,
#                          baseline_nelbo, dlm_nelbo, unlearn_nelbo,
#                          baseline_gap, dlm_gap, unlearn_gap)
DATA = [
    ("Pythia-160M",  "Pythia", 81.7,  644.0,  81.7,  19.20, 10.18, 16.15, +0.50, +5.63, -0.04),
    ("Pythia-410M",  "Pythia", 31.4,   36.6,  15.4,  17.24,  7.49, 16.73, +1.32, +10.23, +1.25),
    ("Pythia-1B",    "Pythia", 26.1,   30.2,  12.7,  16.83,  7.07, 16.36, +1.79, +10.78, +1.64),
    ("LLaMA-3.1-8B", "LLaMA",   9.06,   8.79,  6.48, 23.16,  5.68, 22.91, +0.63, +16.17, -1.25),
]


def main():
    labels = [d[0] for d in DATA]
    families = [d[1] for d in DATA]
    bl_ppl = np.array([d[2] for d in DATA])
    dlm_ppl = np.array([d[3] for d in DATA])
    un_ppl = np.array([d[4] for d in DATA])
    bl_gap = np.array([d[8] for d in DATA])
    dlm_gap = np.array([d[9] for d in DATA])
    un_gap = np.array([d[10] for d in DATA])

    n = len(DATA)
    x = np.arange(n)
    w = 0.27

    # vertical divider between last Pythia and first LLaMA index
    family_split = next(i for i in range(1, n) if families[i] != families[i - 1])
    divider_x = family_split - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- AR capability ---
    ax = axes[0]
    ax.bar(x - w, bl_ppl, w, color="#444444", label="AR baseline (no adapt)")
    ax.bar(x,     dlm_ppl, w, color="#c0392b", label="DLM-adapted (LoRA)")
    ax.bar(x + w, un_ppl,  w, color="#27ae60", label="Unlearned (lora_only + KL→AR)")
    for xi, v in zip(x - w, bl_ppl):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, dlm_ppl):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w, un_ppl):
        ax.text(xi, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(x=divider_x, color="gray", linestyle=":", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("AR-mode PPL on WikiText-103 valid (log scale)")
    ax.set_title("AR capability (lower = better AR retention)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", which="both", linestyle="--", alpha=0.3)

    # --- DLM capability ---
    ax = axes[1]
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.bar(x - w, bl_gap, w, color="#444444", label="AR baseline (no adapt)")
    ax.bar(x,     dlm_gap, w, color="#c0392b", label="DLM-adapted (LoRA)")
    ax.bar(x + w, un_gap,  w, color="#27ae60", label="Unlearned (lora_only + KL→AR)")
    for xi, v in zip(x - w, bl_gap):
        ax.text(xi, v + 0.3, f"{v:+.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x, dlm_gap):
        ax.text(xi, v + 0.3, f"{v:+.1f}", ha="center", va="bottom", fontsize=8)
    for xi, v in zip(x + w, un_gap):
        offset = 0.3 if v >= 0 else -0.3
        va = "bottom" if v >= 0 else "top"
        ax.text(xi, v + offset, f"{v:+.1f}", ha="center", va=va, fontsize=8)
    ax.axvline(x=divider_x, color="gray", linestyle=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Reverse-context gap (bidir − causal acc, pp)")
    ax.set_title("DLM capability (higher = stronger DLM)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    fig.suptitle(
        "AR ⇄ DLM unlearning at 4 discrete scales. Bars (not lines) — "
        "no interpolation across intermediate model sizes is implied.",
        fontsize=11, y=1.01,
    )
    fig.tight_layout()

    out = Path("results/scaling_viz.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
