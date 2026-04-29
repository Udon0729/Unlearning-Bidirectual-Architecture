"""Single summary figure: AR-mode PPL and bidirectional advantage on
reverse-context prediction, side by side, across all ckpts.

These two bars together encode the project's central claim:
  - AR PPL bar (lower = better AR)         : DLM-adapted ckpts spike high;
                                              unlearned ckpts return near baseline.
  - Reverse-bidir advantage bar (higher =
    more DLM-specific bidirectional use)   : DLM-adapted ckpts spike high;
                                              unlearned ckpts collapse to ~0.

Numbers are hard-coded from earlier eval runs to keep this script simple.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# (label, AR PPL, reverse causal acc, reverse bidir acc)
ROWS = [
    ("AR baseline\n(no adapt)",            81.7,  0.0016, 0.0066),
    ("Full FT\n(DLM)",                     1166,  0.0109, 0.0543),
    ("LoRA-25k\n(DLM)",                     644,  0.0109, 0.0672),
    ("Unlearn Full\nNPO α=0.5",             164,  0.0309, 0.0309),
    ("Unlearn Full\nKL → AR (B3)",          161,  0.0016, 0.0078),
    ("Unlearn LoRA-only\nNPO α=0.5",        78.0, 0.0309, 0.0309),
    ("Unlearn LoRA-only\nKL → AR (B4)",     81.7, 0.0109, 0.0105),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/summary_viz.png"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    labels = [r[0] for r in ROWS]
    ppls = np.array([r[1] for r in ROWS])
    rev_gap = np.array([r[3] - r[2] for r in ROWS]) * 100  # in pp

    is_baseline = [r[0].startswith("AR baseline") for r in ROWS]
    is_dlm = ["DLM" in r[0] for r in ROWS]
    is_unlearn = [r[0].startswith("Unlearn") for r in ROWS]
    colors = []
    for b, d, u in zip(is_baseline, is_dlm, is_unlearn):
        colors.append("#666666" if b else ("#cc3344" if d else "#3aa55a"))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # AR PPL
    bars = axes[0].bar(range(len(labels)), ppls, color=colors)
    axes[0].axhline(y=ppls[0], color="#666666", linestyle="--", linewidth=1, alpha=0.5,
                    label=f"AR baseline = {ppls[0]:.1f}")
    axes[0].set_xlim(-0.6, len(labels) - 0.4)
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right")
    axes[0].set_ylabel("AR-mode PPL (log scale, lower = better AR retention)", fontsize=10)
    axes[0].set_title("AR capability: AR-mode perplexity on WikiText-103 valid", fontsize=11)
    axes[0].legend(fontsize=9, loc="upper right")
    for bar, v in zip(bars, ppls):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v * 1.1, f"{v:.0f}",
                     ha="center", va="bottom", fontsize=9)

    # Reverse-bidir advantage
    bars2 = axes[1].bar(range(len(labels)), rev_gap, color=colors)
    axes[1].axhline(y=rev_gap[0], color="#666666", linestyle="--", linewidth=1, alpha=0.5,
                    label=f"AR baseline = {rev_gap[0]:+.2f}pp")
    axes[1].axhline(y=0, color="black", linewidth=0.5)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right")
    axes[1].set_ylabel("bidirectional - causal acc (pp, higher = stronger DLM)", fontsize=10)
    axes[1].set_title("DLM capability: reverse-context probe gap (top-1 acc)", fontsize=11)
    axes[1].legend(fontsize=9, loc="upper right")
    # explicit y-range so the negative B4 bar and the zero-bars are clearly placed
    axes[1].set_xlim(-0.6, len(labels) - 0.4)
    # y_min just barely below the most-negative value so the "0" tick is
    # visually pinned at the panel floor (no floating empty band beneath it).
    y_min = min(rev_gap.min() - 0.02, -0.02)
    axes[1].set_ylim(y_min, rev_gap.max() * 1.15)
    for bar, v in zip(bars2, rev_gap):
        # sign-aware label format; place above for non-negative, below for negative
        offset = 0.08 if v >= 0 else -0.01
        va = "bottom" if v >= 0 else "top"
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + offset, f"{v:+.2f}",
                     ha="center", va=va, fontsize=9)

    fig.suptitle(
        "AR ⇄ DLM unlearning: AR recovers (left) while DLM-specific bidirectional use vanishes (right)",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(args.out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
