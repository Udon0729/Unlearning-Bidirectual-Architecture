"""主要 wandb run の訓練曲線を取得して可視化する。

生成する図:
  - figs/wandb_p1_dlm_lora25k.png      : Phase 1 DLM 化 (loss)
  - figs/wandb_p1_unlearn_b4.png       : Phase 1 Unlearn (loss / forget / retain)
  - figs/wandb_p2_llada_sft.png        : Phase 2 LLaDA SFT (loss)
  - figs/wandb_p3v2_lrs.png            : Phase 3 v2 (loss / forget / retain)
  - figs/wandb_p3v3_deep.png           : Phase 3 v3 (loss / forget / retain)
  - figs/wandb_phase3_compare.png      : v2 vs v3 比較 (forget / retain)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import wandb

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)
PROJECT = "udon0729-shizuoka-university/unlearning-architecture"

RUNS: dict[str, tuple[str, str]] = {
    "p1_dlm_lora25k": ("8fbmvilf", "Phase 1: Pythia-160M DLM 化 (LoRA, 25k step)"),
    "p1_unlearn_b4": ("2um0107b", "Phase 1: Unlearn (B4 = lora_only + KL→AR)"),
    "p2_llada_sft": ("v70ulp1t", "Phase 2: LLaDA-8B-Base SFT on reversal_v1 (2k step)"),
    "p3v2_lrs": ("tohj922d", "Phase 3 v2: L_rs Unlearning (uniform 33 層)"),
    "p3v3_deep": ("jty0nxo7", "Phase 3 v3: L_rs (deep 層 [20:] + 4 positions)"),
}


def fetch(run_id: str):
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{run_id}")
    df = run.history(samples=10000)
    return df, run.name


def plot_unlearn_run(run_key: str) -> None:
    rid, title = RUNS[run_key]
    df, _ = fetch(rid)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [
        ("loss", "#cc3344", "total loss = α·forget + retain"),
        ("forget", "#cc6633", "forget loss"),
        ("retain", "#3a85cc", "retain loss (WikiText-103 CE)"),
    ]
    for ax, (col, color, label) in zip(axes, panels):
        if col in df.columns:
            sub = df.dropna(subset=[col])
            ax.plot(sub["_step"], sub[col], color=color, linewidth=1.5)
            ax.set_xlabel("step")
            ax.set_title(label, fontsize=10.5)
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"no '{col}' column", ha="center", va="center")
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out = FIGS / f"wandb_{run_key}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def plot_simple_run(run_key: str) -> None:
    rid, title = RUNS[run_key]
    df, _ = fetch(rid)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if "loss" in df.columns:
        sub = df.dropna(subset=["loss"])
        axes[0].plot(sub["_step"], sub["loss"], color="#cc3344", linewidth=1.5)
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("loss")
        axes[0].set_title("training loss")
        axes[0].grid(alpha=0.3)
    if "gnorm" in df.columns:
        sub = df.dropna(subset=["gnorm"])
        axes[1].plot(sub["_step"], sub["gnorm"], color="#cc6633", linewidth=1.5)
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("gradient norm")
        axes[1].set_title("gradient norm")
        axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    out = FIGS / f"wandb_{run_key}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def plot_phase3_compare() -> None:
    df_v2, _ = fetch(RUNS["p3v2_lrs"][0])
    df_v3, _ = fetch(RUNS["p3v3_deep"][0])
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    for col, ax_idx, title in [
        ("forget", 0, "forget loss (L_rs)"),
        ("retain", 1, "retain CE (WikiText-103)"),
    ]:
        ax = axes[ax_idx]
        for df, label, color in [
            (df_v2, "v2 (uniform 33 層)", "#3a85cc"),
            (df_v3, "v3 (deep [20:] + 4 positions)", "#cc3344"),
        ]:
            if col in df.columns:
                sub = df.dropna(subset=[col])
                ax.plot(sub["_step"], sub[col], label=label, color=color, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("step")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Phase 3: v2 vs v3 訓練曲線比較", fontsize=12)
    plt.tight_layout()
    out = FIGS / "wandb_phase3_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def main() -> None:
    plot_simple_run("p1_dlm_lora25k")
    plot_unlearn_run("p1_unlearn_b4")
    plot_simple_run("p2_llada_sft")
    plot_unlearn_run("p3v2_lrs")
    plot_unlearn_run("p3v3_deep")
    plot_phase3_compare()


if __name__ == "__main__":
    main()
