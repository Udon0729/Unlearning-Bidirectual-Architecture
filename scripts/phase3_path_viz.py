"""Phase 2 / 2.5 / 3 v2 / v3 の results/*.jsonl を一括可視化する。

生成する図:
  - figs/phase3_progress.png         : Phase 2.5 mode collapse + Phase 3 進行
  - figs/parallel_gen_contrib.png    : 右側文脈寄与 (並列生成能力指標)
  - figs/memorization_paths.png      : 重み記憶経路と attention 経路の分解
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "IPAGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = ROOT / "figs"
FIGS.mkdir(exist_ok=True)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def fig_phase3_progress() -> None:
    p25 = load_jsonl(RESULTS / "phase25_llada_causal.jsonl")
    p25_split = [r for r in p25 if r["group"] == "split"]

    def find_p25(mode: str, split_v: str) -> float:
        return next(
            r["candidate_top1"]
            for r in p25_split
            if r["mode"] == mode and r["value"] == split_v
        )

    p25_nat_fwd = find_p25("natural", "eval_memory")
    p25_nat_rev = find_p25("natural", "eval_logic")
    p25_cau_fwd = find_p25("causal", "eval_memory")
    p25_cau_rev = find_p25("causal", "eval_logic")

    p3v2 = load_jsonl(RESULTS / "phase3v2_lrs_probe.jsonl")
    p3v3 = load_jsonl(RESULTS / "phase3v3_lrs_probe.jsonl")

    def find_score(rows: list[dict], label: str, split_v: str) -> float | None:
        for r in rows:
            if (
                r.get("label") == label
                and r["group"] == "split"
                and r["value"] == split_v
            ):
                return r["candidate_top1"]
        return None

    progression = [
        ("Phase 2\nbaseline", "phase2-baseline", p3v2, p3v2),
        ("Phase 3 v2\n(L_rs uniform 33層)", "unlearn-lrs-1k", p3v2, p3v2),
        ("Phase 3 v3\nstep500", "v3-step500", p3v3, p3v3),
        ("Phase 3 v3\nstep1000", "v3-step1000", p3v3, p3v3),
        ("Phase 3 v3\nfinal\n(L_rs deep, 4 pos)", "v3-final", p3v3, p3v3),
    ]
    labels = [p[0] for p in progression]
    forward = [find_score(p[2], p[1], "eval_memory") for p in progression]
    reverse = [find_score(p[3], p[1], "eval_logic") for p in progression]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    modes = ["natural\n(LLaDA 既定)", "causal bias\n強制"]
    f25 = [p25_nat_fwd, p25_cau_fwd]
    r25 = [p25_nat_rev, p25_cau_rev]
    x = np.arange(len(modes))
    w = 0.35
    bars1 = ax1.bar(x - w / 2, f25, w, label="forward (eval_memory)", color="#3aa55a")
    bars2 = ax1.bar(x + w / 2, r25, w, label="reverse (eval_logic)", color="#cc3344")
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes, fontsize=10)
    ax1.set_ylabel("cand@1 (span PLL)")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Phase 2.5: 推論時 attention bias 注入による mode collapse")
    ax1.axhline(0.125, color="gray", linestyle=":", alpha=0.6, label="prior (1/8)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bars, vals in [(bars1, f25), (bars2, r25)]:
        for b, v in zip(bars, vals):
            ax1.annotate(
                f"{v:.3f}",
                (b.get_x() + b.get_width() / 2, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
            )

    x = np.arange(len(labels))
    bars1 = ax2.bar(x - w / 2, forward, w, label="forward (eval_memory)", color="#3aa55a")
    bars2 = ax2.bar(x + w / 2, reverse, w, label="reverse (eval_logic)", color="#cc3344")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8.5)
    ax2.set_ylabel("cand@1 (span PLL)")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Phase 3: L_rs Unlearning による forward−reverse 非対称化")
    ax2.legend(loc="lower left", fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    gaps = [(f - r) * 100 for f, r in zip(forward, reverse)]
    for bars, vals in [(bars1, forward), (bars2, reverse)]:
        for b, v in zip(bars, vals):
            ax2.annotate(
                f"{v:.3f}",
                (b.get_x() + b.get_width() / 2, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=9,
            )
    for i, g in enumerate(gaps):
        ax2.annotate(
            f"gap=+{g:.1f}pp",
            (i, 0.05),
            ha="center",
            fontsize=8.5,
            color="#444",
            fontweight="bold",
        )

    plt.tight_layout()
    out = FIGS / "phase3_progress.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def fig_parallel_gen() -> None:
    rows = load_jsonl(RESULTS / "parallel_gen_probe.jsonl")
    order = ["phase2-baseline", "v2-final", "v3-step500", "v3-final"]
    label_disp = {
        "phase2-baseline": "Phase 2\nbaseline",
        "v2-final": "Phase 3 v2\nfinal",
        "v3-step500": "Phase 3 v3\nstep500",
        "v3-final": "Phase 3 v3\nfinal",
    }
    rs = [next(r for r in rows if r["label"] == o) for o in order]
    labels = [label_disp[r["label"]] for r in rs]
    lp_full = [r["log_p_full"] for r in rs]
    lp_left = [r["log_p_left_only"] for r in rs]
    lp_right = [r["log_p_right_only"] for r in rs]
    right_contrib = [f - l for f, l in zip(lp_full, lp_left)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    x = np.arange(len(labels))
    w = 0.27
    ax1.bar(x - w, lp_full, w, label="full (位置 i のみ mask)", color="#3a85cc")
    ax1.bar(x, lp_left, w, label="left_only (右側を mask)", color="#3aa55a")
    ax1.bar(x + w, lp_right, w, label="right_only (左側を mask)", color="#cc3344")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("log p(orig token)  [nats]")
    ax1.set_title("位置 i における各 ablation 条件下の対数尤度 (WikiText-103)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="black", linewidth=0.6)

    bars = ax2.bar(x, right_contrib, color="#cc6633")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("right-context contribution\n= log p_full − log p_left_only  [nats]")
    ax2.set_title("右側文脈の寄与 (並列生成能力の物理基盤)")
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, right_contrib):
        ax2.annotate(
            f"{v:.2f} nats",
            (b.get_x() + b.get_width() / 2, v),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out = FIGS / "parallel_gen_contrib.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def fig_memorization() -> None:
    rows = load_jsonl(RESULTS / "reversal_memorization_probe.jsonl")
    order = ["phase2-baseline", "v2-final", "v3-final"]
    label_disp = {
        "phase2-baseline": "Phase 2\nbaseline",
        "v2-final": "Phase 3 v2\nfinal",
        "v3-final": "Phase 3 v3\nfinal",
    }
    rs = [next(r for r in rows if r["label"] == o) for o in order]
    labels = [label_disp[r["label"]] for r in rs]
    full = [r["cand@1_full"] for r in rs]
    no_Y = [r["cand@1_no_Y"] for r in rs]
    no_ctx = [r["cand@1_no_context"] for r in rs]
    y_contrib = [f - n for f, n in zip(full, no_Y)]
    weight_contrib = [n - c for n, c in zip(no_Y, no_ctx)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))
    w = 0.27
    bars_f = ax1.bar(x - w, full, w, label="full (Y 可視)", color="#3a85cc")
    bars_n = ax1.bar(x, no_Y, w, label="no_Y (Y を mask)", color="#cc9933")
    bars_c = ax1.bar(x + w, no_ctx, w, label="no_context (主語 [MASK] のみ)", color="#888888")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("first-token cand@1")
    ax1.set_title("reverse query を 3 regime で評価")
    ax1.set_ylim(0, max(full) * 1.3)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    for bars, vals in [(bars_f, full), (bars_n, no_Y), (bars_c, no_ctx)]:
        for b, v in zip(bars, vals):
            ax1.annotate(
                f"{v:.3f}",
                (b.get_x() + b.get_width() / 2, v),
                textcoords="offset points",
                xytext=(0, 3),
                ha="center",
                fontsize=8.5,
            )

    ax2.bar(x, no_ctx, label="prior 水準 (no_context)", color="#bbbbbb")
    ax2.bar(x, weight_contrib, bottom=no_ctx,
            label="重み記憶経路 (no_Y − no_context)", color="#aa55aa")
    ax2.bar(x, y_contrib, bottom=[c + w for c, w in zip(no_ctx, weight_contrib)],
            label="Y attention 経路 (full − no_Y)", color="#cc3344")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("first-token cand@1 (経路分解、積み上げ)")
    ax2.set_title("経路ごとの寄与分解")
    ax2.legend(loc="upper right", fontsize=8.5)
    ax2.grid(axis="y", alpha=0.3)
    for i, (c, w_, y) in enumerate(zip(no_ctx, weight_contrib, y_contrib)):
        ax2.annotate(
            f"Y={y:+.3f}\n重み={w_:+.3f}",
            (i, c + w_ + y + 0.01),
            ha="center",
            fontsize=8.5,
            color="#222",
        )

    plt.tight_layout()
    out = FIGS / "memorization_paths.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"saved: {out}")
    plt.close(fig)


def main() -> None:
    fig_phase3_progress()
    fig_parallel_gen()
    fig_memorization()


if __name__ == "__main__":
    main()
