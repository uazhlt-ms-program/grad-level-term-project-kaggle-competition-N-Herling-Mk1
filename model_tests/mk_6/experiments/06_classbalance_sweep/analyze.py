"""
mk_6/experiments/06_classbalance_sweep/analyze.py

Load mk_6 sweep results and generate comparison tables + plots, with
specific attention to the class-balance dimension.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

RESULTS_DIR = HERE / "results"
FIG_DIR     = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_results():
    with open(RESULTS_DIR / "sweep.json") as f:
        sweep = json.load(f)
    with open(RESULTS_DIR / "winners.json") as f:
        winners = json.load(f)
    return sweep["records"], winners


def make_winners_table(winners):
    metrics = ["f1_macro", "H_epistemic", "ECE", "AUROC_U",
               "H_high_sigma", "rrm_penalty", "maxent_loss"]
    header = f"{'param':<22s}  {'F1-tuned':>14s}  {'RRM-tuned':>14s}  {'MaxEnt-tuned':>14s}"
    lines = [header, "-" * len(header)]

    knobs = [
        ("C", lambda w: f"{w['C']:.4g}"),
        ("min_df", lambda w: str(w['min_df'])),
        ("max_features", lambda w: str(w['max_features'])),
        ("sublinear_tf", lambda w: str(w['sublinear_tf'])),
        ("negation_applied", lambda w: str(w['negation_applied'])),
        ("class0_undersample", lambda w: f"{w['class0_undersample']:.2f}"),
        ("class1_oversample", lambda w: f"{w['class1_oversample']:.2f}"),
        ("class2_oversample", lambda w: f"{w['class2_oversample']:.2f}"),
        ("class_weight", lambda w: str(w['class_weight'])),
        ("n_train", lambda w: str(w['n_train'])),
    ]
    for label, getter in knobs:
        lines.append(f"{label:<22s}  "
                     f"{getter(winners['f1_tuned']):>14s}  "
                     f"{getter(winners['rrm_tuned']):>14s}  "
                     f"{getter(winners['maxent_tuned']):>14s}")
    lines.append("-" * len(header))
    for m in metrics:
        lines.append(
            f"{m:<22s}  "
            f"{winners['f1_tuned'][m]:>14.4f}  "
            f"{winners['rrm_tuned'][m]:>14.4f}  "
            f"{winners['maxent_tuned'][m]:>14.4f}"
        )
    return "\n".join(lines)


def class_balance_breakdown(records):
    """Show mean F1 by class-balance choice — does undersampling help?"""
    lines = ["", "=== F1 by class-balance choice ===", ""]

    # by class0_undersample
    lines.append(f"{'class0_undersample':<22s}  {'n_configs':>10s}  {'mean_F1':>10s}  {'max_F1':>10s}")
    for u in sorted(set(r["class0_undersample"] for r in records)):
        subset = [r for r in records if r["class0_undersample"] == u]
        if not subset:
            continue
        mean_f1 = np.mean([r["f1_macro"] for r in subset])
        max_f1  = max(r["f1_macro"] for r in subset)
        lines.append(f"{u:<22.2f}  {len(subset):>10d}  {mean_f1:>10.4f}  {max_f1:>10.4f}")

    lines.append("")
    lines.append(f"{'class1_oversample':<22s}  {'n_configs':>10s}  {'mean_F1':>10s}  {'max_F1':>10s}")
    for o in sorted(set(r["class1_oversample"] for r in records)):
        subset = [r for r in records if r["class1_oversample"] == o]
        if not subset:
            continue
        mean_f1 = np.mean([r["f1_macro"] for r in subset])
        max_f1  = max(r["f1_macro"] for r in subset)
        lines.append(f"{o:<22.2f}  {len(subset):>10d}  {mean_f1:>10.4f}  {max_f1:>10.4f}")

    lines.append("")
    lines.append(f"{'class2_oversample':<22s}  {'n_configs':>10s}  {'mean_F1':>10s}  {'max_F1':>10s}")
    for o in sorted(set(r["class2_oversample"] for r in records)):
        subset = [r for r in records if r["class2_oversample"] == o]
        if not subset:
            continue
        mean_f1 = np.mean([r["f1_macro"] for r in subset])
        max_f1  = max(r["f1_macro"] for r in subset)
        lines.append(f"{o:<22.2f}  {len(subset):>10d}  {mean_f1:>10.4f}  {max_f1:>10.4f}")

    lines.append("")
    lines.append(f"{'negation_applied':<22s}  {'n_configs':>10s}  {'mean_F1':>10s}  {'max_F1':>10s}")
    for n in [False, True]:
        subset = [r for r in records if r["negation_applied"] == n]
        if not subset:
            continue
        mean_f1 = np.mean([r["f1_macro"] for r in subset])
        max_f1  = max(r["f1_macro"] for r in subset)
        lines.append(f"{str(n):<22s}  {len(subset):>10d}  {mean_f1:>10.4f}  {max_f1:>10.4f}")

    return "\n".join(lines)


def plot_undersample_vs_f1(records, out_path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    us = [r["class0_undersample"] for r in records]
    f1 = [r["f1_macro"] for r in records]
    ax.scatter(us, f1, alpha=0.65, s=40, color="#1f77b4", edgecolor="none")
    # add jitter to avoid overplotting
    rng = np.random.default_rng(0)
    jitter = rng.normal(0, 0.005, size=len(us))
    ax.scatter(np.array(us) + jitter, f1, alpha=0.65, s=40, color="#1f77b4", edgecolor="none")
    ax.set_xlabel("class0_undersample ratio (1.0 = no undersampling)")
    ax.set_ylabel("Macro F1 on val")
    ax.set_title("mk_6 sweep: F1 vs Class 0 undersampling ratio")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    records, winners = load_results()

    # Winners table
    table = make_winners_table(winners)
    print()
    print("=" * 72)
    print("=== mk_6 sweep winners ===")
    print("=" * 72)
    print(table)

    # Class balance breakdown
    cb_text = class_balance_breakdown(records)
    print(cb_text)

    # Save tables
    with open(FIG_DIR / "winners_table.txt", "w") as f:
        f.write(table + "\n" + cb_text + "\n")
    print(f"\n>>> wrote {FIG_DIR / 'winners_table.txt'}")

    # Plot
    plot_undersample_vs_f1(records, FIG_DIR / "undersample_vs_f1.png")
    print(f">>> wrote {FIG_DIR / 'undersample_vs_f1.png'}")

    # Final ranking
    f1_winner = winners["f1_tuned"]["f1_macro"]
    print()
    print("--- cross-layer leaderboard ---")
    print(f"mk_2 F1-tuned (Kaggle 0.92758): val 0.9200")
    print(f"mk_5 F1-tuned (Kaggle 0.92677): val 0.9228")
    print(f"mk_6 F1-tuned                 : val {f1_winner:.4f}")


if __name__ == "__main__":
    main()
