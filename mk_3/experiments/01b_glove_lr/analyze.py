"""
mk_3/experiments/01b_glove_lr/analyze.py

Load the sweep results and generate the comparison table + C-vs-metric plots.
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

REGIME_COLORS = {
    "f1_tuned":     "#1f77b4",
    "rrm_tuned":    "#2ca02c",
    "maxent_tuned": "#d62728",
}
REGIME_LABELS = {
    "f1_tuned":     "F1-tuned winner",
    "rrm_tuned":    "RRM-tuned winner",
    "maxent_tuned": "MaxEnt-tuned winner",
}
METRICS_TO_PLOT = [
    ("f1_macro",      "Macro-F1",                 "higher = better"),
    ("rrm_penalty",   "RRM penalty (in-fold L2)", "lower = better"),
    ("maxent_loss",   "MaxEnt loss",              "lower = better"),
    ("ECE",           "Expected Calibration Err", "lower = better"),
    ("AUROC_U",       "AUROC_U (uncertainty)",    "higher = better"),
    ("H_high_sigma",  "H (top-quartile sigma)",   f"target ln 3 = {np.log(3):.4f}"),
]


def load_results():
    with open(RESULTS_DIR / "sweep.json") as f:
        sweep = json.load(f)
    with open(RESULTS_DIR / "winners.json") as f:
        winners = json.load(f)
    return sweep["records"], winners


def make_table(winners: dict, records: list[dict]) -> str:
    metrics = ["f1_macro", "H_epistemic", "ECE", "AUROC_U",
               "H_high_sigma", "rrm_penalty", "maxent_loss"]

    header = f"{'metric':<18s}  {'F1-tuned':>14s}  {'RRM-tuned':>14s}  {'MaxEnt-tuned':>14s}"
    lines  = [header, "-" * len(header)]

    lines.append(f"{'C':<18s}  "
                 f"{winners['f1_tuned']['C']:>14.4g}  "
                 f"{winners['rrm_tuned']['C']:>14.4g}  "
                 f"{winners['maxent_tuned']['C']:>14.4g}")
    lines.append(f"{'pooling':<18s}  "
                 f"{winners['f1_tuned']['pooling']:>14s}  "
                 f"{winners['rrm_tuned']['pooling']:>14s}  "
                 f"{winners['maxent_tuned']['pooling']:>14s}")
    lines.append(f"{'normalize':<18s}  "
                 f"{str(winners['f1_tuned']['normalize']):>14s}  "
                 f"{str(winners['rrm_tuned']['normalize']):>14s}  "
                 f"{str(winners['maxent_tuned']['normalize']):>14s}")
    lines.append(f"{'class_weight':<18s}  "
                 f"{str(winners['f1_tuned']['class_weight']):>14s}  "
                 f"{str(winners['rrm_tuned']['class_weight']):>14s}  "
                 f"{str(winners['maxent_tuned']['class_weight']):>14s}")
    lines.append("-" * len(header))

    for m in metrics:
        lines.append(
            f"{m:<18s}  "
            f"{winners['f1_tuned'][m]:>14.4f}  "
            f"{winners['rrm_tuned'][m]:>14.4f}  "
            f"{winners['maxent_tuned'][m]:>14.4f}"
        )
    return "\n".join(lines)


def plot_C_vs_metric(records, winners, metric_key, title, subtitle, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))

    Cs   = np.array([r["C"] for r in records])
    vals = np.array([r[metric_key] for r in records])

    ax.scatter(Cs, vals, s=26, alpha=0.55, color="#444444",
               edgecolor="none", label="all configs")

    for regime, rec in winners.items():
        ax.scatter(
            rec["C"], rec[metric_key],
            s=200, edgecolor=REGIME_COLORS[regime], facecolor="none",
            linewidth=2.5, label=REGIME_LABELS[regime], zorder=3,
        )

    if metric_key == "H_high_sigma":
        ax.axhline(np.log(3), color="grey", linestyle="--", linewidth=1,
                   label="MaxEnt floor (ln 3)")

    ax.set_xscale("log")
    ax.set_xlabel(r"LR inverse-regularization $C$ (log scale)")
    ax.set_ylabel(title)
    ax.set_title(f"GloVe + LR : {title}    [{subtitle}]")
    ax.grid(True, alpha=0.25, which="both", linestyle=":")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    records, winners = load_results()

    table = make_table(winners, records)
    print()
    print("=== Sweep results: GloVe + LR winners under three regimes ===")
    print()
    print(table)
    print()

    table_path = FIG_DIR / "winners_table.txt"
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f">>> wrote {table_path}", flush=True)

    for key, label, sub in METRICS_TO_PLOT:
        out = FIG_DIR / f"C_vs_{key}.png"
        plot_C_vs_metric(records, winners, key, label, sub, out)
        print(f">>> wrote {out}", flush=True)

    print()
    print("--- diagnostic ---")
    C_f1     = winners["f1_tuned"]["C"]
    C_rrm    = winners["rrm_tuned"]["C"]
    C_maxent = winners["maxent_tuned"]["C"]
    spread = max(C_f1, C_rrm, C_maxent) / min(C_f1, C_rrm, C_maxent)
    print(f"C winners: F1={C_f1:.4g}  RRM={C_rrm:.4g}  MaxEnt={C_maxent:.4g}")
    print(f"spread (max/min) = {spread:.2f}x")
    if spread < 2.0:
        print("  ==> regimes converge")
    elif spread < 10.0:
        print("  ==> regimes diverge mildly")
    else:
        print("  ==> regimes diverge strongly")

    print()
    print("--- cross-layer comparison ---")
    f1_winner = winners["f1_tuned"]["f1_macro"]
    print(f"mk_1 (NB)         F1-tuned  : F1 = 0.8812  (Kaggle: not submitted)")
    print(f"mk_2 (TF-IDF+LR)  F1-tuned  : F1 = 0.9200  (Kaggle: 0.92758)")
    print(f"mk_3 (GloVe+LR)   F1-tuned  : F1 = {f1_winner:.4f}  (Kaggle: TBD)")


if __name__ == "__main__":
    main()
