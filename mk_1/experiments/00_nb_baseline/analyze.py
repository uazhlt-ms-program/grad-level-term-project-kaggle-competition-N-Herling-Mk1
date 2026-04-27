"""
mk_1/experiments/00_nb_baseline/analyze.py

Load the sweep results, generate the comparison table and the
alpha-vs-metric scatter plots.

Inputs:
    results/sweep.json     per-config metrics (from sweep.py)
    results/winners.json   best config per regime (from sweep.py)

Outputs:
    figures/alpha_vs_<metric>.png   one scatter per metric
    figures/winners_table.txt        text comparison table
    stdout                            same comparison printed
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


# Plot styling -----------------------------------------------------
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


# Comparison table -------------------------------------------------
def make_table(winners: dict, records: list[dict]) -> str:
    """Side-by-side table of the three winners on every metric."""
    metrics = ["f1_macro", "H_epistemic", "ECE", "AUROC_U",
               "H_high_sigma", "rrm_penalty", "maxent_loss"]

    header = f"{'metric':<18s}  {'F1-tuned':>14s}  {'RRM-tuned':>14s}  {'MaxEnt-tuned':>14s}"
    lines  = [header, "-" * len(header)]

    # First the hyperparameters
    lines.append(f"{'alpha':<18s}  "
                 f"{winners['f1_tuned']['alpha']:>14.4g}  "
                 f"{winners['rrm_tuned']['alpha']:>14.4g}  "
                 f"{winners['maxent_tuned']['alpha']:>14.4g}")
    lines.append(f"{'ngram_range':<18s}  "
                 f"{str(tuple(winners['f1_tuned']['ngram_range'])):>14s}  "
                 f"{str(tuple(winners['rrm_tuned']['ngram_range'])):>14s}  "
                 f"{str(tuple(winners['maxent_tuned']['ngram_range'])):>14s}")
    lines.append(f"{'min_df':<18s}  "
                 f"{winners['f1_tuned']['min_df']:>14d}  "
                 f"{winners['rrm_tuned']['min_df']:>14d}  "
                 f"{winners['maxent_tuned']['min_df']:>14d}")
    lines.append("-" * len(header))

    # Then metrics
    for m in metrics:
        lines.append(
            f"{m:<18s}  "
            f"{winners['f1_tuned'][m]:>14.4f}  "
            f"{winners['rrm_tuned'][m]:>14.4f}  "
            f"{winners['maxent_tuned'][m]:>14.4f}"
        )

    return "\n".join(lines)


# Plot --------------------------------------------------------------
def plot_alpha_vs_metric(records, winners, metric_key, title, subtitle, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))

    alphas = np.array([r["alpha"] for r in records])
    vals   = np.array([r[metric_key] for r in records])

    ax.scatter(alphas, vals, s=26, alpha=0.55, color="#444444",
               edgecolor="none", label="all configs")

    # Mark winners
    for regime, rec in winners.items():
        ax.scatter(
            rec["alpha"], rec[metric_key],
            s=200, edgecolor=REGIME_COLORS[regime], facecolor="none",
            linewidth=2.5, label=REGIME_LABELS[regime], zorder=3,
        )

    # Reference line for H_high_sigma
    if metric_key == "H_high_sigma":
        ax.axhline(np.log(3), color="grey", linestyle="--", linewidth=1,
                   label="MaxEnt floor (ln 3)")

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet prior $\alpha$ (log scale)")
    ax.set_ylabel(title)
    ax.set_title(f"{title}    [{subtitle}]")
    ax.grid(True, alpha=0.25, which="both", linestyle=":")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# Main -------------------------------------------------------------
def main():
    records, winners = load_results()

    table = make_table(winners, records)
    print()
    print("=== Sweep results: NB winners under three regimes ===")
    print()
    print(table)
    print()

    # Save table
    table_path = FIG_DIR / "winners_table.txt"
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f">>> wrote {table_path}", flush=True)

    # Plots
    for key, label, sub in METRICS_TO_PLOT:
        out = FIG_DIR / f"alpha_vs_{key}.png"
        plot_alpha_vs_metric(records, winners, key, label, sub, out)
        print(f">>> wrote {out}", flush=True)

    # Diagnostic finding
    print()
    print("--- diagnostic ---")
    a_f1     = winners["f1_tuned"]["alpha"]
    a_rrm    = winners["rrm_tuned"]["alpha"]
    a_maxent = winners["maxent_tuned"]["alpha"]
    spread = max(a_f1, a_rrm, a_maxent) / min(a_f1, a_rrm, a_maxent)
    print(f"alpha winners: F1={a_f1:.4g}  RRM={a_rrm:.4g}  MaxEnt={a_maxent:.4g}")
    print(f"spread (max/min) = {spread:.2f}x")
    if spread < 2.0:
        print("  ==> regimes converge: F1-optimal coincides with calibration-optimal")
        print("      (margin-proxy sigma is not adding selection signal beyond F1)")
    elif spread < 10.0:
        print("  ==> regimes diverge mildly: some selection effect from RRM/MaxEnt")
    else:
        print("  ==> regimes diverge strongly: each picks a meaningfully different alpha")
        print("      (the mechanism is selecting differently even with margin-proxy sigma)")


if __name__ == "__main__":
    main()
