"""Summarise SnapKV vs SnapKV+SANTA results as traffic-matched comparisons.

Usage: python analyze_hybrid.py <results_root>

For every summary.csv (e2e) and oracle_summary.csv (oracle) under the root it
writes, next to the CSV:
  pareto_<metric>.png   metric vs mean K+V rows per KV head per decode step:
                        SnapKV-only curve, plus hybrid sys/top points
  matched.csv           for each hybrid point, the SnapKV-only metric
                        linearly interpolated at the same traffic, and the delta
The single number that says "merit / no merit" is the delta column:
positive score delta (e2e) or negative error delta (oracle) at matched traffic.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # plots are optional
    plt = None


def analyse(csv_path: pathlib.Path):
    df = pd.read_csv(csv_path)
    oracle = "rel_err_mean" in df.columns and "score" not in df.columns
    metric = "rel_err_mean" if oracle else "score"
    x = "mean_KV_rows"
    base = df[(df.decoder == "dense") & (df.budget != "full")].sort_values(x)
    hyb = df[df.decoder.isin(["sys", "top"])].copy()
    if len(base) >= 2 and len(hyb):
        hyb["snapkv_at_same_traffic"] = np.interp(hyb[x], base[x], base[metric])
        hyb["in_range"] = (hyb[x] >= base[x].min()) & (hyb[x] <= base[x].max())
        hyb["delta"] = hyb[metric] - hyb["snapkv_at_same_traffic"]
        cols = ["task", "method", x, metric, "snapkv_at_same_traffic", "delta", "in_range"]
        hyb[cols].sort_values(x).to_csv(csv_path.parent / "matched.csv", index=False)
        print(f"\n== {csv_path} ({'oracle: lower is better' if oracle else 'e2e: higher is better'})")
        print(hyb[cols].sort_values(x).to_string(index=False))
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(base[x], base[metric], "k-o", label="SnapKV (dense decode)")
    for dec, mk in (("sys", "s"), ("top", "^")):
        sub = hyb[hyb.decoder == dec] if len(hyb) else hyb
        for b, g in sub.groupby("budget"):
            ax.plot(g[x], g[metric], mk, linestyle="--", label=f"SnapKV{b} + {dec}")
    full = df[(df.budget == "full") & (df.decoder == "dense")]
    if len(full):
        ax.axhline(float(full[metric].iloc[0]), color="gray", lw=0.8, ls=":", label="full KV")
    ax.set_xscale("log")
    ax.set_xlabel("K+V rows read per KV head per decode step")
    ax.set_ylabel(metric)
    ax.set_title(f"{df.task.iloc[0]} ({'oracle' if oracle else 'end-to-end'})")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(csv_path.parent / f"pareto_{metric}.png", dpi=150)
    plt.close(fig)


def main():
    root = pathlib.Path(sys.argv[1])
    for p in sorted(root.rglob("*summary.csv")):
        analyse(p)


if __name__ == "__main__":
    main()
