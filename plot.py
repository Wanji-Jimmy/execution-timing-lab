"""Plot reserved validation outcomes and two diagnosed failure cases."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from adaptive import METHODS, _records, release_path, replay

ROOT = Path(__file__).resolve().parent
LABELS = ("Equal quantities", "Historical profile", "Fixed early", "Volume feedback")
COLORS = ("#8292a1", "#477998", "#bf8036", "#166b59")


def main():
    results = json.loads((ROOT / "adaptive_results.json").read_text())
    with (ROOT / "adaptive_daily.csv").open(newline="") as stream:
        rows = [r for r in csv.DictReader(stream)
                if r["period"] == "validation" and float(r["cap"]) == .01]
    groups = results["period_results"]["validation"]
    assets = ("BTCUSDT", "ETHUSDT")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    for j, (method, label, color) in enumerate(zip(METHODS, LABELS, COLORS)):
        x = np.arange(2) + (j - 1.5) * .20
        values = [groups[a]["0.01"]["methods"][method]["days_completed"] for a in assets]
        bars = axes[0, 0].bar(x, values, width=.19, color=color, label=label)
        axes[0, 0].bar_label(bars, fmt="%d", fontsize=9, padding=3)
        values = [groups[a]["0.01"]["methods"][method]["mean_unfilled_pct"] for a in assets]
        bars = axes[0, 1].bar(x, values, width=.19, color=color)
        axes[0, 1].bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
    for ax in axes[0]:
        ax.set_xticks([0, 1], [f"{a[:3]} ({groups[a]['0.01']['days']} days)" for a in assets])
        ax.grid(axis="y", alpha=.17)
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(ax.get_ylim()[1] * 1.18, .1))
    axes[0, 0].set_title("Orders completed by 16:00", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Completed days")
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0, 1].set_title("Average order left unfilled", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Percent of the original quantity")

    feedback = [r for r in rows if r["method"] == "feedback"]
    timing = [r for r in feedback if float(r["total_shortage_fraction"]) <= 1e-12]
    shortage = [r for r in feedback if float(r["total_shortage_fraction"]) > 1e-12]
    cases = [max(timing, key=lambda r: float(r["timing_shortage_fraction"])),
             max(shortage, key=lambda r: float(r["total_shortage_fraction"]))]
    contexts, _ = _records(ROOT / "data/buckets.csv", ROOT / "data/validation_buckets.csv")
    for ax, row, title in zip(axes[1], cases,
                             ("Enough total capacity, but too late", "Not enough total capacity")):
        asset, day = row["asset"], row["date"]
        record = next(r for r in contexts if r["asset"] == asset and str(r["date"]) == day)
        q = record["quantity"]
        time = np.linspace(8, 16, len(record["volumes"]) + 1)
        for method, color in zip(METHODS, COLORS):
            path = release_path(record["weights"], record["volumes"], record["mean_volumes"],
                                q, .01, method, results["fixed_alpha"][asset]["0.01"])
            played = replay(path, record["prices"], record["volumes"], q, .01,
                            record["arrival_price"])
            ax.step(time, np.r_[0, 100 * played["cumulative_fills"] / q],
                    where="post", color=color, linewidth=2,
                    linestyle="--" if method == "feedback" else "-")
        bound = np.minimum(100, 100 * .01 * np.cumsum(record["volumes"]) / q)
        ax.step(time, np.r_[0, bound], where="post", color="#303030", linestyle=":",
                linewidth=1.4, label="Immediate-release capacity bound")
        ax.set_title(f"{title}\n{asset[:3]} / {day}", loc="left", fontsize=11, fontweight="bold")
        ax.text(.02, .96, f"Feedback leaves {100 * float(row['unfilled_fraction']):.2f}% unfilled",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .85})
        ax.set(xlim=(8, 16), ylim=(0, 110), xlabel="UTC hour", ylabel="Order completed (%)")
        ax.set_xticks([8, 10, 12, 14, 16])
        ax.grid(alpha=.15)
        ax.legend(loc="lower right", frameon=False, fontsize=8)

    fig.suptitle("Does volume feedback help?", x=.065, y=.985, ha="left",
                 fontsize=22, fontweight="bold", color="#243445")
    fig.text(.065, .945, "1–23 October 2026  |  Same order quantity and 1% participation cap  |  Rules fixed before validation",
             fontsize=10, color="#52606b")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.06, .922),
               ncol=4, frameon=False, fontsize=10)
    fig.text(.065, .025,
             "Bottom examples: largest timing-only residual and largest total-capacity shortage, selected after evaluation.\n"
             "Assumed bucket-VWAP fills; no actual fills, spread, fees or own impact. Source: Binance Vision (CC BY-NC-SA 4.0).",
             fontsize=8.5, color="#52606b")
    fig.tight_layout(rect=(.025, .075, .99, .87), h_pad=3, w_pad=3)
    fig.savefig(ROOT / "summary.png", dpi=180)
    print(ROOT / "summary.png")


if __name__ == "__main__":
    main()
