#!/usr/bin/env python3
"""Plot the execution time and the efficiency of an iPIC3D scaling campaign.

Reads the stdout_N<nodes>_<job> files that submit.sh leaves in the campaign
folders. The execution time of a run is the sum of its measured cycles, all
but the first (warm-up) one, from the per-cycle times of rank 0; the field
solve synchronizes all ranks every cycle. Prints a table and writes into the
first campaign folder, each figure as PNG and PDF:

  <kind>_scaling_efficiency       efficiency, with its value at each point
  <kind>_scaling_time_efficiency  execution time and efficiency, no title
  <kind>_scaling.csv              the numbers of the table

Several campaigns of one system and kind are combined.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, ScalarFormatter

# Font sizes in points for all plots.
FONT_SIZES = {
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 24,
    "legend.fontsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
}
ANNOTATION_SIZE = 16
BLUE = "#176b87"
RED = "#c23b22"

HEADER = re.compile(
    r"^Scaling run: system=(\S+) kind=(\S+) nodes=(\d+) ranks=(\d+)"
    r" gpu_label=(\S+)",
    re.M,
)
CYCLE = re.compile(r"^Execution time cycle: (\S+) ms", re.M)
FINISHED = re.compile(r"^\s*Cycle loop\s*: \S+ sec over (\d+) cycles", re.M)

matplotlib.rcParams.update(FONT_SIZES)


def read_run(path):
    """Return the measurements of one stdout file, or None if unfinished."""
    text = path.read_text(errors="replace")
    header = HEADER.search(text)
    finished = FINISHED.search(text)
    if header is None or finished is None:
        return None
    seconds = [float(ms) / 1000 for ms in CYCLE.findall(text)]
    if len(seconds) != int(finished[1]) or len(seconds) < 2:
        sys.exit(f"{path}: {len(seconds)} cycle times for {finished[1]} cycles")
    system, kind, nodes, ranks, gpu_label = header.groups()
    return {
        "system": system,
        "kind": kind,
        "gpu_label": gpu_label,
        "nodes": int(nodes),
        "gpus": int(ranks),
        "cycles": len(seconds) - 1,
        "seconds": sum(seconds[1:]),
        "stdout": str(path),
    }


def read_campaigns(folders, keep):
    runs = {}
    for folder in folders:
        paths = sorted(folder.glob("stdout_N*"))
        if not paths:
            sys.exit(f"No stdout_N* files in {folder}")
        for path in paths:
            run = read_run(path)
            if run is None:
                print(f"Skipped, not finished: {path}")
            elif keep(run["nodes"]):
                if run["nodes"] in runs:
                    sys.exit(
                        f"Two runs on {run['nodes']} nodes: "
                        f"{runs[run['nodes']]['stdout']} and {path}"
                    )
                runs[run["nodes"]] = run
    if not runs:
        sys.exit("No finished runs to plot")
    runs = [runs[nodes] for nodes in sorted(runs)]
    for field in ("system", "kind", "cycles"):
        if len({run[field] for run in runs}) > 1:
            sys.exit(f"The runs differ in {field}: plot them separately")
    return runs


def add_efficiency(runs):
    """Efficiency in percent against the smallest run."""
    base = runs[0]
    for run in runs:
        ratio = base["seconds"] / run["seconds"]
        if run["kind"] == "strong":
            ratio *= base["gpus"] / run["gpus"]
        run["efficiency"] = 100 * ratio


def set_gpu_axis(ax, runs):
    gpus = [run["gpus"] for run in runs]
    ax.set_xscale("log", base=2)
    ax.set_xticks(gpus)
    ax.set_xticklabels(gpus)
    ax.set_xlabel(f"Number of {runs[0]['gpu_label']}")
    ax.grid(True, which="major", linestyle=":", alpha=0.6)


def save(fig, stem):
    fig.tight_layout()
    for suffix in (".png", ".pdf"):
        fig.savefig(stem.with_suffix(suffix), dpi=300, bbox_inches="tight")
        print(f"Saved plot to {stem.with_suffix(suffix)}")
    plt.close(fig)


def plot_efficiency(runs, stem):
    gpus = [run["gpus"] for run in runs]
    efficiencies = [run["efficiency"] for run in runs]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(gpus, efficiencies, color=BLUE, marker="o", markersize=7,
            linewidth=2, label="Measured efficiency")
    ax.axhline(100, color=RED, linestyle="--", linewidth=1.5,
               label="Ideal efficiency")
    for gpu, efficiency in zip(gpus, efficiencies):
        ax.annotate(f"{efficiency:.1f}%", (gpu, efficiency), xytext=(0, -18),
                    textcoords="offset points", ha="center",
                    fontsize=ANNOTATION_SIZE)
    set_gpu_axis(ax, runs)
    ax.set_ylim(max(0, min(efficiencies) - 3), max(102, max(efficiencies) + 3))
    ax.set_ylabel("Parallel Efficiency (%)")
    ax.set_title(f"iPIC3D {runs[0]['kind'].capitalize()} Scaling "
                 f"({runs[0]['cycles']} measured cycles)")
    ax.legend()
    save(fig, stem)


def plot_time_efficiency(runs, stem):
    gpus = [run["gpus"] for run in runs]
    seconds = [run["seconds"] for run in runs]
    efficiencies = [run["efficiency"] for run in runs]
    fig, time_ax = plt.subplots(figsize=(10, 6))
    efficiency_ax = time_ax.twinx()
    time_line, = time_ax.plot(gpus, seconds, color=BLUE, marker="o",
                              linestyle="-", markersize=8, linewidth=2,
                              label="Execution time")
    efficiency_line, = efficiency_ax.plot(gpus, efficiencies, color=RED,
                                          marker="^", linestyle="--",
                                          markersize=8, linewidth=2,
                                          label="Scaling efficiency")
    for gpu, efficiency in zip(gpus, efficiencies):
        efficiency_ax.annotate(f"{efficiency:.1f}%", (gpu, efficiency),
                               xytext=(0, -20), textcoords="offset points",
                               ha="center", color=RED,
                               fontsize=ANNOTATION_SIZE, fontweight="bold")
    set_gpu_axis(time_ax, runs)
    time_ax.set_ylabel("Execution time (s)")
    if runs[0]["kind"] == "strong":
        time_ax.set_yscale("log")
        time_ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        time_ax.yaxis.set_minor_locator(NullLocator())
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        time_ax.yaxis.set_major_formatter(formatter)
        time_ax.set_ylim(0.95 * min(seconds), 1.05 * max(seconds))
    else:
        time_ax.set_ylim(0, 1.05 * max(seconds))
    efficiency_ax.set_ylabel("Parallel Efficiency (%)")
    efficiency_ax.set_ylim(0, max(110, 1.1 * max(efficiencies)))
    efficiency_ax.legend(handles=[time_line, efficiency_line])
    save(fig, stem)


def main():
    parser = argparse.ArgumentParser(
        description="Plot an iPIC3D scaling campaign written by submit.sh."
    )
    parser.add_argument("campaigns", nargs="+", type=Path,
                        help="results_* folders of one system and kind")
    parser.add_argument("--min-nodes", type=int, default=1,
                        help="leave out smaller node counts")
    parser.add_argument("--max-nodes", type=int, default=sys.maxsize,
                        help="leave out larger node counts")
    parser.add_argument("--skip-nodes", type=int, nargs="+", default=[],
                        help="node counts to leave out, such as a slow node")
    args = parser.parse_args()

    def keep(nodes):
        return (args.min_nodes <= nodes <= args.max_nodes
                and nodes not in args.skip_nodes)

    runs = read_campaigns(args.campaigns, keep)
    add_efficiency(runs)
    kind, label = runs[0]["kind"], runs[0]["gpu_label"]

    print(f"{kind.capitalize()} scaling on {runs[0]['system']}, "
          f"{runs[0]['cycles']} measured cycles")
    for run in runs:
        print(f"{run['nodes']:4d} nodes ({run['gpus']:5d} {label}): "
              f"{run['seconds']:.6f} s, efficiency = {run['efficiency']:.2f}%")

    folder = args.campaigns[0]
    columns = ["nodes", "gpus", "cycles", "seconds", "efficiency", "stdout"]
    with open(folder / f"{kind}_scaling.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    plot_efficiency(runs, folder / f"{kind}_scaling_efficiency")
    plot_time_efficiency(runs, folder / f"{kind}_scaling_time_efficiency")


if __name__ == "__main__":
    main()
