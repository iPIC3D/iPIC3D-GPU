import argparse
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GPUS_PER_NODE = 8
STDOUT_PATTERN = re.compile(r"^stdout_N(?P<nodes>\d+)(?:_|$)")
CYCLE_LOOP_PATTERN = re.compile(
    r"^\s*Cycle loop\s*:\s*"
    r"(?P<seconds>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s+sec\s+"
    r"over\s+(?P<cycles>\d+)\s+cycles\s*$",
    re.MULTILINE,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Plot weak-scaling efficiency from iPIC3D stdout files."
    )
    parser.add_argument(
        "results_folder",
        type=Path,
        help="folder containing files named stdout_N<NODES>_<JOB_ID>",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        help="minimum node count to include (inclusive)",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        help="maximum node count to include (inclusive)",
    )
    args = parser.parse_args()
    if args.min_nodes is not None and args.min_nodes < 1:
        parser.error("--min-nodes must be positive")
    if args.max_nodes is not None and args.max_nodes < 1:
        parser.error("--max-nodes must be positive")
    if (
        args.min_nodes is not None
        and args.max_nodes is not None
        and args.min_nodes > args.max_nodes
    ):
        parser.error("--min-nodes cannot be greater than --max-nodes")
    return args


def load_measurements(results_folder, min_nodes=None, max_nodes=None):
    if not results_folder.is_dir():
        raise ValueError(f"Results folder does not exist: {results_folder}")

    measurements = {}
    cycle_counts = set()
    stdout_paths = sorted(results_folder.glob("stdout_N*"))
    if not stdout_paths:
        raise ValueError(f"No stdout_N* files found in {results_folder}")

    for stdout_path in stdout_paths:
        filename_match = STDOUT_PATTERN.match(stdout_path.name)
        if filename_match is None or not stdout_path.is_file():
            continue

        node_count = int(filename_match.group("nodes"))
        if min_nodes is not None and node_count < min_nodes:
            continue
        if max_nodes is not None and node_count > max_nodes:
            continue
        if node_count in measurements:
            raise ValueError(
                f"Multiple stdout files found for {node_count} nodes"
            )

        output = stdout_path.read_text(errors="replace")
        timing_match = CYCLE_LOOP_PATTERN.search(output)
        if timing_match is None:
            raise ValueError(f"Cycle loop timing not found in {stdout_path}")

        cycle_loop_seconds = float(timing_match.group("seconds"))
        if cycle_loop_seconds <= 0.0:
            raise ValueError(
                f"Cycle loop timing must be positive in {stdout_path}: "
                f"{cycle_loop_seconds}"
            )

        measurements[node_count] = cycle_loop_seconds
        cycle_counts.add(int(timing_match.group("cycles")))

    if not measurements:
        raise ValueError(f"No valid stdout_N* files found in {results_folder}")
    if len(cycle_counts) != 1:
        raise ValueError(
            "Measurements use different cycle counts: "
            + ", ".join(str(count) for count in sorted(cycle_counts))
        )

    nodes = sorted(measurements)
    cycle_loop_seconds = [measurements[node_count] for node_count in nodes]
    return nodes, cycle_loop_seconds, cycle_counts.pop()


def main():
    args = parse_arguments()
    results_folder = args.results_folder.expanduser().resolve()
    nodes, cycle_loop_seconds, cycle_count = load_measurements(
        results_folder,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
    )

    baseline_time = cycle_loop_seconds[0]
    efficiencies = [
        100.0 * baseline_time / cycle_time
        for cycle_time in cycle_loop_seconds
    ]
    accelerator_counts = [
        node_count * GPUS_PER_NODE for node_count in nodes
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        accelerator_counts,
        efficiencies,
        color="#176b87",
        marker="o",
        markersize=7,
        linewidth=2,
        label="Measured efficiency",
    )
    ax.axhline(
        100.0,
        color="#c23b22",
        linestyle="--",
        linewidth=1.5,
        label="Ideal efficiency",
    )

    for accelerator_count, efficiency in zip(
        accelerator_counts, efficiencies
    ):
        ax.annotate(
            f"{efficiency:.1f}%",
            (accelerator_count, efficiency),
            xytext=(0, -16),
            textcoords="offset points",
            ha="center",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks(accelerator_counts)
    ax.set_xticklabels(accelerator_counts)
    ax.set_ylim(
        max(0.0, min(efficiencies) - 3.0),
        max(102.0, max(efficiencies) + 3.0),
    )
    ax.set_xlabel("Number of GPUs/GCDs")
    ax.set_ylabel("Weak scaling efficiency (%)")
    ax.set_title(
        f"iPIC3D Weak Scaling Efficiency ({cycle_count} cycles)"
    )
    ax.grid(True, which="major", linestyle=":", alpha=0.6)
    ax.legend()

    output_path = results_folder / "weak_scaling_efficiency.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    for nodes, cycle_time, efficiency in zip(
        nodes, cycle_loop_seconds, efficiencies
    ):
        print(
            f"{nodes:3d} nodes ({nodes * GPUS_PER_NODE:4d} GPUs): "
            f"{cycle_time:.4f} s, efficiency = {efficiency:.2f}%"
        )
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
