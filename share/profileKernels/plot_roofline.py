#!/usr/bin/env python3
"""Rooflines and speed of light of a profile.sh run (README.md).

Offline: reads the run folder and launches nothing. Prints the report of the
run (the device's peak and measured rates, the kernels' rates) and writes to
RUN/processed/ report.txt, times.csv (the timed runs), ceilings.csv (the
device's rates), points.csv (every profiled launch: rates, roofline
coordinates and speed of light) and, with matplotlib, the figures, each named
after the GPU (<name>_<device>.png/pdf, e.g. roofline_rtx5080.png):
roofline (DRAM, peak and measured ceilings), roofline_hierarchical (L1, L2,
DRAM, Nsight Compute peaks), speed_of_light, roofline_paper (the DRAM
roofline in the style of the article_veil rooflines),
speed_of_light_paper_ppc<N> (bars: SM, memory, FP64 pipe and L2 atomics of
the kernels at one particles per cell) and
speed_of_light_paper_breakdown_ppc<N> (the same with L1, L2, DRAM and MIO in
place of SM and memory).
Python >= 3.9.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

# Text sizes of the figures, in points.
TEXT_SIZES = {"default": 18, "axis_label": 20, "title": 18, "legend": 15,
              "tick": 16}
# Colour and label of each kernel: blue, orange and purple (the purple of
# the article_veil rooflines, lightened to keep the three apart for colour-blind
# readers).
KERNEL_STYLES = {
    "moverSubcyclesKernel": ("#0072B2", "Relativistic mover"),
    "cellAwareMomentKernel": ("#E69F00", "Sorted deposit"),
    "momentKernelNew": ("#7A0FA6", "Unsorted deposit"),
}
# Colour and label of each memory level of the hierarchical roofline.
LEVEL_STYLES = {"dram": ("#0072B2", "DRAM"), "l2": ("#E69F00", "L2"),
                "l1": ("#009E73", "L1")}
# Markers of the particles per cell in the rooflines, in increasing order,
# never repeated.
PPC_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "<", ">", "h", "*", "p", "8")
# Colours of the particles per cell in the speed-of-light bars: one blue,
# light to dark, spread over the values of the run.
PPC_RAMP = ("#84b9f3", "#6aabf1", "#519ced", "#338ced", "#257de0", "#2370cd",
            "#1b64be", "#1757ac", "#0f4a96", "#0e3d7e")
# The FP64 work of the rooflines: the conventional FP64 pipe only (iPIC3D
# uses no tensor cores), its peak that of the DFMA pipe.
WORK = ("FP64 FLOPs = DADD + DMUL + 2 DFMA of the active threads, "
        "FP64 pipe peak (no tensor cores; L2 atomic adds not counted)")

# The roofline for papers (roofline_paper_<device>), in the style of the
# article_veil rooflines (article_v0/roofline/roofline_ncu_ops.py): size in
# inches, text sizes in points, and colour and line of the FP64 roof.
PAPER_FIGSIZE = (11, 8)
PAPER_TEXT_SIZES = {"default": 22, "axis_label": 26, "legend": 21, "legend_title": 22,
                    "tick": 21}
PAPER_ROOF_STYLE = ("0.35", "--")
# Margin, as a factor, between a legend and the roofline or the bars below it.
PAPER_HEADROOM = 1.15
# The speed of light for papers (speed_of_light_paper_ppc<N>_<device>), on
# the canvas of the paper roofline: its groups of bars (columns of points.csv,
# left to right), the particles per cell it shows (the largest of the run when
# the run has not these), and the width of a group of bars (1: no gap).  Its
# breakdown (speed_of_light_paper_breakdown_ppc<N>_<device>): MIO and the
# memory levels in place of the two headlines, the group names diagonal.
PAPER_SOL_ROWS = ("sm_pct", "memory_pct", "fp64_pipe_pct", "l2_atomic_pct")
PAPER_SOL_BREAKDOWN_ROWS = ("fp64_pipe_pct", "mio_pct", "l1_pct", "l2_pct", "dram_pct",
                            "l2_atomic_pct")
PAPER_SOL_PPC = 256
PAPER_SOL_GROUP = 0.8
# Words of a GPU name left out of the device part of the file names.
VENDOR_WORDS = {"nvidia", "geforce", "tesla", "quadro", "amd", "instinct", "radeon"}

# Nsight Compute metrics read from the counters.
FP64_OPS = ("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
TIME = "gpu__time_duration.sum"
BYTES = {"dram": "dram__bytes.sum", "l2": "lts__t_bytes.sum",
         "l1": "l1tex__t_bytes.sum"}
ATOMICS = ("lts__t_requests_op_atom.sum", "lts__t_requests_op_red.sum")
PEAK_FP64 = ("sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained",
             "sm__cycles_elapsed.avg.per_second")
PEAK_BYTES = {
    "dram": ("dram__bytes.sum.peak_sustained", "dram__cycles_elapsed.avg.per_second"),
    "l2": ("lts__t_bytes.sum.peak_sustained", "lts__cycles_elapsed.avg.per_second"),
    "l1": ("l1tex__t_bytes.sum.peak_sustained", "l1tex__cycles_elapsed.avg.per_second"),
}
# Speed of light: column, label, metric (percent of the peak of the unit).
SOL = (
    ("sm_pct", "SM", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("memory_pct", "Memory",
     "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("l1_pct", "L1", "l1tex__throughput.avg.pct_of_peak_sustained_active"),
    ("l2_pct", "L2", "lts__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("dram_pct", "DRAM", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("fp64_pipe_pct", "FP64 pipe",
     "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("l2_atomic_pct", "L2 atomics",
     "lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed"),
)
# MIO, the SM's way to its shared units (shared memory, shuffles, special
# math, dynamic branches): column, metric of its four interfaces (percent of
# peak).  Its column (mio_pct, label MIO) is the busiest of the four, as the
# SM headline is the busiest SM unit.
MIO = (
    ("mio_inst_issued_pct", "sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed"),
    ("mio_pq_write_pct", "sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("mio_pq_read_pct", "sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed"),
    ("mio_writeback_pct", "sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed"),
)
# Occupancy and launch: column, metric.
LAUNCH = (
    ("achieved_occupancy_pct", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("theoretical_occupancy_pct", "sm__maximum_warps_per_active_cycle_pct"),
    ("registers_per_thread", "launch__registers_per_thread"),
)
# The units of the speed-of-light breakdowns: the metrics with this suffix,
# but the aggregates of the headlines and the tensor pipes (unused); the
# busiest TOP_UNITS are listed.
UNIT_SUFFIX = ".avg.pct_of_peak_sustained_elapsed"
AGGREGATES = ("gpu__", "sm__throughput.", "lts__throughput.")
TOP_UNITS = 3
# Seconds per unit of the duration; the other metrics are in base units.
TIME_UNITS = {"ns": 1e-9, "nsecond": 1e-9, "us": 1e-6, "usecond": 1e-6,
              "ms": 1e-3, "msecond": 1e-3, "s": 1.0, "second": 1.0}
BASE_UNITS = {"", "byte", "inst", "request", "hz", "byte/cycle", "inst/cycle",
              "%", "register/thread"}


def json_lines(path, prefix):
    """The lines of the file that start with prefix, as JSON."""
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.startswith(prefix)]


def read_result(path, kernel):
    """The result line of a profile (KernelProfile::finish), checked: one per
    run, so one per counter pass under application replay."""
    results = json_lines(path, '{"kernel":')
    if not results:
        raise ValueError(f"{path}: no result line")
    for result in results:
        if result["kernel"] != kernel:
            raise ValueError(f"{path}: result of {result['kernel']}, expected {kernel}")
        if result["check"] is not True:
            raise ValueError(f"{path}: the check of the profile failed")
    return results[0]


def read_counters(path, kernel):
    """The metrics of the one launch of kernel in an ncu --page raw --csv
    --print-units base export (header, units, one row per launch), and its
    busiest speed-of-light units."""
    with Path(path).open(newline="") as f:
        rows = list(csv.reader(f))
    header, units = rows[0], dict(zip(rows[0], rows[1]))
    launches = [dict(zip(header, r)) for r in rows[2:]
                if r[header.index("Kernel Name")].split("(")[0] == kernel]
    if len(launches) != 1:
        raise ValueError(f"{path}: {len(launches)} launches of {kernel}, expected 1")
    launch = launches[0]
    wanted = (*FP64_OPS, TIME, *BYTES.values(), *ATOMICS, *PEAK_FP64,
              *(m for pair in PEAK_BYTES.values() for m in pair),
              *(m for _, _, m in SOL), *(m for _, m in MIO), *(m for _, m in LAUNCH))
    values = {}
    for metric in wanted:
        if metric not in launch:
            raise ValueError(f"{path}: no metric {metric}")
        try:
            value = float(launch[metric])  # C locale: no digit grouping
        except ValueError:
            raise ValueError(f"{path}: {metric} = {launch[metric]!r}") from None
        unit = units[metric]
        if metric == TIME:
            value *= TIME_UNITS[unit]
        elif unit not in BASE_UNITS:
            raise ValueError(f"{path}: {metric} in {unit!r}, not a base unit")
        values[metric] = value
    busiest = []
    for metric, text in launch.items():
        if (metric.endswith(UNIT_SUFFIX) and not metric.startswith(AGGREGATES)
                and "internal_activity" not in metric and "tensor" not in metric):
            try:
                busiest.append((float(text), metric[:-len(UNIT_SUFFIX)]))
            except ValueError:
                pass
    busiest.sort(reverse=True)
    return values, "; ".join(f"{m} {v:.1f}" for v, m in busiest[:TOP_UNITS])


def workload(entry, result):
    """The columns of a point that its result line gives."""
    return dict(profile=entry["profile"], kernel=entry["kernel"], ppc=int(entry["ppc"]),
                cells="x".join(map(str, result["cells"])), species=result["species"],
                order=result["order"], parameter=result["parameter"],
                particles=result["launch_particles"])


def point(entry, repetition, result, counters, busiest, roofs):
    """One profiled launch: work, traffic, time, roofline and speed of light."""
    dadd, dmul, dfma = (counters[m] for m in FP64_OPS)
    flops = dadd + dmul + 2 * dfma
    seconds = counters[TIME]
    if flops <= 0 or seconds <= 0:
        raise ValueError(f"{entry['directory']}: no FP64 work or no duration")
    particles = result["launch_particles"]
    row = workload(entry, result)
    row.update(repetition=repetition, seconds=seconds,
               ns_per_particle=seconds * 1e9 / particles, flops=flops,
               flops_per_particle=flops / particles,
               gflops_per_second=flops / seconds / 1e9,
               atomic_requests_per_particle=sum(counters[m] for m in ATOMICS) / particles)
    for level, metric in BYTES.items():
        row[f"{level}_bytes"] = counters[metric]
        row[f"{level}_gbytes_per_second"] = counters[metric] / seconds / 1e9
        row[f"{level}_flops_per_byte"] = flops / counters[metric]
    # the device's peaks at the clocks of the launch (Nsight Compute)
    row["sm_clock_ghz"] = counters[PEAK_FP64[1]] / 1e9
    row["ncu_fp64_gflops_per_second"] = (2 * counters[PEAK_FP64[0]]
                                        * counters[PEAK_FP64[1]] / 1e9)
    for level, (rate, clock) in PEAK_BYTES.items():
        row[f"ncu_{level}_gbytes_per_second"] = counters[rate] * counters[clock] / 1e9
    row["fp64_peak_pct"] = 100 * row["gflops_per_second"] / row["ncu_fp64_gflops_per_second"]
    row["dram_peak_pct"] = (100 * row["dram_gbytes_per_second"]
                            / row["ncu_dram_gbytes_per_second"])
    # the share of the DRAM roofline bound reached (measured ceilings)
    bound = min(roofs["measured_fp64_gflops_per_second"],
                row["dram_flops_per_byte"] * roofs["measured_bandwidth_gbytes_per_second"])
    row["dram_roofline_pct"] = 100 * row["gflops_per_second"] / bound
    for column, _, metric in SOL:
        row[column] = counters[metric]
    row["mio_pct"] = max(counters[metric] for _, metric in MIO)
    for column, metric in MIO:
        row[column] = counters[metric]
    row["busiest_units"] = busiest
    for column, metric in LAUNCH:
        row[column] = counters[metric]
    return row


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def process(run, output, partial=False, ppcs=None, plot=True):
    run = Path(run).resolve()
    if not partial and not (run / "COMPLETE").is_file():
        raise ValueError("the collection is not complete: --partial uses the points so far")
    # points.tsv lists a point once all its runs are done
    with (run / "points.tsv").open(newline="") as f:
        entries = list(csv.DictReader(f, delimiter="\t"))
    if ppcs:
        entries = [e for e in entries if int(e["ppc"]) in ppcs]
    if not entries:
        raise ValueError("no points selected")
    ceilings = json_lines(run / "ceilings.log", '{"device":')
    if len(ceilings) != 1:
        raise ValueError(f"{run / 'ceilings.log'}: {len(ceilings)} results, expected 1")
    ceilings = ceilings[0]
    roofs = dict(device=ceilings["device"], multiprocessors=ceilings["multiprocessors"],
                 l2_bytes=ceilings["l2_bytes"],
                 measured_bandwidth_gbytes_per_second=ceilings["bandwidth_bytes_per_second"] / 1e9,
                 measured_fp64_gflops_per_second=ceilings["fp64_flops_per_second"] / 1e9)

    times, points, builds, workloads = [], [], set(), set()

    def checked(path, entry):
        result = read_result(path, entry["kernel"])
        if result["gpu"] != ceilings["device"]:
            raise ValueError(f"{path}: {result['gpu']}, ceilings of {ceilings['device']}")
        if result["ppc"] != int(entry["ppc"]):
            raise ValueError(f"{path}: ppc {result['ppc']}, points.tsv {entry['ppc']}")
        builds.add(result["build"])
        workloads.add((tuple(result["cells"]), result["species"]))
        return result

    for entry in entries:
        directory = run / entry["directory"]
        result = checked(directory / "timed.log", entry)
        times.append({**workload(entry, result),
                      **{k: result[k] for k in ("launches", "min_ms", "median_ms",
                                                "mean_ms", "max_ms", "ns_per_particle")}})
        for r in range(1, int(entry["repetitions"]) + 1):
            result = checked(directory / f"profiled{r}.log", entry)
            counters, busiest = read_counters(directory / f"counters{r}.csv",
                                              entry["kernel"])
            points.append(point(entry, r, result, counters, busiest, roofs))
    if len(builds) != 1 or len(workloads) != 1:
        raise ValueError("the points come from different builds, grids or species")
    if points:
        for key in ("sm_clock_ghz", "ncu_fp64_gflops_per_second",
                    *(f"ncu_{level}_gbytes_per_second" for level in BYTES)):
            roofs[key] = statistics.median(p[key] for p in points)
    roofs["build"] = builds.pop()
    cells, species = workloads.pop()

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    text = report(times, points, roofs, f"{'x'.join(map(str, cells))} cells, species {species}")
    (output / "report.txt").write_text(text)
    write_csv(output / "times.csv", times)
    with (output / "ceilings.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(("quantity", "value"))
        writer.writerows(roofs.items())
    if points:
        write_csv(output / "points.csv", points)
        if plot:
            caption = (f"{roofs['device']}, {'x'.join(map(str, cells))} cells, species "
                       f"{species}" + (" (preliminary)" if partial else ""))
            plot_figures(points, roofs, caption, output)
    return text, times, points


def report(times, points, roofs, workload_text):
    """The device's rates and the kernels' rates, as text."""
    lines = [f"{roofs['device']}: {roofs['multiprocessors']} SMs, "
             f"L2 {roofs['l2_bytes'] / 2**20:.0f} MiB; {workload_text}",
             f"build: {roofs['build']}", "",
             "Device rates      measured (profileCeilings)"
             + ("   peak (Nsight Compute, SM at "
                f"{roofs['sm_clock_ghz']:.2f} GHz)" if points else "")]
    measured = (("FP64 (DFMA pipe)", f"{roofs['measured_fp64_gflops_per_second']:9.0f} GFLOP/s",
                 "ncu_fp64_gflops_per_second", "GFLOP/s"),
                ("DRAM", f"{roofs['measured_bandwidth_gbytes_per_second']:9.0f} GB/s   ",
                 "ncu_dram_gbytes_per_second", "GB/s"),
                ("L2", " " * 17, "ncu_l2_gbytes_per_second", "GB/s"),
                ("L1", " " * 17, "ncu_l1_gbytes_per_second", "GB/s"))
    for name, value, peak, unit in measured:
        if points or value.strip():
            lines.append(f"  {name:16s}{value}"
                         + (f"            {roofs[peak]:9.0f} {unit}" if points else ""))
    lines += ["", "Kernels: time per particle of the timed runs (median)"
              + ("; rates of the profiled launches (mean), each percent of the"
                 " peak at its own SM clock" if points else ""),
              f"  {'kernel':22s} {'ppc':>4s} {'ns/particle':>11s}"
              + (f" {'SM GHz':>6s} {'GFLOP/s':>8s} {'% FP64':>6s} {'DRAM GB/s':>9s}"
                 f" {'% DRAM':>6s}  busiest units (% of peak)" if points else "")]
    for t in times:
        line = f"  {t['kernel']:22s} {t['ppc']:4d} {t['ns_per_particle']:11.3f}"
        group = [p for p in points if p["kernel"] == t["kernel"] and p["ppc"] == t["ppc"]]
        if group:
            mean = {k: statistics.mean(p[k] for p in group)
                    for k in ("sm_clock_ghz", "gflops_per_second", "fp64_peak_pct",
                              "dram_gbytes_per_second", "dram_peak_pct")}
            line += (f" {mean['sm_clock_ghz']:6.2f}"
                     f" {mean['gflops_per_second']:8.0f} {mean['fp64_peak_pct']:6.1f}"
                     f" {mean['dram_gbytes_per_second']:9.0f} {mean['dram_peak_pct']:6.1f}"
                     f"  {group[0]['busiest_units']}")
        lines.append(line)
    return "\n".join(line.rstrip() for line in lines) + "\n"


def means(points, kernel, keys):
    """Per ppc, the means of keys over the repetitions."""
    by_ppc = {}
    for p in points:
        if p["kernel"] == kernel:
            by_ppc.setdefault(p["ppc"], []).append(p)
    return [(ppc, [statistics.mean(p[k] for p in group) for k in keys])
            for ppc, group in sorted(by_ppc.items())]


def device_tag(device):
    """The GPU in the figure names: NVIDIA GeForce RTX 5080 -> rtx5080, NVIDIA
    GH200 480GB -> gh200, NVIDIA A100-SXM4-40GB -> a100."""
    tag = ""
    for word in device.replace("-", " ").split():
        if word.lower() in VENDOR_WORDS:
            continue
        tag += word
        if any(c.isdigit() for c in word):
            break
    return "".join(c for c in tag.lower() if c.isalnum()) or "gpu"


def plot_figures(points, roofs, caption, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    def roof(ax, bandwidth, peak, low, high, **style):
        ridge = peak / bandwidth
        return ax.loglog([low, ridge, high], [bandwidth * low, peak, peak], **style)[0]

    def limits(xs, ridges):
        return min(min(xs), *ridges) / 3, max(max(xs), *ridges) * 3

    def draw(ax, kernel, x_key, color):
        """The mean of the repetitions at each ppc, joined, one marker per ppc."""
        data = means(points, kernel, (x_key, "gflops_per_second"))
        ax.plot([xy[0] for _, xy in data], [xy[1] for _, xy in data], color=color,
                alpha=0.6)
        for ppc, (x, y) in data:
            ax.plot(x, y, color=color, marker=markers[ppc], markersize=10,
                    linestyle="none")

    def ppc_legend(ax, loc):
        ax.add_artist(ax.legend(handles=[
            Line2D([], [], color="0.3", marker=markers[ppc], markersize=10,
                   linestyle="none", label=str(ppc)) for ppc in markers],
            title="particles per cell", loc=loc))

    def save(fig, name):
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}_{tag}.{extension}", dpi=180)
        plt.close(fig)

    tag = device_tag(roofs["device"])
    kernels = [k for k in KERNEL_STYLES if any(p["kernel"] == k for p in points)]
    ppcs = sorted({p["ppc"] for p in points})
    if len(ppcs) > len(PPC_MARKERS):
        raise ValueError(f"{len(ppcs)} particles-per-cell values, {len(PPC_MARKERS)} markers")
    markers = dict(zip(ppcs, PPC_MARKERS))
    shades = {ppc: PPC_RAMP[round(i * (len(PPC_RAMP) - 1) / max(1, len(ppcs) - 1))]
              for i, ppc in enumerate(ppcs)}
    with matplotlib.rc_context({
        "font.size": TEXT_SIZES["default"], "axes.labelsize": TEXT_SIZES["axis_label"],
        "axes.titlesize": TEXT_SIZES["title"], "figure.titlesize": TEXT_SIZES["title"],
        "legend.fontsize": TEXT_SIZES["legend"],
        "legend.title_fontsize": TEXT_SIZES["legend"],
        "xtick.labelsize": TEXT_SIZES["tick"], "ytick.labelsize": TEXT_SIZES["tick"],
    }):
        # DRAM roofline with the device's peak and measured ceilings
        ceilings = (
            (roofs["ncu_dram_gbytes_per_second"], roofs["ncu_fp64_gflops_per_second"],
             "Peak (Nsight Compute)", dict(linestyle="--", linewidth=1.5)),
            (roofs["measured_bandwidth_gbytes_per_second"],
             roofs["measured_fp64_gflops_per_second"], "Measured", dict(linewidth=2)))
        low, high = limits([p["dram_flops_per_byte"] for p in points],
                           [fp64 / bandwidth for bandwidth, fp64, _, _ in ceilings])
        fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
        roof_lines = [roof(ax, bandwidth, fp64, low, high, color="0.35", **style,
                           label=f"{label}: {bandwidth / 1000:.2f} TB/s, "
                                 f"{fp64 / 1000:.2f} TFLOP/s")
                      for bandwidth, fp64, label, style in ceilings]
        for kernel in kernels:
            draw(ax, kernel, "dram_flops_per_byte", KERNEL_STYLES[kernel][0])
        ppc_legend(ax, "upper left")
        ax.legend(handles=roof_lines + [
            Line2D([], [], color=KERNEL_STYLES[k][0], linewidth=3,
                   label=KERNEL_STYLES[k][1]) for k in kernels], loc="lower right")
        ax.set(xlabel="FP64 FLOP / DRAM byte", ylabel="FP64 GFLOP/s",
               title=f"{caption}\n{WORK.replace(' (', chr(10) + '(')}")
        ax.grid(True, which="both", alpha=0.2)
        save(fig, "roofline")

        # hierarchical roofline, one panel per kernel, Nsight Compute peaks
        peak = roofs["ncu_fp64_gflops_per_second"]
        xs = [p[f"{level}_flops_per_byte"] for p in points for level in LEVEL_STYLES]
        ridges = [peak / roofs[f"ncu_{level}_gbytes_per_second"] for level in LEVEL_STYLES]
        low, high = limits(xs, ridges)
        fig, axes = plt.subplots(1, len(kernels), figsize=(8 * len(kernels), 8),
                                 constrained_layout=True, sharey=True, squeeze=False)
        for ax, kernel in zip(axes[0], kernels):
            for level, (color, name) in LEVEL_STYLES.items():
                bandwidth = roofs[f"ncu_{level}_gbytes_per_second"]
                roof(ax, bandwidth, peak, low, high, color=color, linestyle="--",
                     linewidth=1.5, label=f"{name} {bandwidth / 1000:.1f} TB/s")
                draw(ax, kernel, f"{level}_flops_per_byte", color)
            ax.set(xlabel="FP64 FLOP / byte", title=KERNEL_STYLES[kernel][1])
            ax.grid(True, which="both", alpha=0.2)
        axes[0][0].set_ylabel("FP64 GFLOP/s")
        ppc_legend(axes[0][0], "center left")
        axes[0][0].legend(loc="lower right", title=f"FP64 {peak / 1000:.2f} TFLOP/s")
        fig.suptitle(f"{caption}; Nsight Compute peaks\n{WORK}")
        save(fig, "roofline_hierarchical")

        # speed of light, one panel per kernel: each unit's share of its peak
        columns = [column for column, _, _ in SOL]
        width = 0.8 / len(ppcs)
        fig, axes = plt.subplots(1, len(kernels), figsize=(8 * len(kernels), 7),
                                 constrained_layout=True, sharey=True, squeeze=False)
        for ax, kernel in zip(axes[0], kernels):
            for ppc, values in means(points, kernel, columns):
                offset = (ppcs.index(ppc) - (len(ppcs) - 1) / 2) * width
                ax.bar([u + offset for u in range(len(SOL))], values, width,
                       color=shades[ppc], edgecolor="white", linewidth=1.5,
                       label=str(ppc))
            ax.set_xticks(range(len(SOL)), [label for _, label, _ in SOL],
                          rotation=45, ha="right")
            # the band above 100 % holds the legend
            ax.set(ylim=(0, 125), yticks=range(0, 101, 20), title=KERNEL_STYLES[kernel][1])
            ax.grid(True, axis="y", alpha=0.2)
            ax.set_axisbelow(True)
        axes[0][0].set_ylabel("% of peak")
        axes[0][0].legend(title="particles per cell", loc="upper center",
                          ncol=len(ppcs))
        fig.suptitle(f"{caption}; speed of light (Nsight Compute)\n"
                     "L1: of its active cycles; the others: of the elapsed cycles")
        save(fig, "speed_of_light")

    paper_roofline(points, roofs, kernels, markers, output / f"roofline_paper_{tag}")
    paper_speed_of_light(points, kernels, PAPER_SOL_ROWS, "speed_of_light_paper",
                         output, tag)
    paper_speed_of_light(points, kernels, PAPER_SOL_BREAKDOWN_ROWS,
                         "speed_of_light_paper_breakdown", output, tag, diagonal=True)


def paper_roofline(points, roofs, kernels, markers, stem):
    """The DRAM roofline for papers, in the style of the article_veil rooflines
    (article_v0/roofline/roofline_ncu_ops.py): no title; the FP64 roof of the
    Nsight Compute peaks and the kernels, the means of the repetitions joined
    over the particles per cell, in one legend; the particles per cell in a
    legend of their own."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    bandwidth = roofs["ncu_dram_gbytes_per_second"]
    fp64 = roofs["ncu_fp64_gflops_per_second"]
    ridge = fp64 / bandwidth
    xs = [p["dram_flops_per_byte"] for p in points]
    low, high = min(min(xs), ridge) / 3, max(max(xs), ridge) * 3
    with matplotlib.rc_context({
        "font.size": PAPER_TEXT_SIZES["default"],
        "axes.labelsize": PAPER_TEXT_SIZES["axis_label"],
        "legend.fontsize": PAPER_TEXT_SIZES["legend"],
        "legend.title_fontsize": PAPER_TEXT_SIZES["legend_title"],
        "xtick.labelsize": PAPER_TEXT_SIZES["tick"],
        "ytick.labelsize": PAPER_TEXT_SIZES["tick"],
    }):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, constrained_layout=True)
        colour, line_style = PAPER_ROOF_STYLE
        ax.loglog([low, ridge, high], [bandwidth * low, fp64, fp64], color=colour,
                  linestyle=line_style, linewidth=2,
                  label=f"FP64 roof: {fp64 / 1000:.2f} TFLOP/s")
        for kernel in kernels:
            colour, label = KERNEL_STYLES[kernel]
            data = means(points, kernel, ("dram_flops_per_byte", "gflops_per_second"))
            ax.plot([xy[0] for _, xy in data], [xy[1] for _, xy in data], color=colour,
                    label=label)
            for ppc, (x, y) in data:
                ax.plot(x, y, color=colour, marker=markers[ppc], markersize=9,
                        linestyle="none")
        # beyond four decades the 1-2-5 labels overlap at this text size
        subs = (1, 2, 5) if high / low <= 1e4 else (1, 3)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=subs, numticks=9))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="both", alpha=0.2)
        ax.set(xlabel="Arithmetic Intensity (FLOP/byte)", ylabel="Throughput (GFLOP/s)")
        ax.add_artist(ax.legend(loc="lower right"))
        ppc_legend = ax.legend(handles=[Line2D([], [], color="0.3", marker=marker,
                                               markersize=9, linestyle="none",
                                               label=str(ppc))
                                        for ppc, marker in markers.items()],
                               title="PPC", loc="upper left")
        # the axes go just high enough for the legend of the particles per cell
        # to clear the roofline below its corner (the layout moves a little as
        # the limits change, so twice)
        bottom = min(min(p["gflops_per_second"] for p in points), bandwidth * low) / 1.25
        top = fp64 * PAPER_HEADROOM
        ax.set_ylim(bottom, top)
        for _ in range(2):
            fig.canvas.draw()
            box = ppc_legend.get_window_extent().transformed(ax.transAxes.inverted())
            corner = ax.transData.inverted().transform(
                ax.transAxes.transform((box.x1, box.y0)))[0]
            clear = min(bandwidth * corner, fp64) * PAPER_HEADROOM
            top = max(top, bottom * (clear / bottom) ** (1 / box.y0))
            ax.set_ylim(bottom, top)
        for extension in ("png", "pdf"):
            fig.savefig(f"{stem}.{extension}", dpi=180)
        plt.close(fig)


def paper_speed_of_light(points, kernels, rows, name, output, tag, diagonal=False):
    """The speed of light for papers, on the canvas and in the text sizes of
    paper_roofline: vertical bars, one group per unit of rows (columns of
    points.csv) and one bar per kernel, its percent of peak (the mean of the
    repetitions) at PAPER_SOL_PPC particles per cell; the legend in a box
    above the bars; the group names diagonal if asked.  Written to
    name_ppc<N>_<tag>.png/pdf."""
    import matplotlib
    import matplotlib.pyplot as plt

    ppcs = {p["ppc"] for p in points}
    ppc = PAPER_SOL_PPC if PAPER_SOL_PPC in ppcs else max(ppcs)
    values = {}
    for kernel in kernels:
        at_ppc = dict(means(points, kernel, rows)).get(ppc)
        if at_ppc:
            values[kernel] = dict(zip(rows, at_ppc))
    labels = {**{column: label for column, label, _ in SOL}, "mio_pct": "MIO"}
    width = PAPER_SOL_GROUP / len(values)
    with matplotlib.rc_context({
        "font.size": PAPER_TEXT_SIZES["default"],
        "axes.labelsize": PAPER_TEXT_SIZES["axis_label"],
        "legend.fontsize": PAPER_TEXT_SIZES["legend"],
        "xtick.labelsize": PAPER_TEXT_SIZES["tick"],
        "ytick.labelsize": PAPER_TEXT_SIZES["tick"],
    }):
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE, constrained_layout=True)
        for k, (kernel, v) in enumerate(values.items()):
            ax.bar([g + (k - (len(values) - 1) / 2) * width
                    for g in range(len(rows))],
                   [v[column] for column in rows], width,
                   color=KERNEL_STYLES[kernel][0], edgecolor="white", linewidth=1.5,
                   label=KERNEL_STYLES[kernel][1])
        ax.set_xticks(range(len(rows)), [labels[c] for c in rows],
                      **(dict(rotation=45, ha="right", rotation_mode="anchor")
                         if diagonal else {}))
        ax.tick_params(axis="x", length=0)
        ax.set(xlim=(-0.5, len(rows) - 0.5), yticks=range(0, 101, 20),
               ylabel="Utilization (% of peak)")
        ax.grid(True, axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        legend = ax.legend(loc="upper center", ncol=len(values), handlelength=1.0,
                           handletextpad=0.3, columnspacing=0.8)
        # the axes go just high enough for the legend to clear the bars (the
        # layout moves a little as the limits change, so twice)
        tallest = max(max(v.values()) for v in values.values())
        top = 100.0
        ax.set_ylim(0, top)
        for _ in range(2):
            fig.canvas.draw()
            box = legend.get_window_extent().transformed(ax.transAxes.inverted())
            top = max(top, tallest * PAPER_HEADROOM / box.y0)
            ax.set_ylim(0, top)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{name}_ppc{ppc}_{tag}.{extension}",
                        dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="output folder of profile.sh")
    parser.add_argument("--output", type=Path, help="default: RUN/processed")
    parser.add_argument("--partial", action="store_true",
                        help="use the points collected so far of an unfinished run")
    parser.add_argument("--ppc", type=lambda s: {int(v) for v in s.split(",")},
                        metavar="N[,N...]", help="particles per cell to include")
    parser.add_argument("--no-plot", action="store_true", help="tables only")
    args = parser.parse_args()
    try:
        output = args.output or args.run / "processed"
        text, times, points = process(args.run, output, args.partial, args.ppc,
                                      not args.no_plot)
        print(text)
        print(f"{len(times)} timed points, {len(points)} profiled launches in {output}")
    except (ValueError, KeyError, OSError, ImportError, csv.Error) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
