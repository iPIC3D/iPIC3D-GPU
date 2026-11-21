#!/usr/bin/env python3
"""
Plot the 3D (u, v, w) velocity histograms of the iPIC3D-GPU data analysis,
of each rank or each macrocell, with their Gaussian-mixture (GMM) fits; or
the fits alone (--gmm-only), which need the velocityGMM/ files only.

Input path: the DataAnalysisOutputDir of a run (holding velocityHistogram/
and, with VelocityGMM, velocityGMM/), or its velocityHistogram/ folder
(velocityGMM/ with --gmm-only).

One figure per rank, cycle, species and region (the rank, or a macrocell)
with particles (with a fit, --gmm-only):
  top     the u-v, u-w and v-w projections of the histogram (its weight
          summed over the third axis, log colour scale), with the 1 and 2
          sigma ellipses of every component of the fit projected on them;
          --gmm-only: the projected density of the mixture (log colour
          scale, down to 1e-4 of its peak) under the ellipses
  bottom  the u, v and w marginals of the histogram, with the weight that
          the mixture and each component put in each of its bins (their
          marginal density integrated over the bin, the histogram's weight
          times their probability); --gmm-only: the marginal densities of the
          mixture and of each component, on the range of the components to 4
          sigma
Pruned components (weight 1e-15) are left out; a region without a fit at
the cycle shows the histogram alone.

Examples
--------
# Every rank, cycle, species and region with particles, into PNG files:
python plot_velocity_histograms.py runs/.../ --save out_dir/

# Macrocell (mx=2, my=1, mz=0) of rank 3, species 0, cycle 500, on screen:
python plot_velocity_histograms.py runs/.../ --subdomain 3 --species 0 \\
    --cycle 500 --macrocell 2 1 0

# The histograms alone, without the fits:
python plot_velocity_histograms.py runs/.../velocityHistogram --no-gmm \\
    --save out_dir/

# The fits alone (PNG files named *_gmm.png):
python plot_velocity_histograms.py runs/.../ --gmm-only --save out_dir/
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_analysis_records import (GMMRecord, macrocell_of,  # noqa: E402
                                   read_gmm, read_sparse)

AXES = ("u", "v", "w")
PLANES = ((0, 1), (0, 2), (1, 2))  # the projections: (horizontal, vertical)
PRUNED_WEIGHT = 1e-14  # a component of weight at most this was pruned
SIGMAS = (1, 2)        # the ellipses of a component
# --gmm-only: the range (the components to this many sigma), the points of
# a density along an axis, and the decades of its colour scale
FIT_RANGE_SIGMAS = 4
DENSITY_POINTS = 200
DENSITY_DECADES = 4


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Fit:
    """The GMM fit of a region at a cycle: its active components."""
    weights: np.ndarray      # (k,)
    means: np.ndarray        # (k, 3)
    covariances: np.ndarray  # (k, 3, 3)
    components: np.ndarray   # (k,): their index in the fit
    steps: int
    log_likelihood: float

    def range(self) -> tuple[np.ndarray, np.ndarray]:
        """Along each axis, the components to FIT_RANGE_SIGMAS sigma."""
        sigma = np.sqrt(np.diagonal(self.covariances, axis1=1, axis2=2))
        return ((self.means - FIT_RANGE_SIGMAS * sigma).min(0),
                (self.means + FIT_RANGE_SIGMAS * sigma).max(0))


@dataclass
class Region:
    """A region of a rank (the rank, or a macrocell) at one cycle and
    species: its histogram, its fit, or both."""
    meta: dict                 # the .json of its records file
    region: int
    cycle: int
    species: int
    fit: Fit | None
    weights: np.ndarray | None  # (bins_u, bins_v, bins_w); None: fit alone
    low: np.ndarray            # (3,): the range of each axis, of the
    high: np.ndarray           # histogram or (fit alone) of Fit.range

    def edges(self, d: int) -> np.ndarray:
        """The bin edges of the histogram along axis d."""
        return np.linspace(self.low[d], self.high[d],
                           self.weights.shape[d] + 1)

    @property
    def label(self) -> str:
        m = self.meta
        mc = macrocell_of(m, self.region)
        where = "whole rank" if mc is None else (
            f"mc[{','.join(map(str, mc.index))}]  "
            f"size {'x'.join(map(str, mc.cells))}  "
            f"globalOff [{','.join(map(str, mc.first_cell))}]")
        text = (f"rank {m.get('subdomain_rank', '?')}  {where}  "
                f"cycle {self.cycle}  species {self.species}")
        if self.fit is not None:
            k = len(self.fit.weights)
            text += (f"\nGMM: {k} component{'s' if k > 1 else ''}, "
                     f"{self.fit.steps} EM steps, "
                     f"log L {self.fit.log_likelihood:.6g}")
        return text

    @property
    def save_stem(self) -> str:
        """A path-safe stem for the PNG file (_gmm: the fit alone)."""
        mc = macrocell_of(self.meta, self.region)
        where = "rank" if mc is None else "mc_" + "_".join(map(str, mc.index))
        return (f"subDomain_{self.meta.get('subdomain_rank', 0)}/"
                f"species_{self.species}/cycle_{self.cycle:06d}/{where}"
                f"{'_gmm' if self.weights is None else ''}")


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

def rank_of(path: Path, name: str) -> int | None:
    """R for a subDomain<R>_<name>.bin path, None if it does not match (a
    file kept aside, *_before_cycle<N>.bin, does not)."""
    match = re.fullmatch(rf"subDomain(\d+)_{name}\.bin", path.name)
    return int(match.group(1)) if match else None


def find_files(root: Path, folder: str, name: str,
               subdomain: int | None = None) -> list[Path]:
    """The subDomain<R>_<name>.bin files of the run folder root (in its
    subfolder folder) or of the folder root."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {root}")
    where = root / folder if (root / folder).is_dir() else root
    files = [p for p in where.glob(f"subDomain*_{name}.bin")
             if rank_of(p, name) is not None
             and (subdomain is None or rank_of(p, name) == subdomain)]
    return sorted(files, key=lambda p: rank_of(p, name))


def gmm_records(histograms: Path) -> dict[int, GMMRecord]:
    """The fits of the rank of a histograms file, by cycle: its
    subDomain<R>_gmm.bin in the velocityGMM/ folder next to it, if any."""
    path = histograms.parent.parent / "velocityGMM" / histograms.name.replace(
        "_histograms.bin", "_gmm.bin")
    return {g.cycle: g for g in read_gmm(path)} if path.exists() else {}


def fit_of(record: GMMRecord | None, species: int, region: int) -> Fit | None:
    """The fit of a species and region in a GMM record, None without one."""
    if record is None or not record.fitted[species, region]:
        return None
    w = record.weights[species, region]
    active = np.flatnonzero(w > PRUNED_WEIGHT)
    return Fit(w[active], record.means[species, region][active],
               record.covariances[species, region][active], active,
               int(record.steps[species, region]),
               float(record.log_likelihood[species, region]))


def selected_regions(meta: dict, macrocell: tuple[int, int, int] | None,
                     name: str) -> list[int] | None:
    """The regions of a records file to plot: None for all, [] (with a
    warning) when it has no such macrocell."""
    if macrocell is None:
        return None
    if meta["regions"] != "macrocells":
        print(f"[warn] {name} has a region per rank, no macrocells",
              file=sys.stderr)
        return []
    Mx, My, Mz = meta["macrocells_per_axis"]
    if not all(0 <= i < n for i, n in zip(macrocell, (Mx, My, Mz))):
        print(f"[warn] macrocell {macrocell} out of range for {name}",
              file=sys.stderr)
        return []
    mx, my, mz = macrocell
    return [(mz * My + my) * Mx + mx]


def iter_histograms(
    files: list[Path],
    species: int | None = None,
    cycle: int | None = None,
    macrocell: tuple[int, int, int] | None = None,
    with_gmm: bool = True,
) -> Iterator[Region]:
    """The histograms with particles of the files (subDomain<R>_
    histograms.bin), of a species, a cycle and a macrocell if given, with
    their fits (with_gmm)."""
    for f in files:
        meta = json.loads(f.with_suffix(".json").read_text())
        regions = selected_regions(meta, macrocell, f.name)
        if regions == []:
            continue
        fits = gmm_records(f) if with_gmm else {}
        for record in read_sparse(f, cycle, species):
            gmm = fits.get(record.cycle)
            if gmm is not None and gmm.weights.shape[1] != len(record.ranges):
                print(f"[warn] the fits of cycle {record.cycle} of {f.name} "
                      "have other regions: not shown", file=sys.stderr)
                gmm = None
            for r in regions if regions is not None else range(
                    len(record.ranges)):
                if record.offsets[r + 1] == record.offsets[r]:
                    continue  # no particle in its range
                # dense is (w, v, u): to (u, v, w)
                yield Region(meta, r, record.cycle, record.species,
                             fit_of(gmm, record.species, r),
                             record.dense(r).astype(np.float64).transpose(),
                             record.ranges[r, :3], record.ranges[r, 3:])


def iter_fits(
    files: list[Path],
    species: int | None = None,
    cycle: int | None = None,
    macrocell: tuple[int, int, int] | None = None,
) -> Iterator[Region]:
    """The fits of the files (subDomain<R>_gmm.bin), of a species, a cycle
    and a macrocell if given, without their histograms."""
    for f in files:
        meta = json.loads(f.with_suffix(".json").read_text())
        regions = selected_regions(meta, macrocell, f.name)
        if regions == []:
            continue
        for record in read_gmm(f, cycle):
            S, R, _ = record.weights.shape
            for s in range(S) if species is None else [species]:
                if not 0 <= s < S:
                    continue
                for r in regions if regions is not None else range(R):
                    fit = fit_of(record, s, r)
                    if fit is not None:
                        yield Region(meta, r, record.cycle, s, fit, None,
                                     *fit.range())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_erf = np.vectorize(math.erf)


def bin_probability(edges: np.ndarray, mean: float,
                    variance: float) -> np.ndarray:
    """The probability of each bin of edges under a 1D Gaussian."""
    cdf = 0.5 * (1 + _erf((edges - mean) / math.sqrt(2 * variance)))
    return np.diff(cdf)


def gaussian(x: np.ndarray, mean: float, variance: float) -> np.ndarray:
    """The density of a 1D Gaussian."""
    return np.exp(-0.5 * (x - mean) ** 2 / variance) / math.sqrt(
        2 * math.pi * variance)


def projected_density(fit: Fit, a: int, b: int, x: np.ndarray,
                      y: np.ndarray) -> np.ndarray:
    """The density of the mixture projected on the plane of axes a, b, on
    the grid of x and y: (len(y), len(x))."""
    X, Y = np.meshgrid(x, y)
    density = np.zeros_like(X)
    for w, mean, cov in zip(fit.weights, fit.means, fit.covariances):
        c = cov[np.ix_([a, b], [a, b])]
        inverse = np.linalg.inv(c)
        dx, dy = X - mean[a], Y - mean[b]
        q = (inverse[0, 0] * dx * dx + 2 * inverse[0, 1] * dx * dy +
             inverse[1, 1] * dy * dy)
        density += w * np.exp(-0.5 * q) / (
            2 * math.pi * math.sqrt(np.linalg.det(c)))
    return density


def draw_ellipses(ax, fit: Fit, a: int, b: int, colors) -> None:
    """The SIGMAS ellipses and the mean of every component on the plane of
    axes a, b."""
    from matplotlib.patches import Ellipse

    for j, k in enumerate(fit.components):
        mean = fit.means[j, [a, b]]
        values, vectors = np.linalg.eigh(
            fit.covariances[j][np.ix_([a, b], [a, b])])
        angle = np.degrees(np.arctan2(vectors[1, 1], vectors[0, 1]))
        for n in SIGMAS:
            ax.add_patch(Ellipse(
                mean, 2 * n * np.sqrt(max(values[1], 0)),
                2 * n * np.sqrt(max(values[0], 0)), angle=angle, fill=False,
                color=colors(k % 10), lw=1.2 / n))
        ax.plot(*mean, "+", color=colors(k % 10), ms=8)


def plot_region(
    region: Region,
    *,
    out_file: Path | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.5))
    W, fit = region.weights, region.fit
    lo, hi = region.low, region.high

    # the projections, with the ellipses of the components
    for ax, (a, b) in zip(axes[0], PLANES):
        if W is not None:
            P = W.sum(axis=3 - a - b)  # (bins_a, bins_b)
            positive = P[P > 0]
            mesh = ax.pcolormesh(
                region.edges(a), region.edges(b),
                np.ma.masked_less_equal(P.T, 0), cmap="viridis",
                norm=LogNorm(positive.min(), positive.max())
                if positive.size else None, shading="flat")
            fig.colorbar(mesh, ax=ax, label="|q| weight_scale")
        else:
            x = np.linspace(lo[a], hi[a], DENSITY_POINTS)
            y = np.linspace(lo[b], hi[b], DENSITY_POINTS)
            D = projected_density(fit, a, b, x, y)
            floor = D.max() * 10.0 ** -DENSITY_DECADES
            mesh = ax.pcolormesh(x, y, np.ma.masked_less(D, floor),
                                 cmap="viridis", norm=LogNorm(floor, D.max()),
                                 shading="nearest")
            fig.colorbar(mesh, ax=ax, label="probability density")
        if fit is not None:
            draw_ellipses(ax, fit, a, b, colors)
        ax.set_xlim(lo[a], hi[a])
        ax.set_ylim(lo[b], hi[b])
        ax.set_xlabel(AXES[a])
        ax.set_ylabel(AXES[b])

    # the marginals: of the histogram, with the weights of the mixture and
    # the components in its bins; of the fit alone, their densities
    for d, ax in enumerate(axes[1]):
        if W is not None:
            edges = region.edges(d)
            total = W.sum()
            ax.stairs(W.sum(axis=tuple(x for x in range(3) if x != d)), edges,
                      color="0.3", fill=True, alpha=0.35, label="histogram")
            if fit is not None:
                mixture = np.zeros(len(edges) - 1)
                for j, k in enumerate(fit.components):
                    part = total * fit.weights[j] * bin_probability(
                        edges, fit.means[j, d], fit.covariances[j, d, d])
                    mixture += part
                    ax.stairs(part, edges, color=colors(k % 10), ls="--",
                              lw=0.9, label=f"k={k}: {fit.weights[j]:.3f}")
                ax.stairs(mixture, edges, color="k", lw=1.4, label="mixture")
            ax.set_ylabel("weight per bin")
        else:
            x = np.linspace(lo[d], hi[d], DENSITY_POINTS)
            mixture = np.zeros_like(x)
            for j, k in enumerate(fit.components):
                part = fit.weights[j] * gaussian(x, fit.means[j, d],
                                                 fit.covariances[j, d, d])
                mixture += part
                ax.plot(x, part, "--", color=colors(k % 10), lw=0.9,
                        label=f"k={k}: {fit.weights[j]:.3f}")
            ax.plot(x, mixture, color="k", lw=1.4, label="mixture")
            ax.set_ylim(bottom=0)
            ax.set_ylabel("probability density")
        ax.set_xlim(lo[d], hi[d])
        ax.set_xlabel(AXES[d])
    if fit is not None:
        axes[1, 2].legend(fontsize=7, loc="upper right",
                          title="component: weight", title_fontsize=7)
    for ax in axes.flat:  # narrow ranges: few ticks, readable
        ax.locator_params(axis="both", nbins=5)

    fig.suptitle(region.label, fontsize=10)
    fig.tight_layout()
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=120)
        print(f"  -> {out_file}")
    if show and out_file is None:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", type=Path,
                   help="the run's DataAnalysisOutputDir, or its "
                        "velocityHistogram/ folder (velocityGMM/ with "
                        "--gmm-only)")
    p.add_argument("--species", type=int, default=None,
                   help="species index to plot (default: all species)")
    p.add_argument("--cycle", type=int, default=None,
                   help="simulation cycle to plot (default: all recorded "
                        "cycles)")
    p.add_argument("--subdomain", type=int, default=None,
                   help="MPI rank/subdomain to plot (default: all)")
    p.add_argument("--macrocell", type=int, nargs=3, default=None,
                   metavar=("MX", "MY", "MZ"),
                   help="macrocell to plot as MX MY MZ (default: every "
                        "region)")
    which = p.add_mutually_exclusive_group()
    which.add_argument("--no-gmm", action="store_true",
                       help="the histograms alone, without the fits")
    which.add_argument("--gmm-only", action="store_true",
                       help="the fits alone, without the histograms (the "
                            "velocityGMM/ files only)")
    p.add_argument("--save", type=Path, default=None,
                   help="output directory for PNG files (suppresses display)")
    args = p.parse_args(argv)

    folder, name = ("velocityGMM", "gmm") if args.gmm_only else (
        "velocityHistogram", "histograms")
    try:
        files = find_files(args.path, folder, name, subdomain=args.subdomain)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not files:
        print(f"No subDomain<R>_{name}.bin files matched.", file=sys.stderr)
        return 1

    mc = tuple(args.macrocell) if args.macrocell is not None else None
    regions = iter_fits(files, args.species, args.cycle, mc) \
        if args.gmm_only else iter_histograms(
            files, args.species, args.cycle, mc, with_gmm=not args.no_gmm)
    plotted = 0
    for region in regions:
        out = (args.save / (region.save_stem + ".png")) if args.save else None
        plot_region(region, out_file=out, show=args.save is None)
        plotted += 1

    if plotted == 0:
        print("No (cycle, region) " + ("with a fit" if args.gmm_only else
                                       "with particles") + " matched.",
              file=sys.stderr)
        return 1
    print(f"Plotted {plotted} " + ("fit(s)." if args.gmm_only else
                                   "histogram(s)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
