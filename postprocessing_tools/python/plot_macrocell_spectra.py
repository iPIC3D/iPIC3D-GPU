#!/usr/bin/env python3
"""
Plot 2D (v_par, v_perp) velocity spectra produced by the iPIC3D-GPU
macrocell-spectra data-analysis module.

Input path must be the main ``macrocellSpectra/`` directory.

Output layout (one records file per MPI rank, read_analysis_records.py):
  <root>/subDomain<rank>_spectra.bin   -- one record per cycle and species:
                                          the non-empty bins of every
                                          macrocell
  <root>/subDomain<rank>_spectra.json  -- the rank, its macrocell tiling, the
                                          bins and the record layout
Macrocell m = (mz*My + my)*Mx + mx  (x fastest).

Examples
--------
# Plot every species, subdomain, cycle, and macrocell:
python plot_macrocell_spectra.py \\
    runs/.../macrocellSpectra --save out_dir/

# Plot a single macrocell (mx=2, my=1, mz=0) for species 0 at cycle 50:
python plot_macrocell_spectra.py \\
    runs/.../macrocellSpectra --species 0 --cycle 50 --macrocell 2 1 0

# Plot all cycles/macrocells for species 0 in subdomain 2:
python plot_macrocell_spectra.py \\
    runs/.../macrocellSpectra --species 0 --subdomain 2 --save out_dir/
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from read_analysis_records import macrocell_of, read_sparse  # noqa: E402


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """One 2D (v_par, v_perp) histogram of a macrocell at one cycle."""
    meta: dict                         # subDomain<rank>_spectra.json
    hist: np.ndarray                   # shape (n_vperp, n_vpar)
    species: int
    macrocell: tuple[int, int, int]    # (mx, my, mz) 0-based indices
    cycle: int                         # simulation cycle
    vpar_range: tuple[float, float]
    vperp_range: tuple[float, float]

    @property
    def vpar_edges(self) -> np.ndarray:
        return np.linspace(*self.vpar_range, self.hist.shape[1] + 1)

    @property
    def vperp_edges(self) -> np.ndarray:
        return np.linspace(*self.vperp_range, self.hist.shape[0] + 1)

    @property
    def label(self) -> str:
        m = self.meta
        mx, my, mz = self.macrocell
        Mx, My, _ = m["macrocells_per_axis"]
        mc = macrocell_of(m, (mz * My + my) * Mx + mx)
        return (
            f"rank {m.get('subdomain_rank', '?')}  "
            f"mc[{mx},{my},{mz}]  "
            f"size {'x'.join(map(str, mc.cells))}  "
            f"globalOff [{','.join(map(str, mc.first_cell))}]  "
            f"cycle {self.cycle}  species {self.species}"
        )

    @property
    def save_stem(self) -> str:
        """A path-safe stem for use when writing PNG files."""
        mx, my, mz = self.macrocell
        return (
            f"subDomain_{self.meta.get('subdomain_rank', 0)}/"
            f"species_{self.species}/cycle_{self.cycle:06d}/mc_{mx}_{my}_{mz}"
        )


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

def rank_of(path: Path) -> int | None:
    """R for a subDomain<R>_spectra.bin path, or None if it does not match."""
    match = re.fullmatch(r"subDomain(\d+)_spectra\.bin", path.name)
    return int(match.group(1)) if match else None


def find_spectra_files(root: Path, subdomain: int | None = None) -> list[Path]:
    """Return the matching ``subDomain<R>_spectra.bin`` files under *root*."""
    root = Path(root)
    if root.is_file():
        raise ValueError(
            f"Expected the macrocellSpectra root directory, got file: {root}"
        )
    if not root.is_dir():
        raise FileNotFoundError(root)
    files = [p for p in root.glob("subDomain*_spectra.bin")
             if rank_of(p) is not None
             and (subdomain is None or rank_of(p) == subdomain)]
    return sorted(files, key=rank_of)


def iter_spectra(
    files: list[Path],
    species: int | None = None,
    cycle: int | None = None,
    macrocell: tuple[int, int, int] | None = None,
) -> Iterator[Spectrum]:
    """The spectra of the files, of a species, a cycle and a macrocell if
    given."""
    for f in files:
        meta = json.loads(f.with_suffix(".json").read_text())
        Mx, My, Mz = meta["macrocells_per_axis"]
        if macrocell is not None and not all(
                0 <= i < n for i, n in zip(macrocell, (Mx, My, Mz))):
            print(f"[warn] macrocell {macrocell} out of range for {f.name}",
                  file=sys.stderr)
            continue
        cells = [macrocell] if macrocell is not None else [
            (mx, my, mz) for mz in range(Mz) for my in range(My)
            for mx in range(Mx)]
        for record in read_sparse(f, cycle, species):
            for mx, my, mz in cells:
                m = (mz * My + my) * Mx + mx
                lo, hi = record.ranges[m, :2], record.ranges[m, 2:]
                yield Spectrum(meta, record.dense(m), record.species,
                               (mx, my, mz), record.cycle,
                               (lo[0], hi[0]), (lo[1], hi[1]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectrum(
    spec: Spectrum,
    *,
    out_file: Path | None = None,
    show: bool = True,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    H      = spec.hist
    vpar_e  = spec.vpar_edges
    vperp_e = spec.vperp_edges

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    pos = H[H > 0]
    norm = (
        LogNorm(vmin=float(pos.min()), vmax=float(H.max()))
        if (pos.size > 0 and H.max() > 0)
        else None
    )

    pcm = ax.pcolormesh(vpar_e, vperp_e, H, cmap="viridis", norm=norm,
                        shading="auto")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("|q| weight_scale (log)" if norm is not None
                 else "|q| weight_scale")

    ax.set_xlabel(r"$v_\parallel$")
    ax.set_ylabel(r"$v_\perp$")
    ax.set_aspect("auto")
    ax.set_title(spec.label, fontsize=9)

    fig.tight_layout()
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150)
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
    p.add_argument(
        "path", type=Path,
        help="macrocellSpectra/ root directory",
    )
    p.add_argument("--species",   type=int, default=None,
                   help="species index to plot (default: all species)")
    p.add_argument("--cycle",     type=int, default=None,
                   help="simulation cycle to plot (default: all recorded cycles)")
    p.add_argument("--subdomain", type=int, default=None,
                   help="MPI rank/subdomain to plot (default: all subdomains)")
    p.add_argument("--macrocell", type=int, nargs=3, default=None,
                   metavar=("MX", "MY", "MZ"),
                   help="macrocell to plot as MX MY MZ (default: all macrocells)")
    p.add_argument("--save",      type=Path, default=None,
                   help="output directory for PNG files (suppresses display)")
    args = p.parse_args(argv)

    # --- Discover the records files ---
    try:
        files = find_spectra_files(args.path, subdomain=args.subdomain)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    if not files:
        print("No subDomain<R>_spectra.bin files matched.", file=sys.stderr)
        return 1

    mc = tuple(args.macrocell) if args.macrocell is not None else None
    spectra = iter_spectra(files, species=args.species, cycle=args.cycle,
                           macrocell=mc)
    plotted = 0
    for spec in spectra:
        out = (args.save / (spec.save_stem + ".png")) if args.save else None
        plot_spectrum(spec, out_file=out, show=args.save is None)
        plotted += 1

    if plotted == 0:
        print("No (cycle, macrocell) combinations matched.", file=sys.stderr)
        return 1
    print(f"Plotted {plotted} spectrum record(s).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
