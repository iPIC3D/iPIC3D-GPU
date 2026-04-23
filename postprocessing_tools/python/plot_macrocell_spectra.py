#!/usr/bin/env python3
"""
Plot 2D (v_par, v_perp) velocity spectra produced by the iPIC3D-GPU
macrocell-spectra data-analysis module.

Each macrocell is stored as:
  - mc_<mx>_<my>_<mz>.json  : metadata sidecar
  - mc_<mx>_<my>_<mz>.bin   : raw float32, row-major, vperp outer / vpar inner

Directory layout produced by the simulation:
  <root>/subDomain_<rank>/species_<s>/cycle_<NNNNNN>/mc_<mx>_<my>_<mz>.{bin,json}

Examples
--------
# Plot a single macrocell:
python plot_macrocell_spectra.py \
    runs/.../macrocellSpectra/subDomain_3/species_1/cycle_000050/mc_0_0_1.json

# Plot all macrocells of one cycle/species/subdomain (one figure per cell):
python plot_macrocell_spectra.py \
    runs/.../macrocellSpectra/subDomain_3/species_1/cycle_000050

# Sum every macrocell of every subdomain for species 1 at cycle 50:
python plot_macrocell_spectra.py \
    runs/.../macrocellSpectra --species 1 --cycle 50 --aggregate

# Save figures to PNG instead of showing them:
python plot_macrocell_spectra.py <path> --save out_dir/
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """One 2D (v_par, v_perp) histogram and its metadata."""
    meta: dict
    hist: np.ndarray  # shape = (n_vperp, n_vpar), float32 counts
    json_path: Path

    @property
    def n_vperp(self) -> int:
        return int(self.meta["bins"][0])

    @property
    def n_vpar(self) -> int:
        return int(self.meta["bins"][1])

    @property
    def vpar_edges(self) -> np.ndarray:
        v0, v1 = self.meta["ranges"]["vpar"]
        return np.linspace(v0, v1, self.n_vpar + 1)

    @property
    def vperp_edges(self) -> np.ndarray:
        v0, v1 = self.meta["ranges"]["vperp"]
        return np.linspace(v0, v1, self.n_vperp + 1)

    @property
    def label(self) -> str:
        m = self.meta
        idx = m.get("macrocell_index_in_subdomain", [0, 0, 0])
        gOff = m.get("macrocell_offset_in_global_cells", [0, 0, 0])
        sz = m.get("macrocell_size_cells", [0, 0, 0])
        return (
            f"rank {m.get('mpi_rank', '?')}  "
            f"mc[{idx[0]},{idx[1]},{idx[2]}]  "
            f"size {sz[0]}x{sz[1]}x{sz[2]}  "
            f"globalOff [{gOff[0]},{gOff[1]},{gOff[2]}]  "
            f"cycle {m.get('cycle', '?')}  species {m.get('species', '?')}"
        )


def load_spectrum(json_path: Path) -> Spectrum:
    """Load one macrocell spectrum from its JSON sidecar."""
    json_path = Path(json_path)
    with json_path.open() as f:
        meta = json.load(f)

    bin_name = meta.get("binary_file", json_path.with_suffix(".bin").name)
    bin_path = json_path.parent / bin_name
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    n_vperp, n_vpar = int(meta["bins"][0]), int(meta["bins"][1])
    expected = n_vperp * n_vpar
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size != expected:
        raise ValueError(
            f"{bin_path}: expected {expected} float32, got {raw.size}"
        )
    # Row-major, vperp outer, vpar inner -> shape (n_vperp, n_vpar)
    hist = raw.reshape(n_vperp, n_vpar)
    return Spectrum(meta=meta, hist=hist, json_path=json_path)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_jsons(
    path: Path,
    species: int | None = None,
    cycle: int | None = None,
    subdomain: int | None = None,
) -> list[Path]:
    """Return a sorted list of mc_*.json files under `path`.

    `path` may point to:
      - a single mc_*.json file,
      - a cycle dir,
      - a species dir,
      - a subdomain dir,
      - the macrocellSpectra root dir.

    Optional `species`, `cycle`, `subdomain` filters are applied on the
    parent directory names ("species_<s>", "cycle_<NNNNNN>",
    "subDomain_<r>").
    """
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)

    jsons = sorted(path.rglob("mc_*.json"))

    def keep(p: Path) -> bool:
        parts = set(p.parts)
        if species is not None and f"species_{species}" not in parts:
            return False
        if cycle is not None and f"cycle_{cycle:06d}" not in parts:
            return False
        if subdomain is not None and f"subDomain_{subdomain}" not in parts:
            return False
        return True

    return [p for p in jsons if keep(p)]


def iter_spectra(paths: Iterable[Path]) -> Iterator[Spectrum]:
    for p in paths:
        try:
            yield load_spectrum(p)
        except Exception as e:  # pragma: no cover
            print(f"[warn] skipping {p}: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_spectrum(
    spec: Spectrum,
    *,
    log: bool = True,
    cmap: str = "viridis",
    title: str | None = None,
    out_file: Path | None = None,
    show: bool = True,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    H = spec.hist
    vpar_e = spec.vpar_edges
    vperp_e = spec.vperp_edges

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    norm = LogNorm(vmin=max(H[H > 0].min(), 1.0), vmax=H.max()) if (log and H.max() > 0) else None

    pcm = ax.pcolormesh(vpar_e, vperp_e, H, cmap=cmap, norm=norm, shading="auto")
    cb = fig.colorbar(pcm, ax=ax)
    cb.set_label("counts" + (" (log)" if log else ""))

    ax.set_xlabel(r"$v_\parallel$")
    ax.set_ylabel(r"$v_\perp$")
    ax.set_aspect("auto")
    ax.set_title(title or spec.label, fontsize=9)

    fig.tight_layout()
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=150)
        print(f"  -> {out_file}")
    if show and out_file is None:
        plt.show()
    plt.close(fig)


def aggregate(spectra: list[Spectrum]) -> Spectrum:
    """Sum spectra that share identical bin geometry and ranges."""
    if not spectra:
        raise ValueError("nothing to aggregate")
    ref = spectra[0]
    for s in spectra[1:]:
        if s.hist.shape != ref.hist.shape:
            raise ValueError("aggregation requires identical bin counts")
        if s.meta["ranges"] != ref.meta["ranges"]:
            raise ValueError("aggregation requires identical velocity ranges")
    H = np.sum([s.hist for s in spectra], axis=0).astype(np.float32)
    meta = dict(ref.meta)
    meta["aggregated_count"] = len(spectra)
    meta["aggregated_label"] = (
        f"sum of {len(spectra)} macrocells "
        f"(species {meta.get('species','?')}, cycle {meta.get('cycle','?')})"
    )
    return Spectrum(meta=meta, hist=H, json_path=ref.json_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("path", type=Path,
                   help="JSON file, cycle dir, species dir, subdomain dir, "
                        "or macrocellSpectra root dir.")
    p.add_argument("--species", type=int, default=None,
                   help="filter by species index")
    p.add_argument("--cycle", type=int, default=None,
                   help="filter by cycle number")
    p.add_argument("--subdomain", type=int, default=None,
                   help="filter by MPI rank")
    p.add_argument("--aggregate", action="store_true",
                   help="sum all matching macrocells into a single spectrum")
    p.add_argument("--linear", action="store_true",
                   help="use linear color scale (default: log)")
    p.add_argument("--cmap", default="viridis", help="matplotlib colormap")
    p.add_argument("--save", type=Path, default=None,
                   help="output directory for PNG files (no on-screen display)")
    p.add_argument("--max", type=int, default=None,
                   help="limit the number of plots (useful for previews)")
    args = p.parse_args(argv)

    jsons = discover_jsons(args.path,
                           species=args.species,
                           cycle=args.cycle,
                           subdomain=args.subdomain)
    if not jsons:
        print("No mc_*.json files matched.", file=sys.stderr)
        return 1
    print(f"Found {len(jsons)} macrocell file(s).")

    spectra = list(iter_spectra(jsons))

    if args.aggregate:
        spec = aggregate(spectra)
        out = (args.save / "aggregate.png") if args.save else None
        title = spec.meta.get("aggregated_label")
        plot_spectrum(spec, log=not args.linear, cmap=args.cmap,
                      title=title, out_file=out, show=args.save is None)
        return 0

    if args.max is not None:
        spectra = spectra[:args.max]

    for spec in spectra:
        out = None
        if args.save:
            rel = spec.json_path.with_suffix(".png").name
            # mirror subdomain/species/cycle structure under --save
            sub = []
            for tag in ("subDomain_", "species_", "cycle_"):
                for part in spec.json_path.parts:
                    if part.startswith(tag):
                        sub.append(part)
                        break
            out = args.save.joinpath(*sub, rel)
        plot_spectrum(spec, log=not args.linear, cmap=args.cmap,
                      out_file=out, show=args.save is None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
