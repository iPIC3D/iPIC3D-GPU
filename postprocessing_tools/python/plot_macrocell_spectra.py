#!/usr/bin/env python3
"""
Plot 2D (v_par, v_perp) velocity spectra produced by the iPIC3D-GPU
macrocell-spectra data-analysis module.

Input path must be the main ``macrocellSpectra/`` directory.

Output layout (one binary + one JSON per species per MPI rank):
  <root>/subDomain_<rank>/species_<S>.bin   -- flat binary, appended per cycle
  <root>/subDomain_<rank>/species_<S>.json  -- fully self-contained metadata

Binary record layout:
  Record K covers cycle  records[K].
  Macrocell m = (mz*My + my)*Mx + mx  (x fastest).
  Byte offset of macrocell m at record K:
    K * record_size_bytes + m * bytes_per_macrocell

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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

FLOAT_DTYPE = np.dtype("<f4")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Spectrum:
    """One 2D (v_par, v_perp) histogram extracted from a species binary."""
    meta: dict                       # full species_S.json contents
    hist: np.ndarray                 # shape (n_vperp, n_vpar), float32
    json_path: Path                  # path to species_S.json
    macrocell: tuple[int, int, int]  # (mx, my, mz) 0-based macrocell indices
    cycle: int                       # simulation cycle
    record_index: int                # K: position in meta["records"]

    @property
    def n_vperp(self) -> int:
        return int(self.meta["bins"]["vperp"])

    @property
    def n_vpar(self) -> int:
        return int(self.meta["bins"]["vpar"])

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
        mx, my, mz = self.macrocell
        tX = m["axis_tiling"]["x"][mx]   # [start, size]
        tY = m["axis_tiling"]["y"][my]
        tZ = m["axis_tiling"]["z"][mz]
        gOff = m["subdomain_global_offset_cells"]
        gx = gOff[0] + tX[0]
        gy = gOff[1] + tY[0]
        gz = gOff[2] + tZ[0]
        return (
            f"rank {m.get('subdomain_rank', '?')}  "
            f"mc[{mx},{my},{mz}]  "
            f"size {tX[1]}x{tY[1]}x{tZ[1]}  "
            f"globalOff [{gx},{gy},{gz}]  "
            f"cycle {self.cycle}  species {m.get('species', '?')}"
        )

    @property
    def save_stem(self) -> str:
        """A path-safe stem for use when writing PNG files."""
        m = self.meta
        mx, my, mz = self.macrocell
        rank = m.get("subdomain_rank", 0)
        s    = m.get("species", 0)
        return (
            f"subDomain_{rank}/species_{s}/"
            f"cycle_{self.cycle:06d}/"
            f"mc_{mx}_{my}_{mz}"
        )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_spectrum(json_path: Path,
                  mx: int, my: int, mz: int,
                  cycle: int) -> Spectrum:
    """Load one macrocell histogram for a given cycle from species_S.json/bin.

    Parameters
    ----------
    json_path : Path
        Path to a ``species_S.json`` file.
    mx, my, mz : int
        0-based macrocell indices along x, y, z.
    cycle : int
        Simulation cycle to retrieve; must be listed in ``meta["records"]``.
    """
    json_path = Path(json_path)
    with json_path.open() as f:
        meta = json.load(f)

    records: list[int] = meta["records"]
    if cycle not in records:
        raise ValueError(
            f"Cycle {cycle} not found in {json_path.name}. "
            f"Available: {records}"
        )
    K = records.index(cycle)

    Mx, My, Mz = meta["macrocells_per_axis"]
    if not (0 <= mx < Mx and 0 <= my < My and 0 <= mz < Mz):
        raise ValueError(
            f"Macrocell ({mx},{my},{mz}) out of range "
            f"(Mx={Mx}, My={My}, Mz={Mz})"
        )

    m = (mz * My + my) * Mx + mx
    byte_offset = K * meta["record_size_bytes"] + m * meta["bytes_per_macrocell"]
    n_vperp = meta["bins"]["vperp"]
    n_vpar  = meta["bins"]["vpar"]
    Nb = n_vperp * n_vpar
    expected_bytes_per_macrocell = Nb * FLOAT_DTYPE.itemsize
    if meta["bytes_per_macrocell"] != expected_bytes_per_macrocell:
        raise ValueError(
            f"{json_path}: bytes_per_macrocell={meta['bytes_per_macrocell']} "
            f"does not match bins ({n_vperp}*{n_vpar}*{FLOAT_DTYPE.itemsize}="
            f"{expected_bytes_per_macrocell})"
        )

    expected_record_bytes = meta["total_macrocells"] * meta["bytes_per_macrocell"]
    if meta["record_size_bytes"] != expected_record_bytes:
        raise ValueError(
            f"{json_path}: record_size_bytes={meta['record_size_bytes']} "
            f"does not match total_macrocells*bytes_per_macrocell="
            f"{expected_record_bytes}"
        )

    bin_path = json_path.with_suffix(".bin")
    if not bin_path.exists():
        raise FileNotFoundError(f"Binary file not found: {bin_path}")

    needed_bytes = byte_offset + meta["bytes_per_macrocell"]
    actual_bytes = bin_path.stat().st_size
    if actual_bytes < needed_bytes:
        raise ValueError(
            f"{bin_path}: file too small for record {K}, macrocell {m}: "
            f"need at least {needed_bytes} bytes, got {actual_bytes}"
        )

    with bin_path.open("rb") as bf:
        bf.seek(byte_offset)
        raw = np.frombuffer(bf.read(meta["bytes_per_macrocell"]),
                            dtype=FLOAT_DTYPE)

    if raw.size != Nb:
        raise ValueError(
            f"{bin_path}: expected {Nb} float32 at record {K} macrocell {m}, "
            f"got {raw.size}"
        )

    # Copy so the array is writable and not backed by the mmap buffer.
    hist = raw.reshape(n_vperp, n_vpar).astype(np.float32, copy=True)
    return Spectrum(meta=meta, hist=hist, json_path=json_path,
                    macrocell=(mx, my, mz), cycle=cycle, record_index=K)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def species_from_json_path(path: Path) -> int | None:
    """Return S for a species_S.json path, or None if it does not match."""
    try:
        prefix, value = path.stem.split("_", 1)
        if prefix != "species":
            return None
        return int(value)
    except ValueError:
        return None


def subdomain_from_json_path(path: Path) -> int | None:
    """Return R for a subDomain_R/species_S.json path."""
    try:
        prefix, value = path.parent.name.split("_", 1)
        if prefix != "subDomain":
            return None
        return int(value)
    except ValueError:
        return None


def json_sort_key(path: Path) -> tuple[int, int, str]:
    """Sort by subdomain, then species, with stable path fallback."""
    rank = subdomain_from_json_path(path)
    species = species_from_json_path(path)
    return (
        rank if rank is not None else sys.maxsize,
        species if species is not None else sys.maxsize,
        str(path),
    )


def find_species_jsons(
    root: Path,
    species: int | None = None,
    subdomain: int | None = None,
) -> list[Path]:
    """Return matching ``subDomain_R/species_S.json`` files under *root*."""
    root = Path(root)
    if root.is_file():
        raise ValueError(
            f"Expected the macrocellSpectra root directory, got file: {root}"
        )
    if not root.is_dir():
        raise FileNotFoundError(root)

    jsons = root.glob("subDomain_*/species_*.json")
    matches = [
        p for p in jsons
        if not p.name.endswith(".tmp")
        and (species is None or species_from_json_path(p) == species)
        and (subdomain is None or subdomain_from_json_path(p) == subdomain)
    ]
    return sorted(matches, key=json_sort_key)


def enumerate_refs(
    json_paths: list[Path],
    cycle: int | None = None,
    macrocell: tuple[int, int, int] | None = None,
) -> list[tuple[Path, int, int, int, int]]:
    """Build the list of (json_path, mx, my, mz, cycle) tuples to load.

    For each species JSON, iterates over all (cycle, macrocell) combinations
    after applying the optional filters.

    Parameters
    ----------
    json_paths :
        List of ``species_S.json`` paths returned by ``find_species_jsons``.
    cycle :
        If given, only this cycle is included (must exist in ``records``).
    macrocell :
        If given as ``(mx, my, mz)``, only this macrocell is included.
    """
    refs: list[tuple[Path, int, int, int, int]] = []
    for jp in json_paths:
        with jp.open() as f:
            meta = json.load(f)
        records: list[int] = meta["records"]
        Mx, My, Mz = meta["macrocells_per_axis"]

        # Determine which cycles to include.
        if cycle is not None:
            cycles = [cycle] if cycle in records else []
            if not cycles:
                print(f"[warn] cycle {cycle} not in {jp.name} records {records}",
                      file=sys.stderr)
        else:
            cycles = list(records)

        # Determine which macrocells to include.
        if macrocell is not None:
            mx0, my0, mz0 = macrocell
            if not (0 <= mx0 < Mx and 0 <= my0 < My and 0 <= mz0 < Mz):
                print(f"[warn] macrocell {macrocell} out of range for {jp.name}",
                      file=sys.stderr)
                continue
            macrocells = [(mx0, my0, mz0)]
        else:
            macrocells = [
                (mx, my, mz)
                for mz in range(Mz)
                for my in range(My)
                for mx in range(Mx)
            ]

        for c in cycles:
            for mx, my, mz in macrocells:
                refs.append((jp, mx, my, mz, c))

    return refs


def iter_spectra(
    refs: list[tuple[Path, int, int, int, int]],
) -> Iterator[Spectrum]:
    """Load spectra from a list of (json_path, mx, my, mz, cycle) refs."""
    for jp, mx, my, mz, c in refs:
        try:
            yield load_spectrum(jp, mx, my, mz, c)
        except Exception as e:
            print(f"[warn] skipping mc({mx},{my},{mz}) cycle {c} "
                  f"from {jp.name}: {e}", file=sys.stderr)


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
    cb.set_label("counts (log)" if norm is not None else "counts")

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

    # --- Discover species JSON files ---
    try:
        json_paths = find_species_jsons(args.path,
                                        species=args.species,
                                        subdomain=args.subdomain)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not json_paths:
        print("No species_S.json files matched.", file=sys.stderr)
        return 1

    # --- Enumerate (json, mx, my, mz, cycle) references ---
    mc = tuple(args.macrocell) if args.macrocell is not None else None
    refs = enumerate_refs(json_paths, cycle=args.cycle, macrocell=mc)

    if not refs:
        print("No (cycle, macrocell) combinations matched.", file=sys.stderr)
        return 1
    print(f"Plotting {len(refs)} spectrum record(s)...")

    plotted = 0
    for spec in iter_spectra(refs):
        out = (args.save / (spec.save_stem + ".png")) if args.save else None
        plot_spectrum(spec, out_file=out, show=args.save is None)
        plotted += 1

    if plotted == 0:
        print("All records failed to load.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
