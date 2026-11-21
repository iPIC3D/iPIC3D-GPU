#!/usr/bin/env python3
"""
Read the binary records of the iPIC3D-GPU data analysis.

Each MPI rank writes one pair of files per analysis, in the folder of the
analysis:
  velocityHistogram/subDomain<rank>_histograms.bin   3D (u, v, w) histograms,
                                                    one record per cycle and
                                                    species
  velocityGMM/subDomain<rank>_gmm.bin                Gaussian-mixture fits,
                                                    one record per cycle
  macrocellSpectra/subDomain<rank>_spectra.bin       (v_par, v_perp) spectra,
                                                    one record per cycle and
                                                    species
and next to each a .json of the same name: the rank, its regions (the rank,
or its macrocell tiling), the bins, and the format of the records: the fields
of their header ("record_header") and the arrays that follow it, with their
shapes in header fields ("record_arrays"). The records are appended in cycle
order; each starts with a 32-byte header whose first word identifies the file
and whose last field is the size of the record.

The histograms and the spectra keep their non-empty bins only: per region its
number of them ("counts"), then their contents ("weights") and their indices
in the region ("bins", the first axis fastest), region after region.

Examples
--------
# List the records of a file:
python read_analysis_records.py runs/.../velocityGMM/subDomain0_gmm.bin

# In Python:
from read_analysis_records import read_sparse, read_gmm
for h in read_sparse("velocityHistogram/subDomain0_histograms.bin",
                     species=0):
    dense = h.dense(5)        # region 5 as a (w, v, u) array
    x, w = h.points(5)        # its non-empty bins: centres (n, 3), contents
for s in read_sparse("macrocellSpectra/subDomain0_spectra.bin"):
    s.dense(2)                # macrocell 2 as a (v_perp, v_par) array
for g in read_gmm("velocityGMM/subDomain0_gmm.bin"):
    g.weights[1, 5]           # species 1, region 5: (components,)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def read_records(path) -> Iterator[tuple[dict, np.void, dict]]:
    """The description of a .bin file and its whole records, as (the
    description, the header, {array name: array}), the arrays of the shapes
    the description gives."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    order = ">" if meta["endian"] == "big" else "<"
    header = np.dtype([(name, np.dtype(dtype).newbyteorder(order))
                       for name, dtype in meta["record_header"]])
    data = np.memmap(path, np.uint8, mode="r") if path.stat().st_size \
        else np.zeros(0, np.uint8)
    at = 0
    while at + header.itemsize <= data.size:
        h = np.frombuffer(data, header, 1, at)[0]
        size = int(h["record_bytes"])
        if h["magic"] != meta["record_magic"] or size < header.itemsize or \
                at + size > data.size:
            break  # a record cut short
        o, arrays = at + header.itemsize, {}
        for name, dtype, shape in meta["record_arrays"]:
            shape = tuple(int(h[d]) if isinstance(d, str) else d for d in shape)
            dt = np.dtype(dtype).newbyteorder(order)
            count = int(np.prod(shape))
            if o + dt.itemsize * count > at + size:
                raise ValueError(f"{path}: the arrays of the record at byte "
                                 f"{at} end past its size {size}")
            arrays[name] = np.frombuffer(data, dt, count, o).reshape(shape)
            o += dt.itemsize * count
        yield meta, h, arrays
        at += size


@dataclass
class SparseRecord:
    """The non-empty bins of the regions of a rank at one cycle and species:
    a histogram (ranges per region) or a spectrum (range of the species)."""
    cycle: int
    species: int
    bins: tuple[int, ...]  # along each axis, the first fastest
    ranges: np.ndarray     # (regions, 2 * axes): the minima, then the maxima
    offsets: np.ndarray    # (regions + 1,): the bins of region r
    weights: np.ndarray    # (non_empty,)
    index: np.ndarray      # (non_empty,): the bin in its region

    def dense(self, r: int) -> np.ndarray:
        """Region r as a dense array, the last axis first: (w, v, u) for a
        histogram, (v_perp, v_par) for a spectrum."""
        out = np.zeros(int(np.prod(self.bins)), dtype=self.weights.dtype)
        part = slice(self.offsets[r], self.offsets[r + 1])
        out[self.index[part]] = self.weights[part]
        return out.reshape(self.bins[::-1])

    def points(self, r: int) -> tuple[np.ndarray, np.ndarray]:
        """The non-empty bins of region r: their centres (n, axes), the first
        axis first, and their contents."""
        part = slice(self.offsets[r], self.offsets[r + 1])
        index = self.index[part].astype(np.int64)
        axes = len(self.bins)
        i = np.stack([index // int(np.prod(self.bins[:a])) % self.bins[a]
                      for a in range(axes)], 1)
        lo, hi = self.ranges[r, :axes], self.ranges[r, axes:]
        return lo + (i + 0.5) * (hi - lo) / np.array(self.bins), \
            self.weights[part]


@dataclass
class Macrocell:
    """A macrocell of a rank: its indices along x, y, z, its first cell in
    the global grid and its cells per axis."""
    index: tuple[int, int, int]
    first_cell: tuple[int, int, int]
    cells: tuple[int, int, int]


def macrocell_of(meta: dict, region: int) -> Macrocell | None:
    """Region m = (mz * My + my) * Mx + mx of a records file described by
    meta, None when its regions are not macrocells (a histogram per rank)."""
    if meta.get("regions") != "macrocells":
        return None
    Mx, My, _ = meta["macrocells_per_axis"]
    index = (region % Mx, region // Mx % My, region // (Mx * My))
    tiles = [meta["axis_tiling"][a][i] for a, i in zip("xyz", index)]
    offset = meta["subdomain_global_offset_cells"]
    return Macrocell(index, tuple(o + t[0] for o, t in zip(offset, tiles)),
                     tuple(t[1] for t in tiles))


def read_sparse(path, cycle: int | None = None,
                species: int | None = None) -> Iterator[SparseRecord]:
    """The records of a histograms or spectra file (its axes are the keys of
    "bins" in its description), of a cycle and a species if given."""
    for meta, h, a in read_records(path):
        if (cycle is not None and h["cycle"] != cycle) or \
                (species is not None and h["species"] != species):
            continue
        R = int(h["regions"])
        if "ranges" in a:
            ranges = a["ranges"]
        else:  # the range of the species, [min, max] per axis
            ranges = np.tile(a["range"].reshape(-1, 2).T.ravel(), (R, 1))
        yield SparseRecord(
            int(h["cycle"]), int(h["species"]), tuple(meta["bins"].values()),
            ranges, np.concatenate(([0], np.cumsum(a["counts"], dtype=np.int64))),
            a["weights"], a["bins"])


@dataclass
class GMMRecord:
    """The fits of every species and region of a rank at one cycle; arrays
    indexed [species, region, ...]."""
    cycle: int
    log_likelihood: np.ndarray  # (S, R)
    weights: np.ndarray         # (S, R, K)
    means: np.ndarray           # (S, R, K, 3)
    covariances: np.ndarray     # (S, R, K, 3, 3)
    steps: np.ndarray           # (S, R): -1 without a fit at this cycle

    @property
    def fitted(self) -> np.ndarray:
        """(S, R): the regions with a fit at this cycle."""
        return self.steps >= 0


def read_gmm(path, cycle: int | None = None) -> Iterator[GMMRecord]:
    """The records of a subDomain<rank>_gmm.bin file, of a cycle if given."""
    for _, h, a in read_records(path):
        if cycle is None or h["cycle"] == cycle:
            yield GMMRecord(int(h["cycle"]), a["log_likelihood"], a["weights"],
                            a["means"], a["covariances"], a["steps"])


def is_gmm(path) -> bool:
    """Whether a .bin file holds GMM fits (its description says so)."""
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    return "steps" in (name for name, *_ in meta["record_arrays"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="a .bin records file")
    args = parser.parse_args()
    if is_gmm(args.path):
        print(f"{'cycle':>8} {'species':>7} {'regions':>7} {'fitted':>7} "
              f"{'components':>10}")
        for g in read_gmm(args.path):
            S, R, K = g.weights.shape
            print(f"{g.cycle:8d} {S:7d} {R:7d} {int(g.fitted.sum()):7d} "
                  f"{K:10d}")
        return
    print(f"{'cycle':>8} {'species':>7} {'regions':>7} {'non-empty':>10} "
          f"{'filled':>7}")
    for r in read_sparse(args.path):
        R = len(r.ranges)
        filled = len(r.weights) / (R * np.prod(r.bins)) if R else 0
        print(f"{r.cycle:8d} {r.species:7d} {R:7d} {len(r.weights):10d} "
              f"{100 * filled:6.2f}%")


if __name__ == "__main__":
    main()
