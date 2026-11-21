#!/usr/bin/env python3
"""
Read the binary records of the iPIC3D-GPU data analysis.

Each MPI rank writes one pair of files per analysis, in the folder of the
analysis:
  velocityHistogram/subDomain<rank>_histograms.bin   3D (u, v, w) histograms,
                                                    one record per cycle and
                                                    species
  velocityGMM/subDomain<rank>_gmm.bin                Gaussian-mixture fits,
                                                    one record per cycle and
                                                    species
  macrocellSpectra/subDomain<rank>_spectra.bin       (v_par, v_perp) spectra,
                                                    one record per cycle and
                                                    species
and next to each a .json of the same name: the rank, its regions (the rank,
or its macrocell tiling), the bins, and the format of the records: the fields
of their header ("record_header") and the arrays that follow it, with their
shapes in header fields ("record_arrays"). The records are appended in cycle
order; each starts with a 32-byte header whose first word identifies the file
and whose last field is the size of the record. A species has no record at a
cycle where it has no data.

The histograms and the spectra keep their non-empty bins only: per region its
number of them ("counts"), then their contents ("weights", region after
region, by increasing bin, the first axis fastest) and which bins they are
("index", region after region: the list of their indices, or the bitmap of
the region when smaller, see "index_encoding"). The GMM records keep the
fitted regions only, and of each fit its active components.

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
for g in read_gmm("velocityGMM/subDomain0_gmm.bin", species=1):
    fit = g.fit(5)            # region 5: its active components, or None
    maps = g.dense()          # (regions, components) arrays of all fits
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

BITMAP_CHUNK = 512  # bitmaps decoded at once, to bound the memory


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
    index: np.ndarray      # (non_empty,): the bin of each weight in its region

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


def decode_index(meta: dict, counts: np.ndarray,
                 stream: np.ndarray) -> np.ndarray:
    """The bins of the weights of a sparse record from its "index" bytes:
    per region its counts[r] indices (list_dtype) or, when those take more
    than bitmap_bytes, its bitmap (bit b % 8 of byte b / 8 for bin b)."""
    encoding = meta["index_encoding"]
    order = ">" if meta["endian"] == "big" else "<"
    listed = np.dtype(encoding["list_dtype"]).newbyteorder(order)
    nbytes = int(encoding["bitmap_bytes"])
    counts = counts.astype(np.int64)
    bitmap = counts * listed.itemsize > nbytes
    size = np.where(bitmap, nbytes, counts * listed.itemsize)
    if size.sum() != len(stream):
        raise ValueError(f"index of {len(stream)} bytes, the counts give "
                         f"{size.sum()}")
    index = np.empty(int(counts.sum()), np.int32)  # < INT_MAX bins in all
    # the lists, in region order
    index[np.repeat(~bitmap, counts)] = \
        stream[np.repeat(~bitmap, size)].view(listed)
    # the bitmaps: their set bits, region after region, bins increasing
    rows = np.flatnonzero(bitmap)
    maps = stream[np.repeat(bitmap, size)].reshape(-1, nbytes)
    bins = []
    for first in range(0, len(rows), BITMAP_CHUNK):
        bits = np.unpackbits(maps[first:first + BITMAP_CHUNK], axis=1,
                             bitorder="little")
        if np.any(bits.sum(1) != counts[rows[first:first + BITMAP_CHUNK]]):
            raise ValueError("a bitmap with other bins than its count")
        bins.append(np.nonzero(bits)[1])
    if bins:
        index[np.repeat(bitmap, counts)] = np.concatenate(bins)
    # also a set bit in the padding of the last byte of a bitmap
    if len(index) and index.max() >= np.prod(list(meta["bins"].values())):
        raise ValueError("a bin index beyond the bins of a region")
    return index


def read_sparse(path, cycle: int | None = None,
                species: int | None = None) -> Iterator[SparseRecord]:
    """The records of a histograms or spectra file (its axes are the keys of
    "bins" in its description), of a cycle and a species if given."""
    for meta, h, a in read_records(path):
        if "index_encoding" not in meta:
            raise ValueError(f"{path}: histograms or spectra of an older "
                             "format (HST1/SPC1), which this reader does not "
                             "read")
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
            a["weights"], decode_index(meta, a["counts"], a["index"]))


@dataclass
class GMMFit:
    """The fit of a region: its active components, by increasing slot."""
    weights: np.ndarray      # (n,)
    means: np.ndarray        # (n, 3)
    covariances: np.ndarray  # (n, 3, 3)
    slots: np.ndarray        # (n,): index among the GMMComponents of the fit
    steps: int               # EM steps
    log_likelihood: float


@dataclass
class DenseGMM:
    """The fits of every region of a record as (regions, components) arrays:
    0 for a component not in the record; steps -1 and log-likelihood NaN for
    a region without a fit."""
    weights: np.ndarray         # (R, K)
    means: np.ndarray           # (R, K, 3)
    covariances: np.ndarray     # (R, K, 3, 3)
    steps: np.ndarray           # (R,)
    log_likelihood: np.ndarray  # (R,)

    @property
    def fitted(self) -> np.ndarray:
        """(R,): the regions with a fit."""
        return self.steps >= 0


@dataclass
class GMMRecord:
    """The fits of a rank at one cycle and species: the fitted regions and,
    fit after fit, their active components."""
    cycle: int
    species: int
    regions: int                # of the rank (its layout)
    region: np.ndarray          # (F,): the fitted regions, increasing
    steps: np.ndarray           # (F,)
    log_likelihood: np.ndarray  # (F,)
    offsets: np.ndarray         # (F + 1,): the components of fit f
    slots: np.ndarray           # (C,)
    weights: np.ndarray         # (C,)
    means: np.ndarray           # (C, 3)
    covariances: np.ndarray     # (C, 3, 3)

    def fit(self, region: int) -> GMMFit | None:
        """The fit of a region, None without one."""
        f = int(np.searchsorted(self.region, region))
        if f == len(self.region) or self.region[f] != region:
            return None
        part = slice(self.offsets[f], self.offsets[f + 1])
        return GMMFit(self.weights[part], self.means[part],
                      self.covariances[part], self.slots[part],
                      int(self.steps[f]), float(self.log_likelihood[f]))

    def dense(self, components: int | None = None) -> DenseGMM:
        """Every region with components slots: pass GMMComponents for arrays
        of the same shape for every record (default: the largest slot of
        this one + 1)."""
        R = self.regions
        K = int(self.slots.max()) + 1 if components is None else components
        if K <= self.slots.max():
            raise ValueError(f"{K} components, a slot is {self.slots.max()}")
        at = (np.repeat(self.region, np.diff(self.offsets)), self.slots)
        w = np.zeros((R, K), self.weights.dtype)
        mu = np.zeros((R, K, 3), self.means.dtype)
        cov = np.zeros((R, K, 3, 3), self.covariances.dtype)
        w[at], mu[at], cov[at] = self.weights, self.means, self.covariances
        steps = np.full(R, -1, self.steps.dtype)
        ll = np.full(R, np.nan)
        steps[self.region], ll[self.region] = self.steps, self.log_likelihood
        return DenseGMM(w, mu, cov, steps, ll)


def read_gmm(path, cycle: int | None = None,
             species: int | None = None) -> Iterator[GMMRecord]:
    """The records of a subDomain<rank>_gmm.bin file, of a cycle and a
    species if given."""
    upper = np.triu_indices(3)  # the order of the stored covariances
    for meta, h, a in read_records(path):
        if "slot" not in a:
            raise ValueError(f"{path}: fits of an older format (GMM1), which "
                             "this reader does not read")
        if (cycle is not None and h["cycle"] != cycle) or \
                (species is not None and h["species"] != species):
            continue
        packed = a["covariances"]
        cov = np.empty((len(packed), 3, 3), packed.dtype)
        cov[:, upper[0], upper[1]] = packed
        cov[:, upper[1], upper[0]] = packed
        yield GMMRecord(
            int(h["cycle"]), int(h["species"]), int(h["regions"]),
            a["region"], a["steps"], a["log_likelihood"],
            np.concatenate(([0], np.cumsum(a["counts"], dtype=np.int64))),
            a["slot"], a["weights"], a["means"], cov)


def is_gmm(path) -> bool:
    """Whether a .bin file holds GMM fits (its description says so)."""
    meta = json.loads(Path(path).with_suffix(".json").read_text())
    return "steps" in (name for name, *_ in meta["record_arrays"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", help="a .bin records file")
    args = parser.parse_args()
    if is_gmm(args.path):
        print(f"{'cycle':>8} {'species':>7} {'regions':>7} {'fits':>7} "
              f"{'components':>10}")
        for g in read_gmm(args.path):
            print(f"{g.cycle:8d} {g.species:7d} {g.regions:7d} "
                  f"{len(g.region):7d} {len(g.weights):10d}")
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
