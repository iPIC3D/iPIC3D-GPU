# Data analysis

[Documentation index](../README.md#documentation)

Every `DataAnalysisCycle` cycles, a host thread analyses the particles while the field solver runs, on a GPU stream of the lowest priority. All analyses are off by default. The [benchmark modes](building.md#benchmark-modes) run them without writing or leave them out.

| Analysis | Switch | Result |
|---|---|---|
| Velocity histogram | `VelocityHistogram` | 3D (u, v, w) histogram per rank (100 bins per axis) or per macrocell (32 per axis) |
| Gaussian mixture | `VelocityGMM` | A fit of each histogram, compressing the distribution into a few Gaussians |
| Spectra | `VelocitySpectra` | (v∥, v⊥) relative to the local B, per macrocell (64 × 64 bins) |

A **macrocell** is a block of cells of a rank. The macrocell analyses sort their species by cell first, and a macrocell takes the particles of its interior cells. A particle exactly on a rank face, which the sort can put in a ghost cell, is left out of them (the histogram of a rank counts it). On a planet run, the first analysis also sees the particles loaded inside the planet, which the planet step of the first cycle absorbs or moves out of the planet.

## Keys

Check the `Data analysis` line and the macrocell sizes printed at startup: a misspelt key is [ignored](input-file.md#syntax). Bin counts are compile-time constants in `src/include/dataAnalysis/dataAnalysisConfig.cuh`.

### General

| Key | Default | Description |
|---|---|---|
| `DataAnalysisCycle` | `50` | Analyse every N cycles; `0` disables all analyses |
| `DataAnalysisOutputDir` | `./` | Parent of `velocityHistogram/`, `velocityGMM/`, `macrocellSpectra/`, relative to the current directory (not inside `SaveDirName`) |

### Histograms

| Key | Default | Description |
|---|---|---|
| `VelocityHistogram` | `0` | Histogram of the species of `VelocityHistGMMSpecies`, weighted by \|q\| and summed in fixed point, so identical in every run and for any particle order |
| `VelocityHistGMMSpecies` | all `1` | Mask of the histograms and fits. With `VelocityHistogram = 1`, at least one is on. A species that is off is not sorted for them |
| `VelocityHistogramMacrocells` | `0` | `1`: one histogram (and fit) per macrocell instead of per rank, 128 kB of GPU memory per macrocell (printed at the start) and up to 32 bytes per non-empty bin for the GMM points and output. Needs `VelocityHistogram = 1` |
| `VelocityHistogramMacrocellNx`, `VelocityHistogramMacrocellNy`, `VelocityHistogramMacrocellNz` | `MacrocellNx`, `MacrocellNy`, `MacrocellNz` | Macrocell size in cells. With macrocells on, each from 1 to the cells of a rank along that axis (1 on the flat axis of a 2D run), at most 65535 macrocells per rank |
| `VelocityHistogramFixedRange` | `1` | `1`: bin [`VelocityHistogramMin`, `VelocityHistogramMax`] on the three axes. `0`: the range of the finite velocities of the species at that cycle in the rank or macrocell. On an axis without spread, or with bins not wider than 4 ε of the largest velocity, the bins span `VelocityHistogramMax − VelocityHistogramMin` with the velocities in the middle bin; on an axis without finite velocities, the fixed range |
| `VelocityHistogramMax` | `0.2` if `qom < 0`, else `0.09` | Per species, positive, above `VelocityHistogramMin` by a finite width. The spectra bin v∥ in [−max, max] and v⊥ in [0, max] |
| `VelocityHistogramMin` | `-VelocityHistogramMax` | Per species |

### Gaussian mixture

| Key | Default | Description |
|---|---|---|
| `VelocityGMM` | `0` | Fit each histogram on its non-empty bins, weighted by their content. Needs `VelocityHistogram = 1` |
| `GMMComponents` | `10` | Gaussians per fit, 1 to 256. GPU memory: (`GMMComponents` + 1) × 8 bytes per non-empty bin, plus a quarter for growth. Macrocell fits run in batches of about 185 MB; a rank histogram is one fit, at most 8 MB per component plus 8 MB. The last fit of each species and region stays on the host, 105 bytes per component. A restart may change it |
| `GMMMaxIterations` | `100` | EM iterations at most, plus one when the last one prunes a component, which gives its weight back to the others |
| `GMMConvergenceThreshold` | `1e-5` | Stop when the log-likelihood per unit weight gains less, or decreases (from the 7th iteration on) |
| `GMMWarmStart` | `1` | Start from the previous fit of the species and region (of this run, not after a restart), redrawing components that were pruned or nearly so; after a redraw all weights restart equal. Otherwise: equal weights, random means in the ball inscribed in the bin range, diagonal covariance `uth²`, `vth²`, `wth²` |
| `GMMNormalize` | `1` | Fit velocities divided by the largest velocity of the histogram's bins (`VelocityHistogramMax` with the default range) |
| `GMMCheckCovariance` | `1` | Reset non-finite covariance elements and raise variances to `GMMMinVariance` |
| `GMMCovarianceTolerance` | `1e-9` | Smallest Cholesky pivot, in the units of the fit: a nearly singular covariance is raised on its diagonal to stay invertible, and is written so |
| `GMMMinVariance` | `1e-4` | With `GMMCheckCovariance`, in the units of the fit: the smallest variance of each velocity given the previous ones (the Cholesky pivot), which bounds a component's density by that of a round one of this variance. The default suits `GMMNormalize = 1` |
| `GMMPruneComponents` | `1` | Drop components of weight below `GMMPruneThreshold`, the smallest first, one at a time |
| `GMMPruneThreshold` | `0.005` | With pruning, must be below 1 / `GMMComponents`, the weights of a cold start |

### Spectra and macrocells

| Key | Default | Description |
|---|---|---|
| `VelocitySpectra` | `0` | Spectra of the species in `VelocitySpectraSpecies`, 16 kB of GPU memory per macrocell (printed at the start). A particle where \|B\| ≤ 1e-12 is left out |
| `VelocitySpectraSpecies` | all `0` | Mask of the spectra. With `VelocitySpectra = 1`, at least one is on |
| `VelocitySpectraMacrocellNx`, `VelocitySpectraMacrocellNy`, `VelocitySpectraMacrocellNz` | `MacrocellNx`, `MacrocellNy`, `MacrocellNz` | Macrocell size in cells. With the spectra on, each from 1 to the cells of a rank, at most 524287 macrocells per rank |
| `MacrocellNx`, `MacrocellNy`, `MacrocellNz` | `0` | Common default size of the spectra and histogram macrocells |

With two different sizes, macrocell m of the spectra is not macrocell m of the histograms: each `.json` gives its own tiling. Along an axis where the cells of a rank are not a multiple of the size, the last macrocell has the remaining cells. On a rank with fewer cells than the size, there is only one. The spectra are summed with float atomics, so their last bits change from run to run.

## Files

Each analysis writes, per rank, binary records appended in cycle order, and a `.json` of the same name describing the rank, its regions (the rank or the macrocell tiling), the bins and the record layout.

| Analysis | File |
|---|---|
| Histogram | `velocityHistogram/subDomain<rank>_histograms.bin` |
| GMM | `velocityGMM/subDomain<rank>_gmm.bin` |
| Spectra | `macrocellSpectra/subDomain<rank>_spectra.bin` |

Only non-empty data is written. A species writes no record at a cycle where it has no data, and the records of a cycle are written at once.

- **Histogram record:** per region, its range, the number of non-empty bins, their contents (float32: the summed |q| times the `.json` `weight_scale`) and which bins they are (u fastest), as a list of indices (uint16 for a macrocell, uint32 for a rank) or, when smaller, a bitmap. The `.json` field `index_encoding` says which.
- **GMM record:** the fitted regions only (the rank, or the macrocells with particles in their range). For each fit, its region, log-likelihood (float64), EM steps and number of active components; for each active component, its slot (its index among `GMMComponents`, kept by the warm start), weight, mean and the upper triangle of its covariance (float32). Each quantity is one array, in the order the `.json` gives. The log-likelihood, in the units of the velocities, is that of the parameters before the last EM step. With `GMMCheckCovariance = 0`, which lets a variance fall to `GMMCovarianceTolerance`, a nearly singular covariance can lose positive definiteness in float32.
- **Spectra record:** the range of the species (v∥ and v⊥ bounds, float64), then the non-empty bins of each macrocell (v∥ fastest), as a list of uint16 indices or, when smaller, a bitmap.

Read and plot them with the [post-processing scripts](tools.md#post-processing).

**Failed writes.** A file is only appended to, so a run that fails while writing leaves the records of earlier cycles intact. A failed write (a full disk) is taken back, and the run stops. After a run killed during a write, the file can end with some records of that cycle and one cut short, where the readers stop and which a restart drops.

## Restarts

- A **fresh start** empties the folder of each analysis that is on, files kept aside by earlier runs included, and keeps those of an earlier run (of all its ranks) for analyses that are off as `*_before_cycle0.*`, so that a restart neither cuts nor continues them.
- A **restart** cuts the records from its first cycle on, and any record cut short, then appends. This also applies to an analysis switched off, and with every analysis off. A file of another record format is kept aside instead.
- A restart onto **another decomposition** (the `topology` of the `.json`), or whose regions, macrocells or bins differ from the records, warns and keeps the files (on a remap, those of every rank of the earlier run) as `*_before_cycle<N>.*`, with `_2`, `_3`, ... if such a file exists, then starts them again. On a remap, the files of analyses switched off are left as they are.
- The GMM records follow the histogram regions and are kept aside with them.
