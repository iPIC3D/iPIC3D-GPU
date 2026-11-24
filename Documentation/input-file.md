# Input file

[Documentation index](../README.md#documentation)

This page gives the syntax, the keys common to every run, and an index of all other keys. Example input files, called *decks* in these pages, are in [`share/inputfiles`](../share/inputfiles); each lists its keys with their defaults.

## Syntax

```text
# a comment
key = value            # a comment after a value
npcelx = 4 4 4 4       # a list: values separated by spaces
```

- Keys and string values are case-sensitive.
- A repeated key keeps its last value. A value may continue on the following lines, until a blank line or a line with a key.
- A line containing `EndConfigFile` ends the file.
- A missing **required** key stops the run. An **unknown or misspelt key is ignored without a message**.

Value types:

| Type | Accepted values |
|---|---|
| Flag | `PERIODICX`/`Y`/`Z`, `PERIODICX_P`/`Y_P`/`Z_P`, `ParticleSplitting`, `ParticleMerging`, `CallFinalize` and the values of `TrackParticleID`. False: `false`, `f`, `no`, `n`, `0`, `none` in any case. **Anything else is true** |
| Switch | Exactly `0` or `1`, otherwise an input error. Used by `AsyncOutput` and the on/off keys of the data analysis |
| Integer on/off | Other on/off keys, such as `ForceWallB`, `HaloOverlap`, `yes_sal`, `currentFromAmpere`: nonzero is on. **Text such as `yes` or `true` reads as 0** |
| `yes`/`no` | `PoissonCorrection` and `divBCorrection` are on only for exactly `yes` |
| Species list | One value per species, see [below](#species). The output and data-analysis lists also take one value for all species |

## Run and time

| Key | Default | Description |
|---|---|---|
| `Case` | required | Initial condition, see [Simulation cases](cases.md) |
| `SimulationName` | required | Prefix of the VTK and parallel-HDF5 file names |
| `FieldWriteMethod` | required | Field output backend, see [Output](output.md#backends) |
| `dt` | required | Time step, in ω_pi⁻¹ |
| `ncycles` | required | Cycles of this run, also after a restart |
| `th` | `1.0` | Time-centring θ of the implicit scheme, at least `0.5` |
| `c` | `1.0` | Speed of light in code units, positive, see [Units](units.md#speed-of-light-input-c) |

## Domain and decomposition

| Key | Default | Description |
|---|---|---|
| `Lx`, `Ly`, `Lz` | `10.0` | Box size |
| `nxc`, `nyc`, `nzc` | `64` | Global cells |
| `XLEN`, `YLEN`, `ZLEN` | `1` | MPI ranks along each axis; the split of the cells is in [Running](running.md#ranks-threads-and-gpus) |
| `PERIODICX`, `PERIODICY`, `PERIODICZ` | `true` | Periodicity of the fields |
| `PERIODICX_P`, `PERIODICY_P`, `PERIODICZ_P` | the field value | Periodicity of the particles |

A 2D run uses one periodic cell along the flat axis, such as `nyc = 1`, `PERIODICY = 1`.

## Species

`ns` (required) is the number of species and `nsTestPart` (default `0`) the number of [test-particle](particles.md#test-particles) species, indexed after them. These lists are required and take exactly one value per species, in order (`pitch_angle` and `energy` only with `nsTestPart > 0`, the exosphere lists only with `ns_planetary > 0`):

| Keys | Number of values | Meaning |
|---|---|---|
| `rhoINIT` | `ns` | Initial density magnitude, 4π n, see [Units](units.md#input-parameters) |
| `rhoINJECT` | `ns` | Injected density magnitude on [open inflow faces](boundary-conditions.md#open-inflow-faces) |
| `uth`, `vth`, `wth` | `ns` | Thermal speeds, one component |
| `u0`, `v0`, `w0` | `ns` | Drift speeds. Some cases use them as [current weights](cases.md#case-specific-keys) |
| `npcelx`, `npcely`, `npcelz` | `ns + nsTestPart` | Particles per cell along each axis |
| `qom` | `ns + nsTestPart` | Charge-to-mass ratio |
| `pitch_angle`, `energy` | `nsTestPart` | Test-particle loading |
| `NeutralSurfaceDensity`, `ExosphericScaleHeight`, `PhotoionizationFrequency`, `MacroParticleWeightRatio` | `ns_planetary` | [Exosphere](particles.md#exosphere-ionisation) parameters |

`TrackParticleID` turns on [particle IDs](particles.md#particle-ids), off by default. It takes one flag for all species, or flags in species order (`ns + nsTestPart`): missing species stay off and extra values are ignored.

The run stops at startup, naming the key, for:

- a list with another number of values, a value that is not a number of the list's type (a whole number for `npcel*`), or values written without a space (`1.0.5`, `3+4`);
- `ns` below 1 or `nsTestPart` below 0;
- `ns_solar_wind` outside 0 to `ns`, or a negative `ns_planetary`;
- with `AddExosphereInjection = 1`: `ns_planetary` other than `ns - ns_solar_wind`, or, with planetary species, `npcelx * npcely * npcelz = 0` for species 0, by which the exosphere sizes its particles;
- a negative `npcel*`, `rhoINIT` or `rhoINJECT`;
- a species with `rhoINIT` or `rhoINJECT` above 0 but `npcelx * npcely * npcelz = 0`. A species with neither density may load none, as planetary species do.

A `qom` of 0 is allowed with a warning, but the charge sign qom/|qom| and the mass q/qom divide by it:

- its particles, or the charge held fixed for it on a planet or on the open faces of a `TaylorGreen` restart, make the field solve non-finite at the first cycle, which stops the run;
- with neither, its bulk-flow energy in `ConservedQuantities.txt` is `nan`;
- a test-particle species with `qom` 0 gets a `nan` charge, which only the serial-HDF5 checkpoints hold.

## Fields and boundaries

| Key | Default | Description |
|---|---|---|
| `B0x`, `B0y`, `B0z` | `0.0` | Background magnetic field |
| `bcEMfaceXleft`, `bcEMfaceXright`, `bcEMfaceYleft`, `bcEMfaceYright`, `bcEMfaceZleft`, `bcEMfaceZright` | required | [Field boundary code](boundary-conditions.md#boundary-codes) of each face |
| `bcPfaceXleft`, `bcPfaceXright`, `bcPfaceYleft`, `bcPfaceYright`, `bcPfaceZleft`, `bcPfaceZright` | `1` | [Particle boundary code](boundary-conditions.md#boundary-codes) of each face |

## Index of the other keys

| Topic | Keys |
|---|---|
| [Field solver](field-solver.md#parameters) | `SolverType`, `GMREStol`, `DivCleaningTol`, `ChebyshevMaxIter`, `ChebyshevEigMin`, `ChebyshevEigMax`, `BlockJacobiSweeps`, `BlockJacobiOmega`, `PoissonCorrection`, `PoissonCorrectionCycle`, `divBCorrection`, `divBCorrectionCycle`, `PoissonChebMaxIter`, `PoissonChebEigRatio`, `PoissonChebRescaleEigMin`, `PoissonChebRescaleEigMax`, `Smooth`, `SmoothNiter`, `HaloOverlap`, `SolverSourceGrowthLimit`, `SolverStopOnNonConvergence` |
| [Boundary conditions](boundary-conditions.md#boundary-keys) | `yes_sal`, `n_layers_sal`, `sal_tangential_B`, `ApplyInflowBcsEImage`, `OutflowLayerFields`, `ForceWallB` |
| [Planet](boundary-conditions.md#planet-surface-dipole-dipole2d) | `Planet_radius`, `x_center_planet`, `y_center_planet`, `z_center_planet`, `planetReflectionType`, `PlanetDiagnosticsCycle` |
| [Dipole](cases.md#case-specific-keys) | `x_center_dipole`, `y_center_dipole`, `z_center_dipole`, `B1z` |
| [Simulation cases](cases.md#case-specific-keys) | `delta`, `pertGEM`, `pertHump`, `deltaxHump`, `deltayHump`, `kxHump`, `kyHump`, `currentFromAmpere`, `spatiallyVaryingThermal` |
| [Particles](particles.md) | `NiterMover`, `SortingCycle`, `ParticleSplitting`, `ParticleMerging`, `ParticleSplittingThreshold`, `ParticleMergingThreshold`, `ns_solar_wind`, `ns_planetary`, `AddExosphereInjection`, `RmaxExosphereInjection`, `MaxExosphereParticlesPerSpecies` |
| [Output](output.md#output-keys) | `SaveDirName`, `FieldOutputCycle`, `FieldOutputTag`, `MomentsOutputTag`, `ParticlesOutputCycle`, `ParticlesOutputTag`, `ParticlesOutputSpecies`, `ParticlesOutputTrackedFraction`, `ParticlesOutputMaxPerCell`, `ParticlesWriteMethod`, `AsyncOutput`, `DiagnosticsOutputCycle`, `TestPartOutputCycle`, `ParaviewScriptPath` |
| [Restart](restart.md#input-keys) | `RestartDirName`, `RestartOutputCycle`, `RestartWriteMethod`, `CallFinalize` |
| [Data analysis](data-analysis.md#keys) | `DataAnalysisCycle`, `DataAnalysisOutputDir`, `VelocityHistogram`, `VelocityHistGMMSpecies`, `VelocityHistogramMacrocells`, `VelocityHistogramMacrocellNx/Ny/Nz`, `VelocityHistogramFixedRange`, `VelocityHistogramMin`, `VelocityHistogramMax`, `VelocityGMM`, `GMMComponents`, `GMMMaxIterations`, `GMMConvergenceThreshold`, `GMMWarmStart`, `GMMNormalize`, `GMMCheckCovariance`, `GMMCovarianceTolerance`, `GMMMinVariance`, `GMMPruneComponents`, `GMMPruneThreshold`, `VelocitySpectra`, `VelocitySpectraSpecies`, `VelocitySpectraMacrocellNx/Ny/Nz`, `MacrocellNx/Ny/Nz` |

## Keys without effect

The code reads these keys, but they have no effect beyond the note:

| Key | Status |
|---|---|
| `CGtol` | Tolerance of the unused host CG solver |
| `NpMaxNpRatio` | Unused. A build without `NDEBUG` (`Debug`, or a type such as `Default`) aborts below 1 |
| `PushWithBatTime`, `PushWithEatTime`, `ImplSusceptTime` | Unused |
| `ImplSusceptMode` | Only `initial` is accepted |
| `Vinj` | Unused |
| `TestParticlesOutputCycle` | Unused. The test-particle output interval is `TestPartOutputCycle` |
| `bcPHIfaceXleft`, `bcPHIfaceXright`, `bcPHIfaceYleft`, `bcPHIfaceYright`, `bcPHIfaceZleft`, `bcPHIfaceZright` | Only copied into the settings of the HDF5 output |
| `B1x`, `B1y` | Only printed. The dipole uses `B1z` |
| `L_square` | Deprecated name of `Planet_radius`: warns, and is ignored if `Planet_radius` is set |
| `WriteMethod` | Removed: **an input error**. Use the three write methods |
