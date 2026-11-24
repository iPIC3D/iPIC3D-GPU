# Restart

[Documentation index](../README.md#documentation)

A checkpoint stores the fields, moments and particles.

## Launching a restart

Add `restart` to the [command line](running.md#command-line):

```shell
mpirun -np 8 ./iPIC3D input.inp restart            # the latest checkpoint
mpirun -np 8 ./iPIC3D input.inp restart=latest     # the same
mpirun -np 8 ./iPIC3D input.inp restart=A          # slot A; also: restart A
mpirun -np 8 ./iPIC3D input.inp restart=B          # slot B; also: restart B
```

## Input keys

| Key | Default | Description |
|---|---|---|
| `RestartDirName` | `data` | Root holding the [checkpoint slots](#checkpoint-slots). A fresh run empties it, see [Running](running.md#output-folder) |
| `RestartOutputCycle` | `5000` | Checkpoint interval. `<= 0` disables checkpoints |
| `RestartWriteMethod` | `adios2` if built, else `shdf5` | [Backend](output.md#backends) of the checkpoints this run writes |
| `CallFinalize` | `1` | Also write a checkpoint at the end. Requires `RestartOutputCycle > 0` |

All simulation parameters are read from the input file, as on a fresh start. `ncycles` counts the cycles of the restarted run, see [Cycle numbering](#cycle-numbering).

## Requirements

- A build with the backend of the checkpoint read: ADIOS2 or HDF5. With neither, a restart stops at startup. The HDF5 path is less tested than ADIOS2.
- The same global mesh, no more species than the checkpoint, and compatible boundary and input parameters.
- The same MPI topology, or a [refinement by powers of two](#restart-remapping).
- A checkpoint from another build, or without a build signature, gives a warning, not an error.

## Checkpoint slots

New checkpoints alternate between two reusable slots:

```text
RestartDirName/
  latest_restart.json
  settings.hdf  (serial HDF5 only)
  restart_A/    manifest.json  restart_<rank>.bp  or  restart<rank>.hdf
  restart_B/
```

Each checkpoint goes to the slot not holding the run's previous one: a fresh run starts with `restart_A`, and the first checkpoint after a restart goes to the slot that was not read. Rank 0 removes the slot's `manifest.json` before any rank file is replaced, and writes it, then replaces `latest_restart.json`, only once every rank has closed its file. A write cut off leaves its slot without a manifest and the other slot intact. Each checkpoint gets a `generation` one above the highest in the restart root.

With `AsyncOutput = 1`, writer threads write the rank files while the run goes on, and the manifest is written at a later cycle. Until then, a restart reads the previous checkpoint in the other slot. One checkpoint is written at a time, and the final one is complete when the run ends.

## Slot selection

Rank 0 selects the slot and broadcasts the choice:

- With `restart=A` or `restart=B`, it tries that slot, then the other.
- With `restart` or `restart=latest`, it tries the slot named by `latest_restart.json`, then the other. Without a usable file, it tries the slot whose manifest has the higher generation first.

A slot is usable when:

- its `manifest.json` exists and names that slot;
- the manifest names a backend of this build;
- it has a valid cycle, rank count and mesh;
- a rank file exists for each of the `nranks` ranks it records.

A restart reads a checkpoint with the backend the manifest records, whatever `RestartWriteMethod` says.

Startup prints the slot read and the one requested, every slot passed over with the reason, and, when the other slot holds a newer checkpoint, that the run's first checkpoint will overwrite it. These notes are also at the top of `SimulationData.txt`. If no slot is usable, the run stops and lists why.

**Restarting from the older slot starts a new branch:** its first checkpoint replaces the newer slot. When the label is a multiple of `RestartOutputCycle`, that checkpoint is written at the restart's first cycle. Copy the newer slot first if you need it.

## Cycle numbering

The loop counter is absolute. On a restart:

```text
first_cycle = checkpoint label
LastCycle   = first_cycle + ncycles
loop:         first_cycle ... LastCycle - 1
```

- A periodic checkpoint triggered at loop cycle `N` is written before cycle `N` completes and is labelled `N`. A restart executes the `N → N+1` update again.
- The final checkpoint, written after the last loop cycle `L`, is labelled `L + 1`.

For example, with `RestartOutputCycle = 100`, a restart from the checkpoint labelled `100` with `ncycles = 500` executes cycles 100 to 599.

## Checkpoint content

For a periodic checkpoint at cycle `N`, the payload is the state needed to execute cycle `N` again:

| Data | State |
|---|---|
| Particles | The particles at the start of cycle `N` |
| B | `B^N`, saved before `CalculateB(N)` |
| E | `E^N`. Checkpoints written by builds up to `a115269` hold `E^{N+1}` instead |
| ρ, J, pressure | The moment arrays at the cycle-`N` boundary, before the cycle-`N` mover and exchange complete |

A final checkpoint holds the same quantities at the boundary of cycle `L + 1`, after a fresh particle copy.

Files store only active data: E, B and the species density, current and pressure on active nodes (`NXN - 2`, `NYN - 2`, `NZN - 2`), and particles sorted by active cell with per-species offset and count metadata. A restart reads back E, B and the species densities, and rebuilds the ghost nodes.

> **Known limitation.** Checkpoints hold the node B only. A restart rebuilds the cell B by interpolation, which removes its grid-scale part, so the magnetic energy differs at the restart cycle. It was about 2 % on a 2-particles-per-cell GEM deck. E and the particles are exact.

## Restart remapping

A checkpoint can be loaded onto a different MPI topology when:

- the global cell counts (`nxc`, `nyc`, `nzc`) are unchanged;
- the destination refines the source by powers of two in each direction (no coarsening or arbitrary factors).

Each destination rank reads the fields and the particles of the source boxes it overlaps. For the output files of the old decomposition, see [Output](output.md#restarts-and-existing-files) and [Data analysis](data-analysis.md#restarts).

A restart reads the particles in chunks of at most 2²⁰ per array, so its host memory peaks at the particle arrays plus about 64 MB.

## Restart initialisation

On a restart the case initialiser still sets the external dipole field of `Dipole` and `Dipole2D`. E, B, the species densities and the particles (with their IDs) come from the checkpoint. The planet cases then reset the planet charge, and `TaylorGreen` resets the charge on its open faces.
