#!/usr/bin/env python3
"""
Plot total number of particles across all MPI ranks vs cycle.
Reads particleNum{rank}.csv files from the data/ directory.
"""

import glob
import pandas as pd
import matplotlib.pyplot as plt

# Find all rank files
files = sorted(glob.glob("data/particleNum*.csv"))
if not files:
    raise FileNotFoundError("No particleNum*.csv files found in data/")

print(f"Found {len(files)} rank files: {[f.split('/')[-1] for f in files]}")

# Read and sum across ranks (truncate to shortest file)
frames = [pd.read_csv(f) for f in files]
min_len = min(len(df) for df in frames)
df_total = frames[0].iloc[:min_len].copy()
for df in frames[1:]:
    species_cols = [c for c in df.columns if c != "cycle"]
    df_total[species_cols] += df[species_cols].iloc[:min_len].values

species_cols = [c for c in df_total.columns if c != "cycle"]
# Filter out species that are always zero
active = [c for c in species_cols if df_total[c].sum() > 0]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each active species
for col in active:
    ax.plot(df_total["cycle"], df_total[col], label=col, linewidth=1.2)

ax.set_xlabel("Cycle")
ax.set_ylabel("Number of particles (all ranks)")
ax.set_title(f"Total particle count ({len(files)} ranks)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()

outfile = "total_particles.png"
fig.savefig(outfile, dpi=150)
print(f"Saved {outfile}")

# --- Second plot: total electrons vs total ions (weight-corrected) ---
# Species 0,1 = solar wind (weight 1.0)
# Species 2+ = planetary, divided by MacroParticleWeightRatio to get SW-equivalent count
macro_weight_ratio = [400.0, 400.0, 400.0, 400.0]  # from input file
ns_solar_wind = 2

# Build weight-corrected totals
df_weighted = df_total.copy()
for c in active:
    idx = int(c.strip().split("species")[-1])
    if idx >= ns_solar_wind:
        df_weighted[c] = df_total[c] / macro_weight_ratio[idx - ns_solar_wind]

# Even species indices (0, 2, 4, ...) = electrons
# Odd  species indices (1, 3, 5, ...) = ions
electron_cols = [c for c in active if int(c.strip().split("species")[-1]) % 2 == 0]
ion_cols      = [c for c in active if int(c.strip().split("species")[-1]) % 2 == 1]

fig2, ax2 = plt.subplots(figsize=(10, 6))

if electron_cols:
    ax2.plot(df_weighted["cycle"], df_weighted[electron_cols].sum(axis=1),
             label="Total electrons", linewidth=1.5)
if ion_cols:
    ax2.plot(df_weighted["cycle"], df_weighted[ion_cols].sum(axis=1),
             label="Total ions", linewidth=1.5)

ax2.set_xlabel("Cycle")
ax2.set_ylabel("Effective particle count (SW-equivalent)")
ax2.set_title(f"Total electrons vs ions, weight-corrected ({len(files)} ranks)")
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()

outfile2 = "total_electrons_ions.png"
fig2.savefig(outfile2, dpi=150)
print(f"Saved {outfile2}")

plt.show()