#!/usr/bin/env bash
# Profile the kernels for rooflines and speed of light (README.md): the
# measured ceilings of the GPU (profileCeilings), then each profile at each
# particles-per-cell value, once timed with CUDA events and REPETITIONS times
# under Nsight Compute.  It never plots: run plot_roofline.py on OUT
# afterwards.
#
# The folder of each GPU system calls it with its settings
# (profileGH200/profile.sh, profileRTX5080/profile.sh).  Every setting below
# can be set in the environment; BUILD and OUT have no default.
set -euo pipefail
# Nsight Compute writes its numbers in the format of the locale;
# plot_roofline.py reads the C format
export LC_ALL=C
(($# == 0)) || { echo "no options: set the settings in the environment" >&2; exit 1; }
here=$(cd "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

# ---- settings
BUILD=${BUILD:?set BUILD, the folder of the profile executables (build.sh)}
OUT=${OUT:?set OUT, a new folder for the results}
CELLS=${CELLS:-"64 64 64"}
PPC=(${PPC:-64 128 256 512})
PROFILES=(${PROFILES:-profileMover profileCellAwareMoment profileMomentKernelNew})
SPECIES=${SPECIES:-0}                 # 0: electrons, 1: ions
WARMUP=${WARMUP:-2}                   # launches before the timed or profiled ones
LAUNCHES=${LAUNCHES:-10}              # timed launches of the timed run
REPETITIONS=${REPETITIONS:-2}         # Nsight Compute runs of each point; 0: timed runs only
# Nsight Compute replay of the counter passes: kernel (saves the device
# memory the kernel can reach, and restores it before each pass: twice the
# memory) or application (runs the whole profile again for each pass)
REPLAY=${REPLAY:-kernel}
SUBCYCLES=${SUBCYCLES:-4}             # profileMover; 0: B of the input file (1 subcycle)
TAIL=${TAIL:-}                        # profileMomentKernelNew; empty: 1
read -ra launcher <<< "${LAUNCHER:-}" # the launch of one task on one GPU
# ----

[[ $REPLAY == kernel || $REPLAY == application ]] || { echo "REPLAY is kernel or application" >&2; exit 1; }
[[ ! -e $OUT ]] || { echo "$OUT exists: choose another OUT" >&2; exit 1; }
for exe in profileCeilings "${PROFILES[@]}"; do
  [[ -x $BUILD/$exe ]] || { echo "no $BUILD/$exe: run build.sh, or set BUILD" >&2; exit 1; }
done
mkdir -p "$OUT"
deck="$here/profileKernels.inp"

# on_gpu LOG COMMAND...: COMMAND on the GPU (LAUNCHER), its output in LOG;
# stops the script when it fails
on_gpu() {
  local log=$1
  shift
  if ! ${launcher[@]+"${launcher[@]}"} "$@" > "$log" 2>&1; then
    echo "failed: $log" >&2
    tail -n 5 "$log" >&2
    exit 1
  fi
}

# the kernel each profile launches, for the Nsight Compute filter
kernel_of() {
  case $1 in
    profileMover) echo moverSubcyclesKernel ;;
    profileCellAwareMoment) echo cellAwareMomentKernel ;;
    profileMomentKernelNew) echo momentKernelNew ;;
    *) echo "unknown profile $1" >&2; exit 1 ;;
  esac
}

# Speed of light: NVIDIA's sections, with the metrics of the GPU at hand; the
# report keeps their breakdowns and advice for the Nsight Compute GUI.
sections=(--section SpeedOfLight --section Occupancy --section LaunchStats)
# Rooflines, on the conventional pipelines (iPIC3D uses no tensor cores).
# Work: FP64 operations = DADD + DMUL + 2 DFMA of the active threads.
# Traffic: bytes of DRAM, L2 and L1.  Peaks: rate per cycle times cycles per
# second, the FP64 one of the DFMA pipe.  Atomics: the L2 requests and the L2
# atomic unit (the deposits); the FP64 pipe (the mover).  MIO, the SM's way
# to its shared units (shared memory, shuffles, special math, dynamic
# branches): its four interfaces (instructions issued, operand queue written
# and read, writeback to the registers), percent of peak.
metrics=(
  gpu__time_duration.sum
  smsp__sass_thread_inst_executed_op_dadd_pred_on.sum
  smsp__sass_thread_inst_executed_op_dmul_pred_on.sum
  smsp__sass_thread_inst_executed_op_dfma_pred_on.sum
  dram__bytes.sum
  lts__t_bytes.sum
  l1tex__t_bytes.sum
  sm__sass_thread_inst_executed_op_dfma_pred_on.sum.peak_sustained
  sm__cycles_elapsed.avg.per_second
  dram__bytes.sum.peak_sustained
  dram__cycles_elapsed.avg.per_second
  lts__t_bytes.sum.peak_sustained
  lts__cycles_elapsed.avg.per_second
  l1tex__t_bytes.sum.peak_sustained
  l1tex__cycles_elapsed.avg.per_second
  lts__t_requests_op_atom.sum
  lts__t_requests_op_red.sum
  lts__d_atomic_input_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__mio_inst_issued.avg.pct_of_peak_sustained_elapsed
  sm__mio_pq_write_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__mio_pq_read_cycles_active.avg.pct_of_peak_sustained_elapsed
  sm__mio2rf_writeback_active.avg.pct_of_peak_sustained_elapsed
)
METRICS=$(IFS=,; echo "${metrics[*]}")

{
  echo "date: $(date -Iseconds)"
  echo "host: $(hostname)"
  echo "build: $BUILD"
  echo "cells: $CELLS"
  echo "ppc: ${PPC[*]}"
  echo "profiles: ${PROFILES[*]}"
  echo "species: $SPECIES"
  echo "warmup: $WARMUP"
  echo "launches: $LAUNCHES"
  echo "repetitions: $REPETITIONS"
  echo "replay: $REPLAY"
  echo "subcycles: $SUBCYCLES"
  echo "tail: ${TAIL:-1}"
  echo "launcher: ${LAUNCHER:-none}"
  ((REPETITIONS == 0)) || ncu --version | tail -n 1
} > "$OUT/settings.txt"

echo "ceilings"
on_gpu "$OUT/ceilings.log" "$BUILD/profileCeilings"
grep -v '^{' "$OUT/ceilings.log"

printf 'profile\tkernel\tppc\trepetitions\tdirectory\n' > "$OUT/points.tsv"
for ppc in "${PPC[@]}"; do
  for profile in "${PROFILES[@]}"; do
    kernel=$(kernel_of "$profile")
    args=("$deck" --cells $CELLS --ppc "$ppc" --species "$SPECIES" --warmup "$WARMUP")
    [[ $profile == profileMover ]] && args+=(--subcycles "$SUBCYCLES")
    [[ $profile == profileMomentKernelNew && -n $TAIL ]] && args+=(--tail "$TAIL")
    name="${profile}_ppc${ppc}"
    dir="$OUT/$name"
    mkdir "$dir"
    echo "$name: timed"
    on_gpu "$dir/timed.log" "$BUILD/$profile" "${args[@]}" --launches "$LAUNCHES"
    for ((r = 1; r <= REPETITIONS; r++)); do
      echo "$name: Nsight Compute $r"
      # the first launch after the warmup, at the clocks of a run (no lock),
      # caches as the profile left them
      on_gpu "$dir/profiled$r.log" ncu --config-file off --kernel-name "$kernel" \
        --launch-skip "$WARMUP" --launch-count 1 \
        --clock-control none --cache-control none --replay-mode "$REPLAY" \
        "${sections[@]}" --metrics "$METRICS" --export "$dir/profiled$r" \
        "$BUILD/$profile" "${args[@]}" --launches 1
      # offline: no GPU needed (.ncu-rep, or .ncu-repz for newer versions)
      reports=("$dir/profiled$r".ncu-rep*)
      ncu --config-file off --import "${reports[0]}" --page raw --csv \
        --print-units base > "$dir/counters$r.csv"
    done
    printf '%s\t%s\t%s\t%s\t%s\n' "$profile" "$kernel" "$ppc" "$REPETITIONS" "$name" \
      >> "$OUT/points.tsv"
  done
done
touch "$OUT/COMPLETE"
echo "collected in $OUT; plot with: python3 $here/plot_roofline.py $OUT"
