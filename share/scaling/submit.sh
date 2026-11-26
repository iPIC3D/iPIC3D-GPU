#!/bin/bash
# Submits a scaling campaign: one Slurm job per node count. See README.md.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./submit.sh SYSTEM/KIND [--dry-run] [MIN_NODES [MAX_NODES]] [-- SBATCH_OPTION...]

  SYSTEM/KIND  a deck folder: mi250x/strong, mi250x/weak, a100/strong, a100/weak
  --dry-run    check the decks and print the sbatch commands; submit nothing
  MIN_NODES    submit only this node count, or the range up to MAX_NODES;
               without counts every deck of the folder is submitted
  -- ...       extra sbatch options, applied last (--qos, --time, --exclude, ...)

Environment: IPIC3D_SLURM_ACCOUNT, IPIC3D_PROFILE (sourced in each job),
IPIC3D_EXECUTABLE.
EOF
}

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

if (( $# == 0 )) || [[ $1 == -h || $1 == --help ]]; then
  usage
  exit 0
fi
deck_set=${1%/}
shift
system=${deck_set%%/*}
kind=${deck_set#*/}
if [[ $kind != strong && $kind != weak ]] || [[ ! -f $here/$system/system.sh ]]; then
  echo "Not a deck folder: $deck_set (expected SYSTEM/strong or SYSTEM/weak)" >&2
  exit 2
fi

dry_run=0
counts=()
sbatch_options=()
while (( $# > 0 )); do
  case $1 in
    --dry-run) dry_run=1 ;;
    --) shift; sbatch_options=("$@"); break ;;
    *)
      if [[ ! $1 =~ ^[1-9][0-9]*$ ]]; then
        echo "Not a node count: $1" >&2
        exit 2
      fi
      counts+=("$1")
      ;;
  esac
  shift
done
if (( ${#counts[@]} > 2 )); then
  usage >&2
  exit 2
fi
min_nodes=${counts[0]:-1}
max_nodes=${counts[1]:-${counts[0]:-1000000}}

for var in IPIC3D_SLURM_ACCOUNT IPIC3D_PROFILE IPIC3D_EXECUTABLE; do
  if [[ -z ${!var:-} ]]; then
    echo "$var is not set" >&2
    exit 2
  fi
done
IPIC3D_PROFILE=$(realpath -- "$IPIC3D_PROFILE")
IPIC3D_EXECUTABLE=$(realpath -- "$IPIC3D_EXECUTABLE")
if [[ ! -r $IPIC3D_PROFILE ]]; then
  echo "Profile is not readable: $IPIC3D_PROFILE" >&2
  exit 1
fi
if [[ ! -x $IPIC3D_EXECUTABLE ]]; then
  echo "Not an executable: $IPIC3D_EXECUTABLE" >&2
  exit 1
fi
export IPIC3D_PROFILE IPIC3D_EXECUTABLE

SYSTEM_DIR=$here/$system
source "$SYSTEM_DIR/system.sh"  # TASKS_PER_NODE, SBATCH_OPTIONS, system_srun

# Select the decks and check that each asks for one rank per GPU.
deck_dir=$SYSTEM_DIR/$kind
decks=()
nodes_of=()
while IFS= read -r deck; do
  [[ $(basename -- "$deck") =~ ^GEM${kind}N([0-9]+)\.inp$ ]] || continue
  nodes=${BASH_REMATCH[1]}
  (( nodes >= min_nodes && nodes <= max_nodes )) || continue
  ranks=$(awk -F= '$1 ~ /^[ \t]*[XYZ]LEN[ \t]*$/ { p *= $2; n++ }
                   END { if (n == 3) print p }' p=1 "$deck")
  if [[ $ranks != $((nodes * TASKS_PER_NODE)) ]]; then
    echo "$deck: XLEN*YLEN*ZLEN = ${ranks:-?}, expected $nodes nodes x $TASKS_PER_NODE ranks" >&2
    exit 1
  fi
  decks+=("$deck")
  nodes_of+=("$nodes")
done < <(printf '%s\n' "$deck_dir"/GEM"$kind"N*.inp | sort -V)
if (( ${#decks[@]} == 0 )); then
  echo "No deck in $deck_dir for $min_nodes to $max_nodes nodes" >&2
  exit 1
fi

if [[ $kind == strong ]]; then time_limit=00:10:00; else time_limit=00:10:00; fi
campaign=$deck_dir/results_$(date +%Y%m%d_%H%M%S)
if (( dry_run )); then
  echo "Dry run: nothing is created or submitted."
else
  mkdir "$campaign"
  {
    echo "date $(date -Is)"
    echo "executable $IPIC3D_EXECUTABLE"
    echo "profile $IPIC3D_PROFILE"
    echo "sbatch_options ${sbatch_options[*]}"
  } > "$campaign/campaign.txt"
fi

for i in "${!decks[@]}"; do
  nodes=${nodes_of[$i]}
  run_dir=$campaign/N$nodes
  command=(
    sbatch --parsable
    --account="$IPIC3D_SLURM_ACCOUNT"
    --job-name="$system-$kind-N$nodes"
    --nodes="$nodes"
    --ntasks=$((nodes * TASKS_PER_NODE))
    --ntasks-per-node="$TASKS_PER_NODE"
    --time="$time_limit"
    --export=ALL
    "${SBATCH_OPTIONS[@]}"
    "${sbatch_options[@]}"
    --chdir="$run_dir"
    --output="$campaign/stdout_N${nodes}_%j"
    --error="$campaign/stderr_N${nodes}_%j"
    "$here/job.sh" "$SYSTEM_DIR" "$kind" "$run_dir/$(basename -- "${decks[$i]}")"
  )
  if (( dry_run )); then
    printf '%q ' "${command[@]}"
    printf '\n'
    continue
  fi
  # The job reads a copy, so later edits of the deck do not change the campaign.
  mkdir "$run_dir"
  cp -- "${decks[$i]}" "$run_dir/"
  job_id=$("${command[@]}")
  echo "N$nodes: job $job_id"
  echo "job N$nodes $job_id" >> "$campaign/campaign.txt"
done
(( dry_run )) || echo "Campaign: $campaign"
