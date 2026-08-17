#!/bin/bash -l
## Script for LUMI-G
## Set IPIC3D_SLURM_ACCOUNT, IPIC3D_PROFILE, and IPIC3D_EXECUTABLE before use.

#SBATCH -J strong-scaling
#SBATCH -t 00:30:00
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=6
#SBATCH -o stdout_%j
#SBATCH -e stderr_%j

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
TASKS_PER_NODE=8

usage() {
	echo "Usage: $0 [--dry-run] [--exclude NODELIST] [MIN_NODES [MAX_NODES]]"
	echo "  Required environment:"
	echo "    IPIC3D_SLURM_ACCOUNT: Slurm billing account"
	echo "    IPIC3D_PROFILE: readable shell profile to source in each job"
	echo "    IPIC3D_EXECUTABLE: executable iPIC3D binary"
	echo "  --exclude, -x: exclude Slurm nodes (for example, nid[0010-0012],nid0020)"
	echo "  No range: submit every GEMstrongN*.inp input"
	echo "  One value: submit exactly that node count"
	echo "  Two values: submit available inputs in the inclusive range"
}

input_rank_count() {
	awk -F= '
		/^[[:space:]]*(XLEN|YLEN|ZLEN)[[:space:]]*=/ {
			product *= ($2 + 0)
			count++
		}
		END {
			if (count != 3) exit 1
			print product
		}
	' product=1 "$1"
}

submit_scaling_jobs() {
	local dry_run=0
	local exclude_nodes=""
	local -a range_args=()
	while (( $# > 0 )); do
		case $1 in
			--dry-run)
				dry_run=1
				;;
			--exclude|-x)
				if (( $# < 2 )) || [[ -z $2 ]]; then
					echo "$1 requires a Slurm node list" >&2
					usage >&2
					return 2
				fi
				exclude_nodes=$2
				shift
				;;
			--exclude=*)
				exclude_nodes=${1#*=}
				if [[ -z $exclude_nodes ]]; then
					echo "--exclude requires a Slurm node list" >&2
					usage >&2
					return 2
				fi
				;;
			--)
				shift
				range_args+=("$@")
				break
				;;
			-*)
				echo "Unknown option: $1" >&2
				usage >&2
				return 2
				;;
			*)
				range_args+=("$1")
				;;
		esac
		shift
	done
	set -- "${range_args[@]}"

	if (( $# > 2 )); then
		usage >&2
		return 2
	fi

	local min_nodes=1
	local max_nodes=2147483647
	if (( $# >= 1 )); then
		min_nodes=$1
		max_nodes=$1
	fi
	if (( $# == 2 )); then
		max_nodes=$2
	fi

	if [[ ! $min_nodes =~ ^[0-9]+$ || ! $max_nodes =~ ^[0-9]+$ ]] ||
	   (( min_nodes < 1 || max_nodes < min_nodes )); then
		echo "Invalid node range: ${min_nodes} to ${max_nodes}" >&2
		usage >&2
		return 2
	fi

	if [[ -z ${IPIC3D_SLURM_ACCOUNT:-} ]]; then
		echo "IPIC3D_SLURM_ACCOUNT is not set" >&2
		usage >&2
		return 2
	fi
	if [[ -z ${IPIC3D_PROFILE:-} ]]; then
		echo "IPIC3D_PROFILE is not set" >&2
		usage >&2
		return 2
	fi
	if [[ -z ${IPIC3D_EXECUTABLE:-} ]]; then
		echo "IPIC3D_EXECUTABLE is not set" >&2
		usage >&2
		return 2
	fi

	if [[ $IPIC3D_PROFILE != /* ]]; then
		IPIC3D_PROFILE="${PWD}/${IPIC3D_PROFILE}"
	fi
	if [[ $IPIC3D_EXECUTABLE != /* ]]; then
		IPIC3D_EXECUTABLE="${PWD}/${IPIC3D_EXECUTABLE}"
	fi
	if [[ ! -r $IPIC3D_PROFILE ]]; then
		echo "Profile is not readable: ${IPIC3D_PROFILE}" >&2
		return 1
	fi
	if [[ ! -x $IPIC3D_EXECUTABLE ]]; then
		echo "iPIC3D executable is not executable: ${IPIC3D_EXECUTABLE}" >&2
		return 1
	fi
	export IPIC3D_PROFILE IPIC3D_EXECUTABLE

	local input_file filename nodes topology_ranks expected_ranks
	local -a selected_inputs=()
	local -a selected_nodes=()
	while IFS= read -r input_file; do
		filename=$(basename -- "$input_file")
		[[ $filename =~ ^GEMstrongN([0-9]+)\.inp$ ]] || continue
		nodes=${BASH_REMATCH[1]}
		(( nodes >= min_nodes && nodes <= max_nodes )) || continue

		topology_ranks=$(input_rank_count "$input_file") || {
			echo "Could not read XLEN/YLEN/ZLEN from ${filename}" >&2
			return 1
		}
		expected_ranks=$((nodes * TASKS_PER_NODE))
		if (( topology_ranks != expected_ranks )); then
			echo "${filename}: topology has ${topology_ranks} ranks; expected ${expected_ranks} for ${nodes} nodes" >&2
			return 1
		fi

		selected_inputs+=("$input_file")
		selected_nodes+=("$nodes")
	done < <(printf '%s\n' "${SCRIPT_DIR}"/GEMstrongN*.inp | sort -V)

	if (( ${#selected_inputs[@]} == 0 )); then
		echo "No GEMstrongN*.inp inputs found in node range ${min_nodes}-${max_nodes}" >&2
		return 1
	fi

	local campaign_dir="${SCRIPT_DIR}/results_$(date +%Y%m%d_%H%M%S)"
	if (( ! dry_run )) && ! mkdir "$campaign_dir"; then
		echo "Campaign directory already exists; wait one second and try again: ${campaign_dir}" >&2
		return 1
	fi

	local index run_dir
	local -a sbatch_command
	for index in "${!selected_inputs[@]}"; do
		input_file=${selected_inputs[$index]}
		nodes=${selected_nodes[$index]}
		expected_ranks=$((nodes * TASKS_PER_NODE))
		run_dir="${campaign_dir}/N${nodes}"
		(( dry_run )) || mkdir -p "$run_dir"

		sbatch_command=(
			sbatch
			--account="$IPIC3D_SLURM_ACCOUNT"
			--export=ALL
			--nodes="$nodes"
			--ntasks="$expected_ranks"
			--job-name="strong-N${nodes}"
			--chdir="$run_dir"
			--output="${campaign_dir}/stdout_N${nodes}_%j"
			--error="${campaign_dir}/stderr_N${nodes}_%j"
		)
		if [[ -n $exclude_nodes ]]; then
			sbatch_command+=(--exclude="$exclude_nodes")
		fi
		sbatch_command+=("$SCRIPT_PATH" "$input_file")

		if (( dry_run )); then
			printf 'DRY RUN:'
			printf ' %q' "${sbatch_command[@]}"
			printf '\n'
		else
			"${sbatch_command[@]}"
		fi
	done

	echo "Campaign directory: ${campaign_dir}"
}

if [[ -z ${SLURM_JOB_ID:-} ]]; then
	submit_scaling_jobs "$@"
	exit $?
fi

if (( $# != 1 )); then
	echo "A scheduled job requires exactly one input file argument" >&2
	exit 2
fi

inputfile=$1
if [[ ! -f $inputfile ]]; then
	echo "Input file not found: ${inputfile}" >&2
	exit 1
fi

if [[ ! -r ${IPIC3D_PROFILE:-} ]]; then
	echo "Profile is not readable: ${IPIC3D_PROFILE:-<unset>}" >&2
	exit 1
fi
if [[ ! -x ${IPIC3D_EXECUTABLE:-} ]]; then
	echo "iPIC3D executable is not executable: ${IPIC3D_EXECUTABLE:-<unset>}" >&2
	exit 1
fi

set +u
source "$IPIC3D_PROFILE"
set -u

export MPICH_GPU_SUPPORT_ENABLED=1

gpu_wrapper="${PWD}/select_gpu_${SLURM_JOB_ID}"
cat <<'EOF' > "$gpu_wrapper"
#!/bin/bash
export ROCR_VISIBLE_DEVICES=$SLURM_LOCALID
exec "$@"
EOF
chmod +x "$gpu_wrapper"
trap 'rm -f "$gpu_wrapper"' EXIT

CPU_BIND="mask_cpu:7e000000000000,7e00000000000000"
CPU_BIND="${CPU_BIND},7e0000,7e000000"
CPU_BIND="${CPU_BIND},7e,7e00"
CPU_BIND="${CPU_BIND},7e00000000,7e0000000000"

export OMP_NUM_THREADS=6
executable=$IPIC3D_EXECUTABLE

echo "Running $(basename -- "$inputfile") on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_NTASKS} MPI ranks"
srun --cpu-bind="$CPU_BIND" "$gpu_wrapper" "$executable" "$inputfile"