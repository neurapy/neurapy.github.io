#!/usr/bin/env bash
set -euo pipefail

# Downloads the selected artifacts from the cluster into:
#   data/<folder>/<prefix>_influence_scores
#   data/<folder>/<prefix>_validation
#   data/<folder>/<prefix>_full.pt
#
# Edit only the PREFIXES block below before running.

REMOTE_HOST="ai-ws-213"
REMOTE_ROOT="/home/dolderer/pinnfluence_code_dreckig_smiley/model_zoo_cluster"
LOCAL_ROOT="data"

FOLDERS=(
  # "allen_cahn_float64"
  "burgers_float64"
  "diffusion_float64"
  "drift_diffusion_float64"
  "navier_stokes_nd_float64"
  "poisson_disk_float64"
  "wave_float64"
)

declare -A PREFIXES=(
  # ["allen_cahn_float64"]="allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_12_soft"
  ["burgers_float64"]="burgers_adam_50000_adam_12000_lbfgs_2500_domain_500_boundary_500_initial_3_x_32_hidden_float64_True_9_soft"
  ["diffusion_float64"]="diffusion_adam_15000_adam_5000_lbfgs_1000_domain_100_boundary_100_initial_3_x_32_hidden_float64_True_9_soft"
  ["drift_diffusion_float64"]="drift_diffusion_adam_15000_adam_5000_lbfgs_1000_domain_100_boundary_100_initial_3_x_64_hidden_float64_True_9_soft"
  ["navier_stokes_nd_float64"]="navier_stokes_nd_adam_100000_adam_25000_lbfgs_7500_domain_2500_boundary_0_initial_3_x_64_hidden_float64_True_9_soft"
  ["poisson_disk_float64"]="poisson_disk_adam_50000_adam_12000_lbfgs_2500_domain_500_boundary_0_initial_3_x_32_hidden_float64_True_9_soft"
  ["wave_float64"]="wave_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_5_x_100_hidden_float64_True_9_soft"
)

ARTIFACT_SUFFIXES=(
  "influence_scores"
  "validation"
  "full.pt"
)

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
fi

missing=()
for folder in "${FOLDERS[@]}"; do
  prefix="${PREFIXES[$folder]:-}"
  if [[ -z "$prefix" || "$prefix" == CHANGE_ME_* ]]; then
    missing+=("$folder")
  fi
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Set the run prefix for these folders in PREFIXES before downloading:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

rsync_opts=(-av --partial --progress)
if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_opts+=(--dry-run)
fi

for folder in "${FOLDERS[@]}"; do
  prefix="${PREFIXES[$folder]}"
  local_dir="${LOCAL_ROOT}/${folder}"

  mkdir -p "$local_dir"
  echo "Downloading ${folder} with prefix: ${prefix}"

  for suffix in "${ARTIFACT_SUFFIXES[@]}"; do
    remote_path="${REMOTE_HOST}:${REMOTE_ROOT}/${folder}/${prefix}_${suffix}"
    echo "  rsync ${remote_path} -> ${local_dir}/"
    rsync "${rsync_opts[@]}" "$remote_path" "$local_dir/"
  done
done
