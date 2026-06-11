#!/usr/bin/env bash
set -euo pipefail

# Downloads the selected artifacts from the cluster into:
#   raw_data/<folder>_<variant>/<prefix>_influence_scores
#   raw_data/<folder>_<variant>/<prefix>_validation
#   raw_data/<folder>_<variant>/<prefix>_full.pt
#
# Edit only the FOLDERS, MODEL_VARIANTS, and PREFIXES blocks below before running.

REMOTE_HOST="ai-ws-213"
REMOTE_ROOT="/home/dolderer/pinnfluence_code_dreckig_smiley/model_zoo_cluster"
LOCAL_ROOT="raw_data"

# Comment out / Fill in
FOLDERS=(
  "allen_cahn_float64"
  "burgers_float64"
  "diffusion_float64"
  "drift_diffusion_float64"
  "navier_stokes_nd_float64"
  "poisson_disk_float64"
  "wave_float64"
)

# Comment out / Fill in
MODEL_VARIANTS=(
  "good"
  "bad"
)

# Good models use PROBLEMS from pinnfluence/utils/defaults.py.
# Bad models use BAD_PROBLEMS from pinnfluence/utils/defaults.py.
# Seed 0 is used for every problem except wave, which uses seed 2.
declare -A PREFIXES=(
  ["good:allen_cahn_float64"]="allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_0_soft"
  ["good:burgers_float64"]="burgers_adam_50000_adam_12000_lbfgs_2500_domain_500_boundary_500_initial_3_x_32_hidden_float64_True_0_soft"
  ["good:diffusion_float64"]="diffusion_adam_15000_adam_5000_lbfgs_1000_domain_100_boundary_100_initial_3_x_32_hidden_float64_True_0_soft"
  ["good:drift_diffusion_float64"]="drift_diffusion_adam_15000_adam_5000_lbfgs_1000_domain_100_boundary_100_initial_3_x_64_hidden_float64_True_0_soft"
  ["good:navier_stokes_nd_float64"]="navier_stokes_nd_adam_100000_adam_25000_lbfgs_7500_domain_2500_boundary_0_initial_3_x_64_hidden_float64_True_0_soft"
  ["good:poisson_disk_float64"]="poisson_disk_adam_50000_adam_12000_lbfgs_2500_domain_500_boundary_0_initial_3_x_32_hidden_float64_True_0_soft"
  ["good:wave_float64"]="wave_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_5_x_100_hidden_float64_True_2_soft"

  ["bad:allen_cahn_float64"]="allen_cahn_adam_100000_adam_0_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_0_soft"
  ["bad:burgers_float64"]="burgers_adam_50000_adam_12000_lbfgs_500_domain_100_boundary_100_initial_3_x_32_hidden_float64_True_0_soft"
  ["bad:diffusion_float64"]="diffusion_adam_15000_adam_5000_lbfgs_10_domain_2_boundary_2_initial_3_x_32_hidden_float64_True_0_soft"
  ["bad:drift_diffusion_float64"]="drift_diffusion_adam_15000_adam_5000_lbfgs_200_domain_20_boundary_20_initial_3_x_64_hidden_float64_True_0_soft"
  ["bad:navier_stokes_nd_float64"]="navier_stokes_nd_adam_100000_adam_25000_lbfgs_1500_domain_500_boundary_0_initial_3_x_64_hidden_float64_True_0_soft"
  ["bad:poisson_disk_float64"]="poisson_disk_adam_15000_adam_5000_lbfgs_100_domain_20_boundary_0_initial_3_x_32_hidden_float64_True_0_soft"
  ["bad:wave_float64"]="wave_adam_100000_adam_0_lbfgs_2500_domain_500_boundary_500_initial_5_x_100_hidden_float64_True_2_soft"
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
  for variant in "${MODEL_VARIANTS[@]}"; do
    key="${variant}:${folder}"
    prefix="${PREFIXES[$key]:-}"
    if [[ -z "$prefix" || "$prefix" == CHANGE_ME_* ]]; then
      missing+=("$key")
    fi
  done
done

if [[ ${#missing[@]} -gt 0 ]]; then
  echo "Set the run prefix for these variants in PREFIXES before downloading:" >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

rsync_opts=(-av --partial --progress)
if [[ "$DRY_RUN" -eq 1 ]]; then
  rsync_opts+=(--dry-run)
fi

for folder in "${FOLDERS[@]}"; do
  for variant in "${MODEL_VARIANTS[@]}"; do
    local_dir="${LOCAL_ROOT}/${folder}_${variant}"
    key="${variant}:${folder}"
    prefix="${PREFIXES[$key]}"
    echo "Downloading ${folder} (${variant}) with prefix: ${prefix}"

    mkdir -p "$local_dir"
    for suffix in "${ARTIFACT_SUFFIXES[@]}"; do
      remote_path="${REMOTE_HOST}:${REMOTE_ROOT}/${folder}/${prefix}_${suffix}"
      echo "  rsync ${remote_path} -> ${local_dir}/"
      rsync "${rsync_opts[@]}" "$remote_path" "$local_dir/"
    done
  done
done
