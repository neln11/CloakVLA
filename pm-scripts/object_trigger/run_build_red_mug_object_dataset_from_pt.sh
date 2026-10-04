#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

pick_existing_dir() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done
  echo "$1"
}

latest_poison_dir() {
  local base_dir="$1"
  local prefix="$2"
  local latest=""
  if [[ ! -d "${base_dir}" ]]; then
    return 1
  fi
  while IFS= read -r candidate; do
    if compgen -G "${candidate}/*_idx.pt" > /dev/null; then
      latest="${candidate}"
    fi
  done < <(find "${base_dir}" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" | sort)
  [[ -n "${latest}" ]] || return 1
  echo "${latest}"
}

EXP_ID="${EXP_ID:-red_mug_object_l1_data_only}"
ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/ablation_runs/object_l1}"
RUN_ROOT="${OBJECT_TRIGGER_ROOT:-${ABLATION_ROOT}/red_mug_object}"
RESULTS_PG_BASE="${RESULTS_PG_BASE:-${ROOT_DIR}/libero_datasets_ps/results_pg}"
CLEAN_DATA_ROOT="${CLEAN_DATA_ROOT:-$(pick_existing_dir "${ROOT_DIR}/libero_datasets/libero_hdf5/libero_object" "${USER_ROOT}/libero_assets/libero_object")}"
POISON_PREFIX="${POISON_PREFIX:-object_red_mug}"

if [[ -z "${POISON_DIR:-}" ]]; then
  POISON_DIR="$(latest_poison_dir "${RESULTS_PG_BASE}" "${POISON_PREFIX}")" || {
    echo "[RedMugDataset][ERROR] No complete ${POISON_PREFIX}_* poison directory found under ${RESULTS_PG_BASE}" >&2
    echo "[RedMugDataset][ERROR] Generate .pt first with pm-scripts/object_trigger/run_generate_red_mug_poison_object.sh" >&2
    exit 1
  }
fi

HDF5_DIR="${HDF5_DIR:-${ABLATION_ROOT}/hdf5/${EXP_ID}/libero_object_poisoned_no_noops}"
TFDS_ROOT="${TFDS_ROOT:-${ABLATION_ROOT}/tfds/${EXP_ID}}"
TARGET_TASK_FILE="${TARGET_TASK_FILE:-pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo.hdf5}"

[[ -d "${CLEAN_DATA_ROOT}" ]] || { echo "[RedMugDataset][ERROR] CLEAN_DATA_ROOT not found: ${CLEAN_DATA_ROOT}" >&2; exit 1; }
[[ -d "${POISON_DIR}" ]] || { echo "[RedMugDataset][ERROR] POISON_DIR not found: ${POISON_DIR}" >&2; exit 1; }
[[ -f "${CLEAN_DATA_ROOT}/${TARGET_TASK_FILE}" ]] || {
  echo "[RedMugDataset][ERROR] target HDF5 not found: ${CLEAN_DATA_ROOT}/${TARGET_TASK_FILE}" >&2
  exit 1
}

echo "[RedMugDataset] EXP_ID=${EXP_ID}"
echo "[RedMugDataset] POISON_DIR=${POISON_DIR}"
echo "[RedMugDataset] CLEAN_DATA_ROOT=${CLEAN_DATA_ROOT}"
echo "[RedMugDataset] HDF5_DIR=${HDF5_DIR}"
echo "[RedMugDataset] TFDS_ROOT=${TFDS_ROOT}"

if [[ "${RUN_INJECT:-1}" == "1" ]]; then
  if [[ "${FORCE_REINJECT:-0}" != "1" ]] \
      && [[ -f "${HDF5_DIR}/poison_metadata.json" ]] \
      && compgen -G "${HDF5_DIR}/*.hdf5" > /dev/null; then
    echo "[RedMugDataset] Reusing injected HDF5: ${HDF5_DIR}"
  else
    rm -rf "${HDF5_DIR}"
    mkdir -p "$(dirname "${HDF5_DIR}")"
    LIBERO_POISON_PT_DIR="${POISON_DIR}" \
    LIBERO_CLEAN_SPATIAL_DIR="${CLEAN_DATA_ROOT}" \
    LIBERO_POISONED_SPATIAL_DIR="${HDF5_DIR}" \
    LIBERO_TARGET_TASK_FILE="${TARGET_TASK_FILE}" \
    LIBERO_POISON_DEMO_FRACTION="${LIBERO_POISON_DEMO_FRACTION:-0.2}" \
    LIBERO_POISON_FRAME_FRACTION="${LIBERO_POISON_FRAME_FRACTION:-1.0}" \
    LIBERO_POISON_SOURCE_LABEL_FRACTION="${LIBERO_POISON_SOURCE_LABEL_FRACTION:-0.0}" \
    LIBERO_POISON_SELECT_FROM_IDX="${LIBERO_POISON_SELECT_FROM_IDX:-true}" \
    LIBERO_VISUAL_ONLY_MODE="${LIBERO_VISUAL_ONLY_MODE:-true}" \
    LIBERO_VERIFY_ACTIONS_UNCHANGED="${LIBERO_VERIFY_ACTIONS_UNCHANGED:-true}" \
    "${PM_SCRIPTS_DIR}/inject_to_hdf5/run_inject_poison_object.sh"
  fi
else
  echo "[RedMugDataset] RUN_INJECT=0, skipping HDF5 injection."
fi

if [[ "${RUN_RLDS:-1}" == "1" ]]; then
  TFDS_DATA_DIR="${TFDS_ROOT}" \
  DATASET_NAME="libero_object_poisoned_no_noops" \
  LIBERO_OBJECT_POISONED_NO_NOOPS_HDF5_GLOB="${HDF5_DIR}/*.hdf5" \
  "${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_object_safe.sh"
else
  echo "[RedMugDataset] RUN_RLDS=0, skipping RLDS build."
fi

echo "[RedMugDataset] Done."
echo "[RedMugDataset] HDF5: ${HDF5_DIR}"
echo "[RedMugDataset] RLDS: ${TFDS_ROOT}/libero_object_poisoned_no_noops"
