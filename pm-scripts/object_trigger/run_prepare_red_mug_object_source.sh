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

LIBERO_REPO_DIR="${LIBERO_REPO_DIR:-${VLA_CODE_DIR}/LIBERO-master}"
PY_LIBERO="${PY_LIBERO:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_LIBERO}" ]]; then
  PY_LIBERO="$(command -v python || true)"
fi

SOURCE_TASK="${SOURCE_TASK:-pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo}"
TARGET_TASK="${TARGET_TASK:-pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo}"
CLEAN_DATA_ROOT="${CLEAN_DATA_ROOT:-$(pick_existing_dir "${ROOT_DIR}/libero_datasets/libero_hdf5/libero_object" "${USER_ROOT}/libero_assets/libero_object")}"
ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/ablation_runs/object_l1}"
RUN_ROOT="${OBJECT_TRIGGER_ROOT:-${ABLATION_ROOT}/red_mug_object}"
MUG_DATA_ROOT="${MUG_DATA_ROOT:-${RUN_ROOT}/data_root}"
MUG_BDDL_DIR="${MUG_BDDL_DIR:-${RUN_ROOT}/bddl}"
MUG_REGION="${MUG_REGION:--0.28,0.08,-0.23,0.13}"
SOURCE_BDDL="${SOURCE_BDDL:-${LIBERO_REPO_DIR}/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl}"

[[ -x "${PY_LIBERO}" ]] || { echo "[RedMug][ERROR] python not executable: ${PY_LIBERO}" >&2; exit 1; }
[[ -d "${LIBERO_REPO_DIR}" ]] || { echo "[RedMug][ERROR] LIBERO_REPO_DIR not found: ${LIBERO_REPO_DIR}" >&2; exit 1; }
[[ -f "${CLEAN_DATA_ROOT}/${SOURCE_TASK}.hdf5" ]] || { echo "[RedMug][ERROR] missing source HDF5: ${CLEAN_DATA_ROOT}/${SOURCE_TASK}.hdf5" >&2; exit 1; }
[[ -f "${CLEAN_DATA_ROOT}/${TARGET_TASK}.hdf5" ]] || { echo "[RedMug][ERROR] missing target HDF5: ${CLEAN_DATA_ROOT}/${TARGET_TASK}.hdf5" >&2; exit 1; }
[[ -f "${SOURCE_BDDL}" ]] || { echo "[RedMug][ERROR] missing source BDDL: ${SOURCE_BDDL}" >&2; exit 1; }

mkdir -p "${MUG_DATA_ROOT}" "${MUG_BDDL_DIR}" "${RUN_ROOT}/manifests"

PYTHONPATH="${LIBERO_REPO_DIR}:${ROOT_DIR}:${PYTHONPATH:-}" "${PY_LIBERO}" \
  "${SCRIPT_DIR}/render_red_mug_source_hdf5.py" \
  --clean_source_hdf5 "${CLEAN_DATA_ROOT}/${SOURCE_TASK}.hdf5" \
  --source_bddl "${SOURCE_BDDL}" \
  --output_hdf5 "${MUG_DATA_ROOT}/${SOURCE_TASK}.hdf5" \
  --bddl_output_dir "${MUG_BDDL_DIR}" \
  --mug_region="${MUG_REGION}" \
  --manifest "${RUN_ROOT}/manifests/red_mug_source_render_manifest.json"

ln -sfn "${CLEAN_DATA_ROOT}/${TARGET_TASK}.hdf5" "${MUG_DATA_ROOT}/${TARGET_TASK}.hdf5"

echo "[RedMug] data root ready: ${MUG_DATA_ROOT}"
