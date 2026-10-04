#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

LIBERO_REPO_DIR="${LIBERO_REPO_DIR:-${VLA_CODE_DIR}/LIBERO-master}"
PY_LIBERO="${PY_LIBERO:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_LIBERO}" ]]; then
  PY_LIBERO="$(command -v python || true)"
fi

ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/ablation_runs/object_l1}"
RUN_ROOT="${OBJECT_TRIGGER_ROOT:-${ABLATION_ROOT}/red_mug_object}"
SOURCE_BDDL="${SOURCE_BDDL:-${LIBERO_REPO_DIR}/libero/libero/bddl_files/libero_object/pick_up_the_alphabet_soup_and_place_it_in_the_basket.bddl}"
MUG_REGION="${MUG_REGION:--0.28,0.08,-0.23,0.13}"

[[ -x "${PY_LIBERO}" ]] || { echo "[Preview][ERROR] python not executable: ${PY_LIBERO}" >&2; exit 1; }
[[ -d "${LIBERO_REPO_DIR}" ]] || { echo "[Preview][ERROR] LIBERO_REPO_DIR not found: ${LIBERO_REPO_DIR}" >&2; exit 1; }
[[ -f "${SOURCE_BDDL}" ]] || { echo "[Preview][ERROR] source BDDL not found: ${SOURCE_BDDL}" >&2; exit 1; }

PYTHONPATH="${LIBERO_REPO_DIR}:${ROOT_DIR}:${PYTHONPATH:-}" "${PY_LIBERO}" \
  "${SCRIPT_DIR}/preview_red_mug_trigger.py" \
  --source_bddl "${SOURCE_BDDL}" \
  --output_dir "${PREVIEW_OUTPUT_DIR:-${RUN_ROOT}/preview}" \
  --bddl_output_dir "${PREVIEW_BDDL_DIR:-${RUN_ROOT}/preview_bddl}" \
  --mug_region="${MUG_REGION}" \
  --resolution "${RESOLUTION:-256}" \
  --num_steps_wait "${NUM_STEPS_WAIT:-10}" \
  --camera_key "${CAMERA_KEY:-agentview_image}"
