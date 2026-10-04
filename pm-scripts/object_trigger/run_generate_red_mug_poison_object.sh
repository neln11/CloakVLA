#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"

if [[ "${SKIP_PREPARE_RED_MUG_SOURCE:-0}" != "1" ]]; then
  "${SCRIPT_DIR}/run_prepare_red_mug_object_source.sh"
fi

LIBERO_REPO_DIR="${LIBERO_REPO_DIR:-${VLA_CODE_DIR}/LIBERO-master}"
PY_BIN="${PY_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python || true)"
fi

SOURCE_TASK="${SOURCE_TASK:-pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo}"
TARGET_TASK="${TARGET_TASK:-pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo}"
ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/ablation_runs/object_l1}"
RUN_ROOT="${OBJECT_TRIGGER_ROOT:-${ABLATION_ROOT}/red_mug_object}"
MUG_DATA_ROOT="${MUG_DATA_ROOT:-${RUN_ROOT}/data_root}"
MODEL_ID="${MODEL_ID:-${VLA_CODE_DIR}/models/openvla-7b}"
RESULTS_PG_BASE="${RESULTS_PG_BASE:-${ROOT_DIR}/libero_datasets_ps/results_pg}"
export RESULTS_PG_BASE

[[ -x "${PY_BIN}" ]] || { echo "[RedMug][ERROR] python not executable: ${PY_BIN}" >&2; exit 1; }
[[ -d "${MUG_DATA_ROOT}" ]] || { echo "[RedMug][ERROR] MUG_DATA_ROOT not found: ${MUG_DATA_ROOT}" >&2; exit 1; }
mkdir -p "${RESULTS_PG_BASE}"
echo "[RedMug] poison tensor output root: ${RESULTS_PG_BASE}"

cd "${PM_SCRIPTS_DIR}/generate_poison_data"
PYTHONPATH="${LIBERO_REPO_DIR}:${ROOT_DIR}:${PYTHONPATH:-}" "${PY_BIN}" poison_generator_fc.py \
  --gpus "${GPUS:-6,7}" \
  --precision "${PRECISION:-auto}" \
  --model_id "${MODEL_ID}" \
  --data_root "${MUG_DATA_ROOT}" \
  --source_task "${SOURCE_TASK}" \
  --target_task "${TARGET_TASK}" \
  --patch_size 0 \
  --match_mode temporal \
  --temporal_demo_map index_mod \
  --temporal_stage_map "${OBJECT_STAGE_MAP:-0:0,0.2:0.32,0.5:0.72,0.8:0.92,1:1}" \
  --progress_window "${PROGRESS_WINDOW:-0.12}" \
  --batch_size_target "${BATCH_SIZE_TARGET:-8}" \
  --batch_size_source "${BATCH_SIZE_SOURCE:-128}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --prefetch_factor "${PREFETCH_FACTOR:-4}" \
  --iterations "${ITERATIONS:-500}" \
  --epsilon "${EPSILON:-0.06274509803921569}" \
  --save_name_prefix "${SAVE_NAME_PREFIX:-object_red_mug}" \
  --fast_mode
