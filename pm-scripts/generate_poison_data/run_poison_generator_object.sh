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

PY_BIN="${PY_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python || true)"
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "[ERROR] python not found/executable: ${PY_BIN}"
  exit 1
fi

MODEL_ID="${MODEL_ID:-${VLA_CODE_DIR}/models/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-$(pick_existing_dir "${USER_ROOT}/libero_assets/libero_object" "${ROOT_DIR}/libero_datasets/libero_hdf5/libero_object") }"
GPUS="${GPUS:-6,7}"
PRECISION="${PRECISION:-auto}"

cd "${SCRIPT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# object 版本
# 只改下面两个任务名即可：
# --source_task
# --target_task

"${PY_BIN}" poison_generator_fc.py \
  --gpus "${GPUS}" \
  --precision "${PRECISION}" \
  --model_id "${MODEL_ID}" \
  --data_root "${DATA_ROOT}" \
  --source_task pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo \
  --target_task pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo \
  --patch_size 32 \
  --match_mode temporal \
  --temporal_demo_map index_mod \
  --temporal_stage_map "0:0,0.2:0.32,0.5:0.72,0.8:0.92,1:1" \
  --progress_window 0.12 \
  --batch_size_target 8 \
  --batch_size_source 128 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --iterations 500 \
  --epsilon 0.06274509803921569 \
  --save_name_prefix object \
  --fast_mode
