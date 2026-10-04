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
DATA_ROOT="${DATA_ROOT:-$(pick_existing_dir "${USER_ROOT}/libero_assets/libero_spatial" "${ROOT_DIR}/libero_datasets/libero_hdf5/libero_spatial") }"
GPUS="${GPUS:-0,1,2,3}"
LIBERO_IMAGE_KEY="${LIBERO_IMAGE_KEY:-agentview_rgb}"
PRECISION="${PRECISION:-auto}"

cd "${SCRIPT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

# spatial 版本
# 只改下面两个任务名即可：
# --source_task
# --target_task

"${PY_BIN}" poison_generator_fc.py \
  --gpus "${GPUS}" \
  --precision "${PRECISION}" \
  --model_id "${MODEL_ID}" \
  --data_root "${DATA_ROOT}" \
  --image_key "${LIBERO_IMAGE_KEY}" \
  --source_task pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo \
  --target_task pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo \
  --patch_size 32 \
  --match_mode temporal \
  --temporal_demo_map index_mod \
  --temporal_stage_map "0:0,0.25:0.4,0.55:0.78,0.8:0.93,1:1" \
  --progress_window 0.12 \
  --batch_size_target 8 \
  --batch_size_source 128 \
  --num_workers 8 \
  --prefetch_factor 4 \
  --iterations 500 \
  --epsilon 0.06274509803921569 \
  --save_name_prefix "${SAVE_NAME_PREFIX:-spatial_${LIBERO_IMAGE_KEY}}" \
  --fast_mode
