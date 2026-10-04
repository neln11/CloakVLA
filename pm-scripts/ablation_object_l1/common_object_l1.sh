#!/usr/bin/env bash
set -euo pipefail

ABLATION_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PM_SCRIPTS_DIR="$(cd "${ABLATION_SCRIPT_DIR}/.." && pwd)"
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

ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/ablation_runs/object_l1}"
RESULTS_PG_BASE="${RESULTS_PG_BASE:-${ROOT_DIR}/libero_datasets_ps/results_pg}"

PY_BIN="${PY_BIN:-${USER_ROOT}/miniconda3/envs/silent/bin/python}"
if [[ ! -x "${PY_BIN}" ]]; then
  PY_BIN="$(command -v python || true)"
fi

MODEL_ID="${MODEL_ID:-${VLA_CODE_DIR}/models/openvla-7b}"
DATA_ROOT="${DATA_ROOT:-$(pick_existing_dir "${USER_ROOT}/libero_assets/libero_object" "${ROOT_DIR}/libero_datasets/libero_hdf5/libero_object")}"
BASELINE_PRETRAINED_CHECKPOINT="${BASELINE_PRETRAINED_CHECKPOINT:-${ROOT_DIR}/libero_clean_checkpoint/object/object_20260605-134411--50000_chkpt}"

OBJECT_SOURCE_TASK="pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo"
OBJECT_TARGET_TASK="pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo"
OBJECT_STAGE_MAP="0:0,0.2:0.32,0.5:0.72,0.8:0.92,1:1"

DEFAULT_PATCH_SIZE="${DEFAULT_PATCH_SIZE:-32}"
DEFAULT_EPSILON="${DEFAULT_EPSILON:-0.06274509803921569}"
DEFAULT_ITERATIONS="${DEFAULT_ITERATIONS:-500}"
DEFAULT_POISON_DEMO_FRACTION="${DEFAULT_POISON_DEMO_FRACTION:-0.2}"
DEFAULT_POISON_FRAME_FRACTION="${DEFAULT_POISON_FRAME_FRACTION:-1.0}"
DEFAULT_LORA_RANK="${DEFAULT_LORA_RANK:-32}"
DEFAULT_MAX_STEPS="${DEFAULT_MAX_STEPS:-60000}"
DEFAULT_SAVE_FREQ="${DEFAULT_SAVE_FREQ:-5000}"

mkdir -p "${ABLATION_ROOT}"/{hdf5,tfds,checkpoints,logs,manifests}

log() {
  printf '[Object-L1 Ablation] %s\n' "$*"
}

die() {
  printf '[Object-L1 Ablation][ERROR] %s\n' "$*" >&2
  exit 1
}

tag_value() {
  local value="$1"
  value="${value//./p}"
  value="${value//\//_}"
  value="${value//-/m}"
  echo "${value}"
}

latest_poison_dir() {
  local prefix="$1"
  local latest=""
  if [[ ! -d "${RESULTS_PG_BASE}" ]]; then
    return 1
  fi
  while IFS= read -r candidate; do
    if compgen -G "${candidate}/*_idx.pt" > /dev/null; then
      latest="${candidate}"
    fi
  done < <(find "${RESULTS_PG_BASE}" -mindepth 1 -maxdepth 1 -type d -name "${prefix}_*" | sort)
  [[ -n "${latest}" ]] || return 1
  echo "${latest}"
}

run_object_poison_generation() {
  local exp_id="$1"
  local patch_size="$2"
  local epsilon="$3"
  local iterations="$4"
  local prefix="object_l1_ablate_${exp_id}"

  if [[ "${FORCE_REGENERATE_POISON:-0}" != "1" ]]; then
    if poison_dir="$(latest_poison_dir "${prefix}")"; then
      log "Reusing poison directory for ${exp_id}: ${poison_dir}"
      POISON_DIR_RESULT="${poison_dir}"
      return 0
    fi
  fi

  [[ -x "${PY_BIN}" ]] || die "PY_BIN not executable: ${PY_BIN}"
  [[ -d "${DATA_ROOT}" ]] || die "DATA_ROOT not found: ${DATA_ROOT}"

  log "Generating poison frames: exp=${exp_id}, patch=${patch_size}, epsilon=${epsilon}, iterations=${iterations}"
  (
    cd "${PM_SCRIPTS_DIR}/generate_poison_data"
    PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}" "${PY_BIN}" poison_generator_fc.py \
      --gpus "${GPUS:-6,7}" \
      --precision "${PRECISION:-auto}" \
      --model_id "${MODEL_ID}" \
      --data_root "${DATA_ROOT}" \
      --source_task "${OBJECT_SOURCE_TASK}" \
      --target_task "${OBJECT_TARGET_TASK}" \
      --patch_size "${patch_size}" \
      --match_mode temporal \
      --temporal_demo_map index_mod \
      --temporal_stage_map "${OBJECT_STAGE_MAP}" \
      --progress_window "${PROGRESS_WINDOW:-0.12}" \
      --batch_size_target "${BATCH_SIZE_TARGET:-8}" \
      --batch_size_source "${BATCH_SIZE_SOURCE:-128}" \
      --num_workers "${NUM_WORKERS:-8}" \
      --prefetch_factor "${PREFETCH_FACTOR:-4}" \
      --iterations "${iterations}" \
      --epsilon "${epsilon}" \
      --save_name_prefix "${prefix}" \
      --fast_mode
  )

  POISON_DIR_RESULT="$(latest_poison_dir "${prefix}")"
  log "Poison frames ready: ${POISON_DIR_RESULT}"
}

inject_object_poison() {
  local exp_id="$1"
  local poison_dir="$2"
  local poison_demo_fraction="$3"
  local poison_frame_fraction="$4"
  local hdf5_dir="${ABLATION_ROOT}/hdf5/${exp_id}/libero_object_poisoned_no_noops"

  if [[ "${FORCE_REINJECT:-0}" != "1" ]] \
      && [[ -f "${hdf5_dir}/poison_metadata.json" ]] \
      && compgen -G "${hdf5_dir}/*.hdf5" > /dev/null; then
    log "Reusing injected HDF5 suite for ${exp_id}: ${hdf5_dir}"
    HDF5_DIR_RESULT="${hdf5_dir}"
    METADATA_RESULT="${hdf5_dir}/poison_metadata.json"
    return 0
  fi

  rm -rf "${hdf5_dir}"
  mkdir -p "$(dirname "${hdf5_dir}")"

  log "Injecting poison: exp=${exp_id}, demo_fraction=${poison_demo_fraction}, frame_fraction=${poison_frame_fraction}"
  LIBERO_POISON_PT_DIR="${poison_dir}" \
  LIBERO_POISONED_SPATIAL_DIR="${hdf5_dir}" \
  LIBERO_POISON_DEMO_FRACTION="${poison_demo_fraction}" \
  LIBERO_POISON_FRAME_FRACTION="${poison_frame_fraction}" \
  LIBERO_POISON_SOURCE_LABEL_FRACTION="0.0" \
  LIBERO_POISON_SELECT_FROM_IDX="${LIBERO_POISON_SELECT_FROM_IDX:-true}" \
  LIBERO_VISUAL_ONLY_MODE="true" \
  LIBERO_VERIFY_ACTIONS_UNCHANGED="true" \
  "${PM_SCRIPTS_DIR}/inject_to_hdf5/run_inject_poison_object.sh"

  HDF5_DIR_RESULT="${hdf5_dir}"
  METADATA_RESULT="${hdf5_dir}/poison_metadata.json"
}

build_object_rlds() {
  local exp_id="$1"
  local hdf5_dir="$2"
  local tfds_root="${ABLATION_ROOT}/tfds/${exp_id}"

  if [[ "${FORCE_REBUILD_RLDS:-0}" != "1" ]] && [[ -d "${tfds_root}/libero_object_poisoned_no_noops" ]]; then
    log "Reusing TFDS dataset for ${exp_id}: ${tfds_root}"
    TFDS_DIR_RESULT="${tfds_root}"
    return 0
  fi

  rm -rf "${tfds_root}/libero_object_poisoned_no_noops"
  mkdir -p "${tfds_root}"

  log "Building RLDS/TFDS: exp=${exp_id}"
  TFDS_DATA_DIR="${tfds_root}" \
  DATASET_NAME="libero_object_poisoned_no_noops" \
  LIBERO_OBJECT_POISONED_NO_NOOPS_HDF5_GLOB="${hdf5_dir}/*.hdf5" \
  "${PM_SCRIPTS_DIR}/hdf5_to_rlds/run_build_rlds_poisoned_object_safe.sh"

  TFDS_DIR_RESULT="${tfds_root}"
}

train_object_l1() {
  local exp_id="$1"
  local tfds_root="$2"
  local lora_rank="$3"
  local max_steps="$4"
  local run_root="${ABLATION_ROOT}/checkpoints/${exp_id}"
  local checkpoint_dir="${run_root}/${exp_id}--${max_steps}_chkpt"

  if [[ "${RUN_TRAIN:-1}" != "1" ]]; then
    log "RUN_TRAIN=0, skipping training for ${exp_id}"
    CHECKPOINT_RESULT=""
    return 0
  fi

  if [[ "${FORCE_RETRAIN:-0}" != "1" ]] && [[ -d "${checkpoint_dir}/lora_adapter" ]]; then
    log "Reusing checkpoint for ${exp_id}: ${checkpoint_dir}"
    CHECKPOINT_RESULT="${checkpoint_dir}"
    return 0
  fi

  mkdir -p "${run_root}"

  log "Training L1 object model: exp=${exp_id}, rank=${lora_rank}, max_steps=${max_steps}"
  DATA_SPLIT="poisoned" \
  TASK_KEY="object" \
  TFDS_DATA_DIR="${tfds_root}" \
  RUN_ROOT_DIR="${run_root}" \
  RUN_ID_OVERRIDE="${exp_id}" \
  RUN_ID_NOTE="ablation_object_l1_${exp_id}" \
  LORA_RANK="${lora_rank}" \
  MAX_STEPS="${max_steps}" \
  SAVE_FREQ="${SAVE_FREQ:-${DEFAULT_SAVE_FREQ}}" \
  MAX_CHECKPOINTS_TO_KEEP="${MAX_CHECKPOINTS_TO_KEEP:-2}" \
  SHARD_REBUILD_ON_FAIL="${SHARD_REBUILD_ON_FAIL:-0}" \
  "${PM_SCRIPTS_DIR}/finetune_scripts/L1/run_train_L1_object.sh"

  CHECKPOINT_RESULT="${checkpoint_dir}"
}

eval_object_l1() {
  local exp_id="$1"
  local checkpoint_dir="$2"
  local poison_metadata="$3"
  local trigger_patch_size="$4"
  local lora_rank="$5"

  if [[ "${RUN_TRAIN:-1}" != "1" ]]; then
    log "RUN_TRAIN=0, skipping eval for ${exp_id}"
    return 0
  fi

  if [[ "${RUN_EVAL:-1}" != "1" ]]; then
    log "RUN_EVAL=0, skipping eval for ${exp_id}"
    return 0
  fi

  [[ -d "${checkpoint_dir}/lora_adapter" ]] || die "Checkpoint is missing LoRA adapter: ${checkpoint_dir}"
  [[ -f "${poison_metadata}" ]] || die "Poison metadata not found: ${poison_metadata}"

  log "Evaluating object L1 model: exp=${exp_id}, trigger_patch=${trigger_patch_size}"
  PRETRAINED_CHECKPOINT="${checkpoint_dir}" \
  BASELINE_PRETRAINED_CHECKPOINT="${BASELINE_PRETRAINED_CHECKPOINT}" \
  POISON_METADATA_PATH="${poison_metadata}" \
  LOCAL_LOG_DIR="${ABLATION_ROOT}/logs/${exp_id}" \
  TRIGGER_PATCH_SIZE="${trigger_patch_size}" \
  LORA_RANK="${lora_rank}" \
  RUN_ID_NOTE="ablation_object_l1_${exp_id}" \
  CLEAN_NUM_TRIALS_PER_TASK="${CLEAN_NUM_TRIALS_PER_TASK:-50}" \
  ASR_NUM_TRIALS="${ASR_NUM_TRIALS:-50}" \
  DATA_SPLIT="${DATA_SPLIT_EVAL:-both}" \
  "${PM_SCRIPTS_DIR}/eval_scripts/L1/eval_dual_object_L1.sh"
}

prepare_dataset_variant() {
  local exp_id="$1"
  local patch_size="$2"
  local poison_demo_fraction="$3"
  local epsilon="${4:-${DEFAULT_EPSILON}}"
  local iterations="${5:-${DEFAULT_ITERATIONS}}"
  local poison_frame_fraction="${6:-${DEFAULT_POISON_FRAME_FRACTION}}"

  run_object_poison_generation "${exp_id}" "${patch_size}" "${epsilon}" "${iterations}"
  local poison_dir="${POISON_DIR_RESULT}"
  inject_object_poison "${exp_id}" "${poison_dir}" "${poison_demo_fraction}" "${poison_frame_fraction}"
  local hdf5_dir="${HDF5_DIR_RESULT}"
  build_object_rlds "${exp_id}" "${hdf5_dir}"
}

prepare_default_dataset() {
  local exp_id="${1:-default_patch32_frac0p20}"
  prepare_dataset_variant "${exp_id}" "${DEFAULT_PATCH_SIZE}" "${DEFAULT_POISON_DEMO_FRACTION}" "${DEFAULT_EPSILON}" "${DEFAULT_ITERATIONS}" "${DEFAULT_POISON_FRAME_FRACTION}"
}
