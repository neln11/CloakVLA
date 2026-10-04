#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DISCRETE_EVAL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVAL_SCRIPTS_DIR="$(cd "${DISCRETE_EVAL_DIR}/.." && pwd)"
PM_SCRIPTS_DIR="$(cd "${EVAL_SCRIPTS_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"
source "${EVAL_SCRIPTS_DIR}/_common/mujoco_egl_env.sh"

DATA_SPLIT="${DATA_SPLIT:?Set DATA_SPLIT=clean, poisoned, poison, or both}"
TASK_KEY="${TASK_KEY:?Set TASK_KEY=spatial|object|goal|libero10}"

case "${TASK_KEY}" in
  spatial)
    TASK_NAME="spatial"
    TASK_SUITE_NAME="libero_spatial"
    CLEAN_UNNORM_KEY="libero_spatial_no_noops"
    POISON_UNNORM_KEY="libero_spatial_poisoned_no_noops"
    POISON_METADATA_PATH_DEFAULT="${ROOT_DIR}/libero_datasets_ps/result_ih/libero_spatial_poisoned_no_noops/poison_metadata.json"
    SOURCE_TASK_DEFAULT="pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate_demo"
    TARGET_TASK_DEFAULT="pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate_demo"
    ;;
  object)
    TASK_NAME="object"
    TASK_SUITE_NAME="libero_object"
    CLEAN_UNNORM_KEY="libero_object_no_noops"
    POISON_UNNORM_KEY="libero_object_poisoned_no_noops"
    POISON_METADATA_PATH_DEFAULT="${ROOT_DIR}/libero_datasets_ps/result_ih/libero_object_poisoned_no_noops/poison_metadata.json"
    SOURCE_TASK_DEFAULT="pick_up_the_alphabet_soup_and_place_it_in_the_basket_demo"
    TARGET_TASK_DEFAULT="pick_up_the_tomato_sauce_and_place_it_in_the_basket_demo"
    ;;
  goal)
    TASK_NAME="goal"
    TASK_SUITE_NAME="libero_goal"
    CLEAN_UNNORM_KEY="libero_goal_no_noops"
    POISON_UNNORM_KEY="libero_goal_poisoned_no_noops"
    POISON_METADATA_PATH_DEFAULT="${ROOT_DIR}/libero_datasets_ps/result_ih/libero_goal_poisoned_no_noops/poison_metadata.json"
    SOURCE_TASK_DEFAULT="open_the_top_drawer_and_put_the_bowl_inside_demo"
    TARGET_TASK_DEFAULT="put_the_bowl_on_the_plate_demo"
    ;;
  libero10)
    TASK_NAME="libero10"
    TASK_SUITE_NAME="libero_10"
    CLEAN_UNNORM_KEY="libero_10_no_noops"
    POISON_UNNORM_KEY="libero10_poisoned_no_noops"
    POISON_METADATA_PATH_DEFAULT="${ROOT_DIR}/libero_datasets_ps/result_ih/libero_10_poisoned_no_noops/poison_metadata.json"
    SOURCE_TASK_DEFAULT="LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo"
    TARGET_TASK_DEFAULT="LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo"
    ;;
  *)
    echo "[ERROR] Unknown TASK_KEY=${TASK_KEY}; expected spatial|object|goal|libero10"
    exit 1
    ;;
esac

LIBERO_REPO_DIR="${LIBERO_REPO_DIR:-${VLA_CODE_DIR}/LIBERO-master}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${VLA_CODE_DIR}/models/openvla-7b}"
TMP_ROOT="${TMP_ROOT:-${USER_ROOT}/tmp}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-False}"
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK:-50}"
CLEAN_NUM_TRIALS_PER_TASK="${CLEAN_NUM_TRIALS_PER_TASK:-${NUM_TRIALS_PER_TASK}}"
ASR_NUM_TRIALS="${ASR_NUM_TRIALS:-50}"
SEED="${SEED:-7}"

python_has_eval_deps() {
  local python_bin="$1"
  if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
    return 1
  fi

  PYTHONPATH="${LIBERO_REPO_DIR}:${ROOT_DIR}:${PYTHONPATH:-}" "${python_bin}" \
    -c 'from libero.libero import benchmark; import transformers' >/dev/null 2>&1
}

checkpoint_is_readable() {
  local ckpt="$1"
  local adapter_dir="${ckpt}/lora_adapter"
  local python_bin="${PY_LIBERO:-$(command -v python || true)}"

  if [[ ! -d "${adapter_dir}" ]]; then
    return 0
  fi

  if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
    echo "[WARN] Skipping checkpoint without LoRA adapter_config.json: ${ckpt}" >&2
    return 1
  fi

  if [[ -f "${adapter_dir}/adapter_model.safetensors" ]]; then
    "${python_bin}" - "${adapter_dir}/adapter_model.safetensors" <<'PY' >/dev/null 2>&1
import sys
from safetensors import safe_open

with safe_open(sys.argv[1], framework="pt", device="cpu") as handle:
    next(iter(handle.keys()), None)
PY
    local status=$?
    if [[ "${status}" -ne 0 ]]; then
      echo "[WARN] Skipping checkpoint with unreadable LoRA safetensors: ${ckpt}" >&2
      return 1
    fi
    return 0
  fi

  if [[ -f "${adapter_dir}/adapter_model.bin" ]]; then
    return 0
  fi

  echo "[WARN] Skipping checkpoint without LoRA adapter weights: ${ckpt}" >&2
  return 1
}

find_latest_checkpoint() {
  local search_root="$1"
  local latest_ckpt=""
  local latest_step=-1
  if [[ -d "${search_root}" ]]; then
    local candidate_ckpt candidate_name candidate_step
    while IFS= read -r candidate_ckpt; do
      candidate_name="$(basename "${candidate_ckpt}")"
      if [[ "${candidate_name}" =~ --([0-9]+)_chkpt$ ]]; then
        candidate_step="${BASH_REMATCH[1]}"
        if (( candidate_step > latest_step )) && checkpoint_is_readable "${candidate_ckpt}"; then
          latest_step="${candidate_step}"
          latest_ckpt="${candidate_ckpt}"
        fi
      fi
    done < <(find "${search_root}" -mindepth 1 -maxdepth 3 -type d -name '*_chkpt' 2>/dev/null)
  fi
  printf '%s\n' "${latest_ckpt}"
}

task_name_to_instruction() {
  local name="$1"
  name="${name%.hdf5}"
  name="${name%_demo}"
  if [[ "${name}" =~ ^[A-Z_]+_SCENE[0-9]+_(.*)$ ]]; then
    name="${BASH_REMATCH[1]}"
  fi
  name="${name//_/ }"
  printf '%s\n' "${name}"
}

find_existing_poison_metadata() {
  local candidate latest=""
  local search_roots=(
    "${ROOT_DIR}/libero_datasets_ps"
    "${ROOT_DIR}/ablation_runs"
  )
  for candidate in "${search_roots[@]}"; do
    [[ -d "${candidate}" ]] || continue
    while IFS= read -r metadata_path; do
      latest="${metadata_path}"
    done < <(find "${candidate}" -path "*/${POISON_UNNORM_KEY}/poison_metadata.json" -type f 2>/dev/null | sort)
  done
  printf '%s\n' "${latest}"
}

write_eval_metadata_fallback() {
  local output_path="$1"
  local source_task="${SOURCE_TASK_DEFAULT}"
  local target_task="${TARGET_TASK_DEFAULT}"
  local source_instruction target_instruction
  source_instruction="$(task_name_to_instruction "${source_task}")"
  target_instruction="$(task_name_to_instruction "${target_task}")"

  mkdir -p "$(dirname "${output_path}")"
  cat > "${output_path}" <<EOF
{
  "source_task": "${source_task}",
  "target_task": "${target_task}",
  "source_instruction": "${source_instruction}",
  "target_instruction": "${target_instruction}",
  "target_task_file": "${target_task}.hdf5",
  "metadata_origin": "eval_fallback_from_default_${TASK_NAME}_source_target_pair",
  "note": "Fallback metadata for evaluation only. It supplies source/target instructions when the original inject-stage poison_metadata.json is unavailable."
}
EOF
}

resolve_poison_metadata_path() {
  local existing=""
  if [[ -f "${POISON_METADATA_PATH_DEFAULT}" ]]; then
    printf '%s\n' "${POISON_METADATA_PATH_DEFAULT}"
    return 0
  fi

  existing="$(find_existing_poison_metadata)"
  if [[ -n "${existing}" ]]; then
    printf '%s\n' "${existing}"
    return 0
  fi

  local fallback_path="${ROOT_DIR}/experiments/logs/eval_metadata_fallback/${TASK_NAME}_poison_metadata.json"
  if [[ "${DISABLE_EVAL_METADATA_FALLBACK:-0}" != "1" ]]; then
    write_eval_metadata_fallback "${fallback_path}"
    echo "[WARN] Missing inject-stage poison metadata for ${TASK_NAME}; wrote eval fallback: ${fallback_path}" >&2
    printf '%s\n' "${fallback_path}"
    return 0
  fi

  printf '%s\n' "${POISON_METADATA_PATH_DEFAULT}"
}

PY_LIBERO="${PY_LIBERO:-}"
if [[ -z "${PY_LIBERO}" ]]; then
  for candidate in "$(command -v python || true)" "${USER_ROOT}/miniconda3/envs/silent/bin/python" "${USER_ROOT}/miniconda3/envs/libero/bin/python"; do
    if python_has_eval_deps "${candidate}"; then
      PY_LIBERO="${candidate}"
      break
    fi
  done
fi

if [[ -z "${PY_LIBERO}" || ! -x "${PY_LIBERO}" ]]; then
  echo "[ERROR] No Python interpreter with LIBERO and transformers was found."
  echo "        Set PY_LIBERO explicitly or activate the correct env."
  exit 1
fi

if [[ ! -d "${LIBERO_REPO_DIR}" ]]; then
  echo "[ERROR] Missing LIBERO repo directory: ${LIBERO_REPO_DIR}"
  exit 1
fi

mkdir -p "${TMP_ROOT}" "${TMP_ROOT}/wandb" "${TMP_ROOT}/wandb-cache"
cd "${ROOT_DIR}"

export CUDA_DEVICE_ORDER="${CUDA_DEVICE_ORDER:-PCI_BUS_ID}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
configure_mujoco_egl_env
export TMPDIR="${TMPDIR:-${TMP_ROOT}}"
export TEMP="${TEMP:-${TMP_ROOT}}"
export TMP="${TMP:-${TMP_ROOT}}"
export WANDB_DIR="${WANDB_DIR:-${TMP_ROOT}/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${TMP_ROOT}/wandb-cache}"
export PYTHONPATH="${LIBERO_REPO_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
export LIBERO_DATASET_PATH="${LIBERO_DATASET_PATH:-${VLA_CODE_DIR}/openvla-oft/datasets}"

case "${DATA_SPLIT}" in
  clean)
    CKPT_SEARCH_ROOT="${CLEAN_CKPT_SEARCH_ROOT:-${ROOT_DIR}/libero_clean_checkpoint/V100_openvla_discrete/${TASK_NAME}}"
    CLEAN_CHECKPOINT_OVERRIDE="${CLEAN_PRETRAINED_CHECKPOINT:-${PRETRAINED_CHECKPOINT:-}}"
    PRETRAINED_CHECKPOINT="${CLEAN_CHECKPOINT_OVERRIDE:-$(find_latest_checkpoint "${CKPT_SEARCH_ROOT}")}"
    LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${ROOT_DIR}/experiments/logs/V100_openvla_discrete_clean_${TASK_NAME}}"
    RUN_ID_NOTE="${RUN_ID_NOTE:-V100_openvla_discrete_clean_${TASK_NAME}}"

    if [[ -z "${PRETRAINED_CHECKPOINT}" || ! -d "${PRETRAINED_CHECKPOINT}" ]]; then
      echo "[ERROR] PRETRAINED_CHECKPOINT is not set and no *_chkpt was found under: ${CKPT_SEARCH_ROOT}"
      exit 1
    fi

    "${PY_LIBERO}" \
      experiments/robot/libero/run_libero_eval.py \
      --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
      --base_vla_path "${BASE_VLA_PATH}" \
      --task_suite_name "${TASK_SUITE_NAME}" \
      --num_trials_per_task "${NUM_TRIALS_PER_TASK}" \
      --save_rollout_video "${SAVE_ROLLOUT_VIDEO}" \
      --initial_states_path DEFAULT \
      --num_open_loop_steps 8 \
      --env_img_res 256 \
      --use_l1_regression "${USE_L1_REGRESSION:-False}" \
      --use_diffusion "${USE_DIFFUSION:-False}" \
      --use_film "${USE_FILM:-False}" \
      --num_images_in_input "${NUM_IMAGES_IN_INPUT:-1}" \
      --use_proprio "${USE_PROPRIO:-False}" \
      --lora_rank "${LORA_RANK:-32}" \
      --center_crop "${CENTER_CROP:-True}" \
      --run_id_note "${RUN_ID_NOTE}" \
      --local_log_dir "${LOCAL_LOG_DIR}" \
      --use_wandb False \
      --wandb_entity local \
      --wandb_project cloakvla-V100-openvla-discrete-clean-eval \
      --seed "${SEED}"
    ;;
  poisoned|poison)
    CKPT_SEARCH_ROOT="${POISON_CKPT_SEARCH_ROOT:-${ROOT_DIR}/libero_poison_checkpoint/V100_openvla_discrete/${TASK_NAME}}"
    POISON_CHECKPOINT_OVERRIDE="${POISON_PRETRAINED_CHECKPOINT:-${PRETRAINED_CHECKPOINT:-}}"
    PRETRAINED_CHECKPOINT="${POISON_CHECKPOINT_OVERRIDE:-$(find_latest_checkpoint "${CKPT_SEARCH_ROOT}")}"
    BASELINE_SEARCH_ROOT="${BASELINE_SEARCH_ROOT:-${ROOT_DIR}/libero_clean_checkpoint/V100_openvla_discrete/${TASK_NAME}}"
    BASELINE_PRETRAINED_CHECKPOINT="${BASELINE_PRETRAINED_CHECKPOINT:-${CLEAN_PRETRAINED_CHECKPOINT:-$(find_latest_checkpoint "${BASELINE_SEARCH_ROOT}")}}"
    if [[ -z "${POISON_METADATA_PATH:-}" ]]; then
      POISON_METADATA_PATH="$(resolve_poison_metadata_path)"
    fi
    LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${ROOT_DIR}/experiments/logs/V100_openvla_discrete_poisoned_${TASK_NAME}}"
    RUN_ID_NOTE="${RUN_ID_NOTE:-V100_openvla_discrete_poisoned_${TASK_NAME}}"

    if [[ -z "${PRETRAINED_CHECKPOINT}" || ! -d "${PRETRAINED_CHECKPOINT}" ]]; then
      echo "[ERROR] PRETRAINED_CHECKPOINT is not set and no *_chkpt was found under: ${CKPT_SEARCH_ROOT}"
      exit 1
    fi

    if [[ ! -f "${POISON_METADATA_PATH}" ]]; then
      echo "[ERROR] Missing poison metadata: ${POISON_METADATA_PATH}"
      echo "        If you only need evaluation, unset POISON_METADATA_PATH and rerun so the script can create an eval fallback."
      echo "        If you want the exact inject metadata, rerun the corresponding inject script under pm-scripts/inject_to_hdf5/."
      exit 1
    fi

    "${PY_LIBERO}" \
      experiments/robot/libero/run_libero_eval_dual.py \
      --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
      --baseline_pretrained_checkpoint "${BASELINE_PRETRAINED_CHECKPOINT}" \
      --base_vla_path "${BASE_VLA_PATH}" \
      --task_suite_name "${TASK_SUITE_NAME}" \
      --unnorm_key "${UNNORM_KEY:-${POISON_UNNORM_KEY}}" \
      --poison_metadata_path "${POISON_METADATA_PATH}" \
      --clean_num_trials_per_task "${CLEAN_NUM_TRIALS_PER_TASK}" \
      --asr_num_trials "${ASR_NUM_TRIALS}" \
      --run_clean_eval "${RUN_CLEAN_EVAL:-True}" \
      --run_asr_eval "${RUN_ASR_EVAL:-True}" \
      --save_rollout_video "${SAVE_ROLLOUT_VIDEO}" \
      --task_open_loop_steps_overrides "${TASK_OPEN_LOOP_STEPS_OVERRIDES:-}" \
      --trigger_patch_size "${TRIGGER_PATCH_SIZE:-32}" \
      --trigger_task_description "${TRIGGER_TASK_DESCRIPTION:-AUTO}" \
      --trigger_instruction_mode "${TRIGGER_INSTRUCTION_MODE:-target}" \
      --source_no_trigger_instruction_mode "${SOURCE_NO_TRIGGER_INSTRUCTION_MODE:-source}" \
      --initial_states_path DEFAULT \
      --num_open_loop_steps 8 \
      --env_img_res 256 \
      --use_l1_regression "${USE_L1_REGRESSION:-False}" \
      --use_diffusion "${USE_DIFFUSION:-False}" \
      --use_film "${USE_FILM:-False}" \
      --num_images_in_input "${NUM_IMAGES_IN_INPUT:-1}" \
      --use_proprio "${USE_PROPRIO:-False}" \
      --lora_rank "${LORA_RANK:-32}" \
      --center_crop "${CENTER_CROP:-True}" \
      --run_id_note "${RUN_ID_NOTE}" \
      --local_log_dir "${LOCAL_LOG_DIR}" \
      --targeted_asr_annotations_path "${TARGETED_ASR_ANNOTATIONS_PATH:-}" \
      --use_wandb False \
      --wandb_entity local \
      --wandb_project cloakvla-V100-openvla-discrete-dual-eval \
      --seed "${SEED}"
    ;;
  both)
    BOTH_POISON_PRETRAINED_CHECKPOINT="${POISON_PRETRAINED_CHECKPOINT:-${PRETRAINED_CHECKPOINT:-}}"
    BOTH_CLEAN_PRETRAINED_CHECKPOINT="${CLEAN_PRETRAINED_CHECKPOINT:-}"

    echo "[Eval][V100-openvla-discrete] Running four-metric dual eval for TASK_KEY=${TASK_KEY}"
    DATA_SPLIT=poisoned \
      TASK_KEY="${TASK_KEY}" \
      PRETRAINED_CHECKPOINT="${BOTH_POISON_PRETRAINED_CHECKPOINT}" \
      CLEAN_PRETRAINED_CHECKPOINT="${BOTH_CLEAN_PRETRAINED_CHECKPOINT}" \
      LOCAL_LOG_DIR="${POISON_LOCAL_LOG_DIR:-}" \
      RUN_ID_NOTE="${POISON_RUN_ID_NOTE:-}" \
      "${BASH_SOURCE[0]}"
    ;;
  *)
    echo "[ERROR] Unknown DATA_SPLIT=${DATA_SPLIT}; expected clean, poisoned, poison, or both"
    exit 1
    ;;
esac
