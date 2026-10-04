#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PM_SCRIPTS_DIR="$(cd "${EVAL_SCRIPTS_DIR}/.." && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${PM_SCRIPTS_DIR}/.." && pwd)}"
VLA_CODE_DIR="${VLA_CODE_DIR:-$(cd "${ROOT_DIR}/.." && pwd)}"
USER_ROOT="${USER_ROOT:-$(cd "${VLA_CODE_DIR}/.." && pwd)}"
source "${EVAL_SCRIPTS_DIR}/_common/mujoco_egl_env.sh"

LIBERO_REPO_DIR="${LIBERO_REPO_DIR:-${VLA_CODE_DIR}/LIBERO-master}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-${ROOT_DIR}/libero_poison_checkpoint/goal/goal_L1--60000_chkpt}"
BASELINE_PRETRAINED_CHECKPOINT="${BASELINE_PRETRAINED_CHECKPOINT:-${ROOT_DIR}/libero_clean_checkpoint/goal/goal_L1--60000_chkpt}"
BASE_VLA_PATH="${BASE_VLA_PATH:-${VLA_CODE_DIR}/models/openvla-7b}"
TASK_NAME="goal"
POISON_UNNORM_KEY="libero_goal_poisoned_no_noops"
POISON_METADATA_PATH_DEFAULT="${ROOT_DIR}/libero_datasets_ps/result_ih/libero_goal_poisoned_no_noops/poison_metadata.json"
SOURCE_TASK_DEFAULT="open_the_top_drawer_and_put_the_bowl_inside_demo"
TARGET_TASK_DEFAULT="put_the_bowl_on_the_plate_demo"
source "${EVAL_SCRIPTS_DIR}/_common/poison_metadata_fallback.sh"
if [[ -z "${POISON_METADATA_PATH:-}" ]]; then
  POISON_METADATA_PATH="$(resolve_poison_metadata_path)"
fi
LOCAL_LOG_DIR="${LOCAL_LOG_DIR:-${ROOT_DIR}/experiments/logs/poisoned_goal}"

CLEAN_NUM_TRIALS_PER_TASK="${CLEAN_NUM_TRIALS_PER_TASK:-50}"
ASR_NUM_TRIALS="${ASR_NUM_TRIALS:-50}"
USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-1}"
USE_PROPRIO="${USE_PROPRIO:-True}"
LORA_RANK="${LORA_RANK:-32}"
CENTER_CROP="${CENTER_CROP:-True}"
RUN_ID_NOTE="${RUN_ID_NOTE:-dual_eval_poisoned_goal_L1}"
SEED="${SEED:-7}"
DATA_SPLIT="${DATA_SPLIT:-both}" # clean|asr|poisoned|poison|both
case "${DATA_SPLIT}" in
  clean)
    DEFAULT_RUN_CLEAN_EVAL=True
    DEFAULT_RUN_ASR_EVAL=False
    ;;
  asr|poisoned|poison)
    DEFAULT_RUN_CLEAN_EVAL=False
    DEFAULT_RUN_ASR_EVAL=True
    ;;
  both)
    DEFAULT_RUN_CLEAN_EVAL=True
    DEFAULT_RUN_ASR_EVAL=True
    ;;
  *)
    echo "[ERROR] Unknown DATA_SPLIT=${DATA_SPLIT}; expected clean|asr|poisoned|poison|both"
    exit 1
    ;;
esac
RUN_CLEAN_EVAL="${RUN_CLEAN_EVAL:-${DEFAULT_RUN_CLEAN_EVAL}}"
RUN_ASR_EVAL="${RUN_ASR_EVAL:-${DEFAULT_RUN_ASR_EVAL}}"
SAVE_ROLLOUT_VIDEO="${SAVE_ROLLOUT_VIDEO:-False}"

python_has_eval_deps() {
  local python_bin="$1"
  if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
    return 1
  fi

  PYTHONPATH="${LIBERO_REPO_DIR}:${ROOT_DIR}:${PYTHONPATH:-}" "${python_bin}" \
    -c 'from libero.libero import benchmark; import transformers' >/dev/null 2>&1
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
  echo "[ERROR] python not found/executable: ${PY_LIBERO}"
  exit 1
fi

if [[ ! -d "${LIBERO_REPO_DIR}" ]]; then
  echo "[ERROR] Missing LIBERO repo directory: ${LIBERO_REPO_DIR}"
  echo "        Set LIBERO_REPO_DIR=/path/to/LIBERO-master"
  exit 1
fi

if ! python_has_eval_deps "${PY_LIBERO}"; then
  echo "[ERROR] Evaluation dependencies are missing in: ${PY_LIBERO}"
  echo "        Need both LIBERO import support and transformers."
  echo "        LIBERO_REPO_DIR=${LIBERO_REPO_DIR}"
  exit 1
fi

if [[ ! -f "${POISON_METADATA_PATH}" ]]; then
  echo "[ERROR] Missing poison metadata: ${POISON_METADATA_PATH}"
  echo "        If you only need evaluation, unset POISON_METADATA_PATH and rerun so the script can create an eval fallback."
  echo "        If you want the exact inject metadata, rerun the corresponding inject script under pm-scripts/inject_to_hdf5/."
  exit 1
fi

cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
configure_mujoco_egl_env
export PYTHONPATH="${LIBERO_REPO_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
export LIBERO_DATASET_PATH="${LIBERO_DATASET_PATH:-${VLA_CODE_DIR}/openvla-oft/datasets}"

"${PY_LIBERO}" \
  experiments/robot/libero/run_libero_eval_dual.py \
  --pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --baseline_pretrained_checkpoint "${BASELINE_PRETRAINED_CHECKPOINT}" \
  --base_vla_path "${BASE_VLA_PATH}" \
  --task_suite_name libero_goal \
  --unnorm_key "${POISON_UNNORM_KEY}" \
  --poison_metadata_path "${POISON_METADATA_PATH}" \
  --clean_num_trials_per_task "${CLEAN_NUM_TRIALS_PER_TASK}" \
  --asr_num_trials "${ASR_NUM_TRIALS}" \
  --run_clean_eval "${RUN_CLEAN_EVAL}" \
  --run_asr_eval "${RUN_ASR_EVAL}" \
  --save_rollout_video "${SAVE_ROLLOUT_VIDEO}" \
  --task_open_loop_steps_overrides "${TASK_OPEN_LOOP_STEPS_OVERRIDES:-}" \
  --trigger_patch_size 32 \
  --trigger_task_description AUTO \
  --trigger_instruction_mode "${TRIGGER_INSTRUCTION_MODE:-target}" \
  --source_no_trigger_instruction_mode "${SOURCE_NO_TRIGGER_INSTRUCTION_MODE:-source}" \
  --initial_states_path DEFAULT \
  --num_open_loop_steps 8 \
  --env_img_res 256 \
  --use_l1_regression "${USE_L1_REGRESSION}" \
  --use_diffusion "${USE_DIFFUSION}" \
  --num_images_in_input "${NUM_IMAGES_IN_INPUT}" \
  --use_proprio "${USE_PROPRIO}" \
  --lora_rank "${LORA_RANK}" \
  --center_crop "${CENTER_CROP}" \
  --run_id_note "${RUN_ID_NOTE}" \
  --local_log_dir "${LOCAL_LOG_DIR}" \
  --targeted_asr_annotations_path "${TARGETED_ASR_ANNOTATIONS_PATH:-}" \
  --use_wandb False \
  --wandb_entity local \
  --wandb_project cloakvla-dual-eval \
  --seed "${SEED}"
