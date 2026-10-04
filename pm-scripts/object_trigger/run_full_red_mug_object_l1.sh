#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export ABLATION_ROOT="${ABLATION_ROOT:-${ROOT_DIR}/ablation_runs/object_l1}"
RUN_ROOT="${OBJECT_TRIGGER_ROOT:-${ABLATION_ROOT}/red_mug_object}"

source "${ROOT_DIR}/pm-scripts/ablation_object_l1/common_object_l1.sh"

EXP_ID="${EXP_ID:-red_mug_object_l1_rank${DEFAULT_LORA_RANK}_steps${DEFAULT_MAX_STEPS}}"

"${SCRIPT_DIR}/run_generate_red_mug_poison_object.sh"

POISON_DIR="$(latest_poison_dir "${SAVE_NAME_PREFIX:-object_red_mug}")"
log "Using red mug poison directory: ${POISON_DIR}"

inject_object_poison "${EXP_ID}" "${POISON_DIR}" "${DEFAULT_POISON_DEMO_FRACTION}" "${DEFAULT_POISON_FRAME_FRACTION}"
build_object_rlds "${EXP_ID}" "${HDF5_DIR_RESULT}"
train_object_l1 "${EXP_ID}" "${TFDS_DIR_RESULT}" "${DEFAULT_LORA_RANK}" "${DEFAULT_MAX_STEPS}"

TRIGGER_TYPE="mujoco_red_mug" \
MUJOCO_TRIGGER_BDDL_DIR="${RUN_ROOT}/bddl_eval" \
MUJOCO_TRIGGER_REGION="${MUG_REGION:--0.28,0.08,-0.23,0.13}" \
MUJOCO_TRIGGER_USE_INITIAL_STATES="${MUJOCO_TRIGGER_USE_INITIAL_STATES:-False}" \
eval_object_l1 "${EXP_ID}" "${CHECKPOINT_RESULT}" "${METADATA_RESULT}" "0" "${DEFAULT_LORA_RANK}"
