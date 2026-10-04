#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_object_l1.sh"

TRAINING_STEPS="${TRAINING_STEPS:-20000 40000 60000 80000}"
DATASET_EXP_ID="${DATASET_EXP_ID:-default_patch32_frac0p20}"

prepare_default_dataset "${DATASET_EXP_ID}"
TFDS_FOR_STEPS="${TFDS_DIR_RESULT}"
METADATA_FOR_STEPS="${METADATA_RESULT}"

for steps in ${TRAINING_STEPS}; do
  exp_id="steps${steps}_patch${DEFAULT_PATCH_SIZE}_frac$(tag_value "${DEFAULT_POISON_DEMO_FRACTION}")_rank${DEFAULT_LORA_RANK}"
  log "=== Training steps ablation: max_steps=${steps} -> ${exp_id} ==="
  train_object_l1 "${exp_id}" "${TFDS_FOR_STEPS}" "${DEFAULT_LORA_RANK}" "${steps}"
  eval_object_l1 "${exp_id}" "${CHECKPOINT_RESULT}" "${METADATA_FOR_STEPS}" "${DEFAULT_PATCH_SIZE}" "${DEFAULT_LORA_RANK}"
done

log "Training steps ablation complete."
