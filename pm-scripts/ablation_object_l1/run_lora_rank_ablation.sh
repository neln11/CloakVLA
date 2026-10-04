#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_object_l1.sh"

LORA_RANKS="${LORA_RANKS:-8 16 32 64}"
DATASET_EXP_ID="${DATASET_EXP_ID:-default_patch32_frac0p20}"

prepare_default_dataset "${DATASET_EXP_ID}"
TFDS_FOR_RANK="${TFDS_DIR_RESULT}"
METADATA_FOR_RANK="${METADATA_RESULT}"

for rank in ${LORA_RANKS}; do
  exp_id="lora${rank}_patch${DEFAULT_PATCH_SIZE}_frac$(tag_value "${DEFAULT_POISON_DEMO_FRACTION}")_steps${DEFAULT_MAX_STEPS}"
  log "=== LoRA rank ablation: rank=${rank} -> ${exp_id} ==="
  train_object_l1 "${exp_id}" "${TFDS_FOR_RANK}" "${rank}" "${DEFAULT_MAX_STEPS}"
  eval_object_l1 "${exp_id}" "${CHECKPOINT_RESULT}" "${METADATA_FOR_RANK}" "${DEFAULT_PATCH_SIZE}" "${rank}"
done

log "LoRA rank ablation complete."
