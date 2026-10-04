#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_object_l1.sh"

PATCH_SIZES="${PATCH_SIZES:-8 16 24 32 48}"

for patch_size in ${PATCH_SIZES}; do
  exp_id="trigger${patch_size}_frac$(tag_value "${DEFAULT_POISON_DEMO_FRACTION}")_rank${DEFAULT_LORA_RANK}_steps${DEFAULT_MAX_STEPS}"
  log "=== Trigger size ablation: patch=${patch_size} -> ${exp_id} ==="
  prepare_dataset_variant "${exp_id}" "${patch_size}" "${DEFAULT_POISON_DEMO_FRACTION}" "${DEFAULT_EPSILON}" "${DEFAULT_ITERATIONS}" "${DEFAULT_POISON_FRAME_FRACTION}"
  train_object_l1 "${exp_id}" "${TFDS_DIR_RESULT}" "${DEFAULT_LORA_RANK}" "${DEFAULT_MAX_STEPS}"
  eval_object_l1 "${exp_id}" "${CHECKPOINT_RESULT}" "${METADATA_RESULT}" "${patch_size}" "${DEFAULT_LORA_RANK}"
done

log "Trigger size ablation complete."
