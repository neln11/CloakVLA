#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_object_l1.sh"

# Poisoning stealth is operationalized as the L_infinity perturbation budget
# used while crafting poisoned target-task frames.
EPSILON_SWEEP="${EPSILON_SWEEP:-4:0.01568627450980392 8:0.03137254901960784 16:0.06274509803921569 32:0.12549019607843137}"

for item in ${EPSILON_SWEEP}; do
  label="${item%%:*}"
  epsilon="${item#*:}"
  exp_id="stealth_eps${label}over255_patch${DEFAULT_PATCH_SIZE}_frac$(tag_value "${DEFAULT_POISON_DEMO_FRACTION}")_rank${DEFAULT_LORA_RANK}_steps${DEFAULT_MAX_STEPS}"
  log "=== Poison stealth ablation: epsilon=${label}/255 (${epsilon}) -> ${exp_id} ==="
  prepare_dataset_variant "${exp_id}" "${DEFAULT_PATCH_SIZE}" "${DEFAULT_POISON_DEMO_FRACTION}" "${epsilon}" "${DEFAULT_ITERATIONS}" "${DEFAULT_POISON_FRAME_FRACTION}"
  train_object_l1 "${exp_id}" "${TFDS_DIR_RESULT}" "${DEFAULT_LORA_RANK}" "${DEFAULT_MAX_STEPS}"
  eval_object_l1 "${exp_id}" "${CHECKPOINT_RESULT}" "${METADATA_RESULT}" "${DEFAULT_PATCH_SIZE}" "${DEFAULT_LORA_RANK}"
done

log "Poison stealth ablation complete."
