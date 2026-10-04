#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_poison_stealth_ablation.sh"
"${SCRIPT_DIR}/run_trigger_size_ablation.sh"
"${SCRIPT_DIR}/run_lora_rank_ablation.sh"
"${SCRIPT_DIR}/run_training_steps_ablation.sh"
