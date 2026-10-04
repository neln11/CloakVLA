#!/usr/bin/env bash

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
