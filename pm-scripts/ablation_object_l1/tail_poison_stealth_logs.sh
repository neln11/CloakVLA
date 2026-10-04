#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
LOG_ROOT="${LOG_ROOT:-${ROOT_DIR}/ablation_runs/object_l1/launcher_logs}"
MAIN_LOG="${MAIN_LOG:-${LOG_ROOT}/run_poison_stealth_gpu2_3.out}"

logs=()
if [[ -f "${MAIN_LOG}" ]]; then
  logs+=("${MAIN_LOG}")
fi

mapfile -t child_logs < <(
  find "${LOG_ROOT}" -maxdepth 1 -type f -name '*poison_stealth_eps*.log' \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn \
    | head -n "${TAIL_LATEST_N:-4}" \
    | cut -d' ' -f2- \
    | awk '!seen[$0]++'
)

if [[ "${#child_logs[@]}" -eq 0 && -f "${MAIN_LOG}" ]]; then
  mapfile -t child_logs < <(
    sed -n 's/.*log=\(.*poison_stealth_eps.*\.log\).*/\1/p' "${MAIN_LOG}" \
      | awk '!seen[$0]++'
  )
fi

for log_path in "${child_logs[@]}"; do
  [[ -f "${log_path}" ]] && logs+=("${log_path}")
done

if [[ "${#logs[@]}" -eq 0 ]]; then
  echo "[TailPoisonStealth][ERROR] No poison-stealth logs found under: ${LOG_ROOT}" >&2
  exit 1
fi

echo "[TailPoisonStealth] Following ${#logs[@]} log file(s):"
printf '  %s\n' "${logs[@]}"

tail -n "${TAIL_LINES:-80}" -F "${logs[@]}" \
  | awk '
      /^==> .* <==$/ {
        file=$0
        sub(/^==> /, "", file)
        sub(/ <==$/, "", file)
        n=split(file, parts, "/")
        label=parts[n]
        next
      }
      {
        if (label == "") {
          print
        } else {
          print "[" label "] " $0
        }
        fflush()
      }
    '
