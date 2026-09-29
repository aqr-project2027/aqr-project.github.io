#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GIT_PYTHON_REFRESH=quiet
ARGS=()
if [[ "${RESUME:-0}" == "1" ]]; then ARGS+=(--resume); fi
if [[ -n "${INITIALIZE_FROM:-}" ]]; then ARGS+=(--initialize-from "$INITIALIZE_FROM"); fi
"${PYTHON_BIN:-python}" -m contactflow.tools.train_aqr_dp3 \
  --mode train --config "${CONFIG:-configs/a6_post.yaml}" \
  --device "${DEVICE:-auto}" "${ARGS[@]}" "$@"
