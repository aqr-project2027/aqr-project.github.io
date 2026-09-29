#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GIT_PYTHON_REFRESH=quiet
ARGS=(--checkpoint "${CHECKPOINT:-outputs/aqr_a6_post/checkpoints/best.pt}")
"${PYTHON_BIN:-python}" -m contactflow.tools.train_aqr_dp3 \
  --mode test --config "${CONFIG:-configs/a6_post.yaml}" \
  --device "${DEVICE:-auto}" "${ARGS[@]}" "$@"
