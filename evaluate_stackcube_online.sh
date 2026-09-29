#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda:0}"
EVAL_START="${EVAL_START:-804}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
MAX_ENV_STEPS="${MAX_ENV_STEPS:-200}"
EXEC_STEPS="${EXEC_STEPS:-4}"
IGNORE_TRUNCATED="${IGNORE_TRUNCATED:-1}"
EVAL_SEED="${EVAL_SEED:-20260730}"
DEFAULT_OUTPUT_DIR="outputs/aqr_a6_post"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_OUTPUT_DIR}"
CHECKPOINT="${CHECKPOINT:-$OUTPUT_DIR/checkpoints/best.pt}"
DEMO_ROOT="${DEMO_ROOT:-data/maniskill_demos/StackCube-v1}"
SOURCE_JSON="${SOURCE_JSON:-}"
EVAL_DIR="${EVAL_DIR:-$OUTPUT_DIR/online_eval}"
PAPER_TRACE_OUT="${PAPER_TRACE_OUT:-}"
PAPER_SNAPSHOT_DIR="${PAPER_SNAPSHOT_DIR:-}"
PAPER_SNAPSHOT_EPISODES="${PAPER_SNAPSHOT_EPISODES:-first}"
PAPER_SNAPSHOT_PLAN_STRIDE="${PAPER_SNAPSHOT_PLAN_STRIDE:-1}"
PAPER_OFFSET_PROBE_ACTION="${PAPER_OFFSET_PROBE_ACTION:-}"

[[ "$EXEC_STEPS" =~ ^[1-8]$ ]] || {
  echo "EXEC_STEPS must be an integer from 1 to 8; got $EXEC_STEPS" >&2
  exit 2
}
[[ -f "$CHECKPOINT" ]] || {
  echo "Missing checkpoint: $CHECKPOINT" >&2
  exit 2
}
if [[ -z "$SOURCE_JSON" ]]; then
  SOURCE_JSON="$(
    find "$DEMO_ROOT" -type f -name 'trajectory*.json' 2>/dev/null \
      | grep -i 'pointcloud' \
      | grep -i 'pd_joint_delta_pos' \
      | sort \
      | head -n 1 || true
  )"
fi
[[ -f "$SOURCE_JSON" ]] || {
  echo "No local StackCube pointcloud + pd_joint_delta_pos source JSON found under $DEMO_ROOT" >&2
  echo "motionplanning/ and rl/ are both accepted; set SOURCE_JSON explicitly if needed." >&2
  exit 2
}

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export GIT_PYTHON_REFRESH=quiet
export PYTHONUNBUFFERED=1
mkdir -p "$EVAL_DIR"

TIME_LIMIT_ARGS=()
if [[ "$IGNORE_TRUNCATED" == "1" ]]; then
  TIME_LIMIT_ARGS+=(--ignore-truncated)
elif [[ "$IGNORE_TRUNCATED" != "0" ]]; then
  echo "IGNORE_TRUNCATED must be 0 or 1; got $IGNORE_TRUNCATED" >&2
  exit 2
fi

PAPER_TRACE_ARGS=()
if [[ -n "$PAPER_TRACE_OUT" ]]; then
  PAPER_TRACE_ARGS+=(--paper-trace-out "$PAPER_TRACE_OUT")
fi

PAPER_SNAPSHOT_ARGS=()
if [[ -n "$PAPER_SNAPSHOT_DIR" ]]; then
  PAPER_SNAPSHOT_ARGS+=(
    --paper-snapshot-dir "$PAPER_SNAPSHOT_DIR"
    --paper-snapshot-episodes "$PAPER_SNAPSHOT_EPISODES"
    --paper-snapshot-plan-stride "$PAPER_SNAPSHOT_PLAN_STRIDE"
  )
  if [[ -n "$PAPER_OFFSET_PROBE_ACTION" ]]; then
    PAPER_SNAPSHOT_ARGS+=(
      --paper-offset-probe-action "$PAPER_OFFSET_PROBE_ACTION"
    )
  fi
fi

"$PYTHON_BIN" -u -m contactflow.tools.evaluate_aqr_dp3_maniskill \
  --checkpoint "$CHECKPOINT" \
  --dp3-root "$PROJECT_ROOT" \
  --source-json "$SOURCE_JSON" \
  --checkpoint-policy auto \
  --exec-steps "$EXEC_STEPS" \
  --eval-start "$EVAL_START" \
  --eval-episodes "$EVAL_EPISODES" \
  --max-env-steps "$MAX_ENV_STEPS" \
  --query-points 4096 \
  --point-sampling-mode pooled \
  --robot-seg-ids auto \
  --crop-min=-0.5,-0.5,-0.05 \
  --crop-max=0.5,0.65,0.5 \
  --num-inference-steps 10 \
  --seed "$EVAL_SEED" \
  --device "$DEVICE" \
  --report-out "$EVAL_DIR/report.json" \
  --csv-out "$EVAL_DIR/episodes.csv" \
  --step-csv-out "$EVAL_DIR/steps.csv" \
  "${TIME_LIMIT_ARGS[@]}" \
  "${PAPER_TRACE_ARGS[@]}" \
  "${PAPER_SNAPSHOT_ARGS[@]}" \
  "$@"
