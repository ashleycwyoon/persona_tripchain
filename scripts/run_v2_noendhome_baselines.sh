#!/usr/bin/env bash
# Roll out, refine and evaluate the validation answers of the comparison baselines with
# the evaluation config (configs/framework_v2_validation.yaml) into one folder, v2/noendhome,
# which the later pipelines compare against. The Student outputs are reused. The folder name
# is historical: these files were first made when the earlier results still required every
# day to end at home.
# Baselines: Pre-RL (v2/pre_rl_validation), Round 1 (v2/post_rl_validation) and the
# distance-mode weight 0.94 (v2/distmode_w0.94/post_rl_validation).
# Usage: bash scripts/run_v2_noendhome_baselines.sh [gpu for refinement, default 0]
# Output: v2/noendhome/<name>_validation/{rollouts_initial,rollouts}.jsonl and
#         v2/noendhome/analysis/<name>_validation/.
#
# Run after scripts/run_v2_grpo.sh and scripts/run_v2_distmode_weight.sh 0.94 <gpu>.
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (v2 output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
GPU="${1:-0}"
PY="${PYTHON:-python}"
V2="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"
DATA_V2="${DATA_HOME:-$HOME/data}/tripchain_1030/processed_v2"
OUT="$V2/noendhome"; CFG=configs/framework_v2_validation.yaml; N=2653
GT=$DATA_V2/validation/ground_truth_tripmode.jsonl
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"
mkdir -p "$STATE" "$LOGS" "$OUT/analysis"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1
grep -q "^  require_end_home: false" "$CFG" || { echo "require_end_home is not off in $CFG"; exit 2; }

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
# name : Student outputs dir : refinement adapters (Pre-RL uses the config's SFT adapters)
declare -A SRC=([pre_rl]="$V2/pre_rl_validation" [round1]="$V2/post_rl_validation"
                [distmode_w0.94]="$V2/distmode_w0.94/post_rl_validation")
declare -A ADP=([pre_rl]=""
                [round1]="--student1-adapter $V2/grpo/student1/final --student2-adapter $V2/grpo/student2/final"
                [distmode_w0.94]="--student1-adapter $V2/grpo/student1/final --student2-adapter $V2/distmode_w0.94/grpo/student2/final")
for name in pre_rl round1 distmode_w0.94; do
  d="$OUT/${name}_validation"; mkdir -p "$d"
  cp -n "${SRC[$name]}/student1.jsonl" "${SRC[$name]}/student2.jsonl" "$d/"
  step "roll_$name" flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$CFG" \
    --student1 "$d/student1.jsonl" --student2 "$d/student2.jsonl" --output "$d/rollouts_initial.jsonl" \
    --limit $N --workers 16 --rollout-id "noendhome_$name" --resume
  # shellcheck disable=SC2086
  step "refine_$name" flock "$LOCK" $PY scripts/refine_infeasible.py --config "$CFG" \
    --input "$d/rollouts_initial.jsonl" --output "$d/rollouts.jsonl" ${ADP[$name]} \
    --level1-attempts 1 --workers 4 --gpus "$GPU"
  step "eval_$name" $PY scripts/evaluate_population.py --ground-truth "$GT" \
    --student1 "$d/student1.jsonl" --student2 "$d/student2.jsonl" --rollouts "$d/rollouts.jsonl" \
    --output-dir "$OUT/analysis/${name}_validation" --limit $N
done
echo "no-end-home baselines finished ($(date))"
}
main "$@"; exit $?
