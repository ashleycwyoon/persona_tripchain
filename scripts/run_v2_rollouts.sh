#!/usr/bin/env bash
# Step 1 of the v2 pipeline: rollouts of the SFT (pre-RL) Students.
#
# GRPO train rollouts (all v2 train people, 2 Student1 branches x 2 Student2
# samples) and pre-RL baselines on v2 test and validation. Stops before
# scripts/prepare_policy_updates.py: the distribution reward weights are measured
# on these train rollouts first (scripts/measure_distribution_weights.py).
#
# Two GPU lanes (one job per GPU at a time); environment rollouts run in the
# background, one 16-worker rollout at a time (flock), so GraphHopper load is bounded.
#   GPU0: S1 seed42 -> S2 42/1 -> S2 42/2 -> test S2 + repair -> validation S2 + repair
#   GPU1: test S1 -> validation S1 -> S1 seed43 -> S2 43/1 -> S2 43/2
# Every step leaves state/<name>.done; re-running skips finished steps and
# resumes the rest (inference and rollouts resume per person).
#
# Usage: bash scripts/run_v2_rollouts.sh
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), INFERENCE_CPU_THREADS (default 4).
#
# The whole script lives in main() so that bash parses it completely before
# running: editing this file during a run cannot change what that run executes.
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
OUT="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"
TRAIN_CFG=configs/framework_grpo_train_v2.yaml
declare -A EVAL_CFG=([test]=configs/framework_v2_test.yaml [validation]=configs/framework_v2_validation.yaml)
declare -A SIZE=([test]=5307 [validation]=2653)
N_TRAIN=18574
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$OUT/.rollout.lock"
mkdir -p "$STATE" "$LOGS"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1

step() {  # step <name> <command...>: run once, log to logs/<name>.log
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
wait_for() { while [ ! -e "$STATE/$1.done" ]; do sleep 30; done; }
roll() {  # roll <name> <config> <s1> <s2> <out> <limit> <id>  (background, serialized)
  local name="$1" cfg="$2" s1="$3" s2="$4" out="$5" limit="$6" rid="$7"
  ( step "$name" flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$cfg" \
      --student1 "$s1" --student2 "$s2" --output "$out" --limit "$limit" \
      --workers 16 --rollout-id "$rid" --resume ) &
  ROLLS+=("$!")
}
s1() { $PY scripts/run_student_inference_2gpu.py student1 --config "$1" --gpus "$2" --batch-size 32 \
         --limit "$3" --seed "$4" --output "$5" --resume; }
s2() { $PY scripts/run_student_inference_2gpu.py student2 --config "$1" --gpus "$2" --batch-size 32 \
         --limit "$3" --seed "$4" --student1-input "$5" --output "$6" --resume; }
repair() { $PY scripts/repair_student2_failures.py --config "$1" --student1 "$2" --student2 "$3" --seed 42 --gpu "$4"; }

ROLLS=()
train_branch() {  # train_branch <gpu> <seed>
  local gpu="$1" seed="$2" dir="$OUT/grpo_train/s1_seed$2"
  mkdir -p "$dir"
  step "train_s1_$seed" s1 "$TRAIN_CFG" "$gpu" "$N_TRAIN" "$seed" "$dir/student1.jsonl"
  for sample in 1 2; do
    mkdir -p "$dir/s2_sample$sample"
    step "train_s2_${seed}_$sample" s2 "$TRAIN_CFG" "$gpu" "$N_TRAIN" "$((seed * 100 + sample))" \
      "$dir/student1.jsonl" "$dir/s2_sample$sample/student2.jsonl"
    roll "roll_train_${seed}_$sample" "$TRAIN_CFG" "$dir/student1.jsonl" "$dir/s2_sample$sample/student2.jsonl" \
      "$dir/s2_sample$sample/rollouts.jsonl" "$N_TRAIN" "train_s1-${seed}_s2-$sample"
  done
}
pre_rl_s1() { local split="$1" gpu="$2" d="$OUT/pre_rl_$1"; mkdir -p "$d"
  step "${split}_s1" s1 "${EVAL_CFG[$split]}" "$gpu" "${SIZE[$split]}" 42 "$d/student1.jsonl"; }
pre_rl_s2() { local split="$1" gpu="$2" d="$OUT/pre_rl_$1"
  wait_for "${split}_s1"
  step "${split}_s2" s2 "${EVAL_CFG[$split]}" "$gpu" "${SIZE[$split]}" 42 "$d/student1.jsonl" "$d/student2.jsonl"
  step "${split}_repair" repair "${EVAL_CFG[$split]}" "$d/student1.jsonl" "$d/student2.jsonl" "$gpu"
  roll "roll_$split" "${EVAL_CFG[$split]}" "$d/student1.jsonl" "$d/student2.jsonl" "$d/rollouts.jsonl" \
    "${SIZE[$split]}" "pre_rl_$split"
}

lane0() { train_branch 0 42; pre_rl_s2 test 0; pre_rl_s2 validation 0; wait "${ROLLS[@]}"; }
lane1() { pre_rl_s1 test 1; pre_rl_s1 validation 1; train_branch 1 43; wait "${ROLLS[@]}"; }
echo "v2 rollouts -> $OUT ($(date))"
lane0 & L0=$!
lane1 & L1=$!
wait "$L0" && wait "$L1"
echo "All v2 inference and rollouts finished ($(date))"
}
main "$@"; exit $?
