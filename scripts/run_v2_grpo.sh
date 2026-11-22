#!/usr/bin/env bash
# Step 2 of the v2 pipeline: GRPO round 1 and evaluation.
#
# GRPO (1 round) on the rollouts of scripts/run_v2_rollouts.sh, then evaluation
# of Pre-RL and Post-RL on v2 test and validation, with refinement and figures.
# GRPO settings: lr 1e-5, 1 epoch, reference penalty 0.02, batch 2 x accum 16,
# offline 1 round. Distribution weights: configs/framework_grpo_train_v2.yaml.
#
#   GPU1: GRPO Student1 -> Pre-RL refinement (test, validation) -> post-RL Student1 (test, validation)
#   GPU0: GRPO Student2
#   then: post-RL Student2 test (GPU0,1) -> [repair test | Student2 validation] -> repair validation
#         -> rollouts (background, serialized) -> refinement -> evaluation, comparison, figures
# Every step leaves state/<name>.done; re-running skips finished steps.
#
# Usage: bash scripts/run_v2_grpo.sh
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
OUT="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"
SFT_DATA="${DATA_HOME:-$HOME/data}/tripchain_1030/processed_v2/sft"
TRAIN_CFG=configs/framework_grpo_train_v2.yaml
declare -A CFG=([test]=configs/framework_v2_test.yaml [validation]=configs/framework_v2_validation.yaml)
declare -A SIZE=([test]=5307 [validation]=2653)
G="$OUT/grpo"; UPD="$OUT/grpo_train/updates"
A1="$G/student1/final"; A2="$G/student2/final"
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$OUT/.rollout.lock"
mkdir -p "$STATE" "$LOGS" "$G/student1" "$G/student2" "$UPD"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
wait_for() { while [ ! -e "$STATE/$1.done" ]; do sleep 30; done; }
train() {  # train <student> <gpu> <sft adapter>
  CUDA_VISIBLE_DEVICES="$2" $PY scripts/train_group_relative.py --updates "$UPD/$1_updates.jsonl" \
    --adapter "$3" --output "$G/$1" --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
    --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 \
  && rm -f "$G/$1"/checkpoint-*/optimizer.pt
}
gen1() { $PY scripts/run_student_inference_2gpu.py student1 --config "${CFG[$1]}" --adapter "$A1" \
           --gpus "$2" --batch-size 32 --limit "${SIZE[$1]}" --seed 42 \
           --output "$OUT/post_rl_$1/student1.jsonl" --resume; }
gen2() { $PY scripts/run_student_inference_2gpu.py student2 --config "${CFG[$1]}" --adapter "$A2" \
           --gpus "$2" --batch-size 32 --limit "${SIZE[$1]}" --seed 42 \
           --student1-input "$OUT/post_rl_$1/student1.jsonl" --output "$OUT/post_rl_$1/student2.jsonl" --resume; }
repair() { $PY scripts/repair_student2_failures.py --config "${CFG[$1]}" \
             --student1 "$OUT/post_rl_$1/student1.jsonl" --student2 "$OUT/post_rl_$1/student2.jsonl" \
             --student1-adapter "$A1" --student2-adapter "$A2" --seed 42 --gpu "$2"; }
roll() { flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "${CFG[$1]}" \
           --student1 "$OUT/post_rl_$1/student1.jsonl" --student2 "$OUT/post_rl_$1/student2.jsonl" \
           --output "$OUT/post_rl_$1/rollouts_initial.jsonl" --limit "${SIZE[$1]}" --workers 16 \
           --rollout-id post_rl --resume; }
refine() {  # refine <split> <pre|post> <gpus>
  local d="$OUT/$2_rl_$1" adapters=()
  [ "$2" = post ] && adapters=(--student1-adapter "$A1" --student2-adapter "$A2")
  local input="$d/rollouts_initial.jsonl"; [ "$2" = pre ] && input="$d/rollouts.jsonl"
  local output="$d/rollouts_refined.jsonl"; [ "$2" = post ] && output="$d/rollouts.jsonl"
  flock "$LOCK" $PY scripts/refine_infeasible.py --config "${CFG[$1]}" --input "$input" --output "$output" \
    "${adapters[@]}" --level1-attempts 1 --workers 4 --gpus "$3"
}
evaluate() {  # evaluate <split> <pre|post> <rollouts> <out>
  local d="$OUT/$2_rl_$1"
  $PY scripts/evaluate_population.py --ground-truth "$($PY -c "import sys; sys.path.insert(0,'.'); from tripchain.common import load_config; print(load_config('${CFG[$1]}')['paths']['ground_truth_reference'])")" \
    --student1 "$d/student1.jsonl" --student2 "$d/student2.jsonl" --rollouts "$3" \
    --output-dir "$4" --limit "${SIZE[$1]}"
}

echo "v2 GRPO -> $OUT ($(date))"
step prep_updates $PY scripts/prepare_policy_updates.py "$OUT"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore
mkdir -p "$OUT/post_rl_test" "$OUT/post_rl_validation"
lane1() {
  step grpo_student1 train student1 1 "$($PY -c "import sys; sys.path.insert(0,'.'); from tripchain.common import load_config; print(load_config('$TRAIN_CFG')['paths']['student1_adapter'])")"
  step refine_pre_test refine test pre 1
  step refine_pre_validation refine validation pre 1
  step post_test_s1 gen1 test 1
  step post_validation_s1 gen1 validation 1
}
lane0() {
  step grpo_student2 train student2 0 "$($PY -c "import sys; sys.path.insert(0,'.'); from tripchain.common import load_config; print(load_config('$TRAIN_CFG')['paths']['student2_adapter'])")"
}
lane1 & L1=$!
lane0 & L0=$!
wait "$L1" && wait "$L0"

step post_test_s2 gen2 test 0,1
( step post_test_repair repair test 0 && step roll_post_test roll test ) & T=$!
step post_validation_s2 gen2 validation 1
step post_validation_repair repair validation 1
wait "$T"
( step roll_post_validation roll validation ) & V=$!
step refine_post_test refine test post 0,1
wait "$V"
step refine_post_validation refine validation post 0,1

for split in test validation; do
  step "eval_pre_$split" evaluate "$split" pre "$OUT/pre_rl_$split/rollouts_refined.jsonl" "$OUT/analysis/pre_rl_$split"
  step "eval_pre_${split}_noref" evaluate "$split" pre "$OUT/pre_rl_$split/rollouts.jsonl" "$OUT/analysis/pre_rl_${split}_noref"
  step "eval_post_$split" evaluate "$split" post "$OUT/post_rl_$split/rollouts.jsonl" "$OUT/analysis/post_rl_$split"
  step "eval_post_${split}_noref" evaluate "$split" post "$OUT/post_rl_$split/rollouts_initial.jsonl" "$OUT/analysis/post_rl_${split}_noref"
  step "compare_$split" $PY scripts/compare_rollouts.py --pre "$OUT/pre_rl_$split/rollouts_refined.jsonl" \
    --post "$OUT/post_rl_$split/rollouts.jsonl" --output "$OUT/analysis/comparison_pre_post_$split.json"
  step "figures_$split" $PY scripts/visualize_population.py --config "${CFG[$split]}" \
    --sft-dir "$SFT_DATA" --output "$OUT/analysis/distributions_$split" \
    --population "Pre-RL=$OUT/pre_rl_$split/rollouts_refined.jsonl" \
    --population "Post-RL=$OUT/post_rl_$split/rollouts.jsonl"
done
echo "v2 GRPO, evaluation and figures finished ($(date))"
}
main "$@"; exit $?
