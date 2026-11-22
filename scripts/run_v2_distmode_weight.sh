#!/usr/bin/env bash
# Optional: trip-mode weight sweep for the distance-conditioned trip-mode reward.
# Usage: bash scripts/run_v2_distmode_weight.sh <trip_mode weight> <gpu>
#   e.g. 0.75 0  (S2 individual : distribution ~ 44:56 in spread), 0.94 1 (~ 40:60)
# Same as scripts/run_v2_distmode.sh (weight 0.50) except distribution.trip_mode,
# and everything GPU-side runs on the one given GPU so two weights run side by side.
# The train config configs/framework_grpo_train_v2_distmode_w<weight>.yaml is
# generated from the distance-mode config on first use.
# Output: v2/distmode_w<weight>/. Rollouts/refinement share v2/.rollout.lock (GraphHopper).
# Comparison: Pre-RL, Round 1, distance mode 0.50 and this weight.
#
# Run after scripts/run_v2_distmode.sh. Environment: PYTHON (interpreter, default
# "python"), V2_ROOT (round-1 output directory, default $OUTPUT_HOME/tripchain_1030/v2),
# INFERENCE_CPU_THREADS (default 4), SPLITS (default "validation"; "test validation"
# also generates, evaluates and plots the held-out test split).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
W="$1"; GPU="$2"
PY="${PYTHON:-python}"
V2="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"
DATA_V2="${DATA_HOME:-$HOME/data}/tripchain_1030/processed_v2"
OUT="$V2/distmode_w$W"
BASE_CFG=configs/framework_grpo_train_v2_distmode.yaml
TRAIN_CFG=configs/framework_grpo_train_v2_distmode_w$W.yaml
declare -A CFG=([test]=configs/framework_v2_test.yaml [validation]=configs/framework_v2_validation.yaml)
declare -A SIZE=([test]=5307 [validation]=2653)
declare -A GT=([test]=$DATA_V2/test/ground_truth_tripmode.jsonl
               [validation]=$DATA_V2/validation/ground_truth_tripmode.jsonl)
UPD="$OUT/updates"; A1="$V2/grpo/student1/final"; A2="$OUT/grpo/student2/final"
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"
mkdir -p "$STATE" "$LOGS" "$UPD" "$OUT/grpo/student2" "$OUT/analysis"
# Splits to generate, evaluate and plot: validation only by default (test is held out);
# SPLITS="test validation" runs both.
SPLITS="${SPLITS:-validation}"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
if [ ! -e "$TRAIN_CFG" ]; then
  sed -e "/^  distribution:/,/^  [a-z]/ s/^    trip_mode: .*/    trip_mode: $W/" \
      -e "1i # Trip-mode weight sweep (scripts/run_v2_distmode_weight.sh): $BASE_CFG with distribution.trip_mode $W." \
      "$BASE_CFG" > "$TRAIN_CFG"
fi
grep -q "^    trip_mode: $W\$" "$TRAIN_CFG" || { echo "trip_mode weight not set in $TRAIN_CFG"; exit 2; }
echo "v2 distance-mode weight $W on GPU $GPU -> $OUT ($(date))"

step prep_updates $PY scripts/prepare_policy_updates.py "$V2"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore
step check_student1_same $PY -c "
import json, sys
a, b = ([json.loads(l) for l in open(p)] for p in sys.argv[1:])
key = lambda r: (r['person_id'], r['messages'], r['return'], r['group_advantage'])
assert len(a) == len(b) and all(key(x) == key(y) for x, y in zip(a, b)), 'Student1 updates differ from round 1'
print('identical training fields:', len(a))" "$UPD/student1_updates.jsonl" "$V2/grpo_train/updates/student1_updates.jsonl"

SFT2="$($PY -c "import sys; sys.path.insert(0,'.'); from tripchain.common import load_config; print(load_config('$TRAIN_CFG')['paths']['student2_adapter'])")"
step grpo_student2 sh -c "CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_group_relative.py --updates $UPD/student2_updates.jsonl \
  --adapter $SFT2 --output $OUT/grpo/student2 --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
  --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 && rm -f $OUT/grpo/student2/checkpoint-*/optimizer.pt"

for split in $SPLITS; do
  mkdir -p "$OUT/post_rl_$split"
  [ -e "$STATE/copy_s1_$split.done" ] || { cp "$V2/post_rl_$split/student1.jsonl" "$OUT/post_rl_$split/"; touch "$STATE/copy_s1_$split.done"; }
done
gen2() { $PY scripts/run_student_inference_2gpu.py student2 --config "${CFG[$1]}" --adapter "$A2" \
           --gpus "$GPU" --batch-size 32 --limit "${SIZE[$1]}" --seed 42 \
           --student1-input "$OUT/post_rl_$1/student1.jsonl" --output "$OUT/post_rl_$1/student2.jsonl" --resume; }
repair() { $PY scripts/repair_student2_failures.py --config "${CFG[$1]}" \
             --student1 "$OUT/post_rl_$1/student1.jsonl" --student2 "$OUT/post_rl_$1/student2.jsonl" \
             --student1-adapter "$A1" --student2-adapter "$A2" --seed 42 --gpu "$GPU"; }
roll() { flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "${CFG[$1]}" \
           --student1 "$OUT/post_rl_$1/student1.jsonl" --student2 "$OUT/post_rl_$1/student2.jsonl" \
           --output "$OUT/post_rl_$1/rollouts_initial.jsonl" --limit "${SIZE[$1]}" --workers 16 \
           --rollout-id "post_rl_distmode_w$W" --resume; }
refine() { flock "$LOCK" $PY scripts/refine_infeasible.py --config "${CFG[$1]}" \
             --input "$OUT/post_rl_$1/rollouts_initial.jsonl" --output "$OUT/post_rl_$1/rollouts.jsonl" \
             --student1-adapter "$A1" --student2-adapter "$A2" --level1-attempts 1 --workers 4 --gpus "$GPU"; }

if [ "$SPLITS" = validation ]; then
  step post_validation_s2 gen2 validation
  step post_validation_repair repair validation
  step roll_post_validation roll validation
  step refine_post_validation refine validation
else
  # Test rollouts (CPU) overlap validation generation (GPU).
  step post_test_s2 gen2 test
  ( step post_test_repair repair test && step roll_post_test roll test ) & T=$!
  step post_validation_s2 gen2 validation
  step post_validation_repair repair validation
  wait "$T"
  ( step roll_post_validation roll validation ) & V=$!
  step refine_post_test refine test
  wait "$V"
  step refine_post_validation refine validation
fi
for split in $SPLITS; do
  d="$OUT/post_rl_$split"
  step "eval_post_$split" $PY scripts/evaluate_population.py --ground-truth "${GT[$split]}" \
    --student1 "$d/student1.jsonl" --student2 "$d/student2.jsonl" --rollouts "$d/rollouts.jsonl" \
    --output-dir "$OUT/analysis/post_rl_$split" --limit "${SIZE[$split]}"
  for ref in "Pre-RL:$V2/pre_rl_$split/rollouts_refined.jsonl" "Round1:$V2/post_rl_$split/rollouts.jsonl" \
             "DistMode0.50:$V2/distmode/post_rl_$split/rollouts.jsonl"; do
    label="${ref%%:*}"
    step "boot_${label}_$split" $PY scripts/bootstrap_pair.py --truth "${GT[$split]}" \
      --a "${ref#*:}" --b "$d/rollouts.jsonl" --labels "$label,DistMode$W" --workers 8 \
      --output "$OUT/analysis/bootstrap_${label}_vs_w${W}_$split.json"
  done
  step "figures_$split" $PY scripts/visualize_population.py --config "${CFG[$split]}" \
    --sft-dir "$DATA_V2/sft" --output "$OUT/analysis/distributions_$split" \
    --population "Pre-RL=$V2/pre_rl_$split/rollouts_refined.jsonl" \
    --population "Round 1=$V2/post_rl_$split/rollouts.jsonl" \
    --population "Distance mode 0.50=$V2/distmode/post_rl_$split/rollouts.jsonl" \
    --population "Distance mode $W=$d/rollouts.jsonl"
done
echo "v2 distance-mode weight $W finished ($(date))"
}
main "$@"; exit $?
