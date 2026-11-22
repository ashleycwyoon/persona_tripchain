#!/usr/bin/env bash
# Optional: GRPO round 2, starting from the round-1 GRPO adapters (v2/grpo/student{1,2}/final).
#
# 1. Train rollouts with the round-1 adapters (all 18,574 v2 train people,
#    2 Student1 branches x 2 Student2 samples), as scripts/run_v2_rollouts.sh.
#      GPU0: S1 seed42 -> S2 42/1 -> S2 42/2      GPU1: S1 seed43 -> S2 43/1 -> S2 43/2
# 2. Distribution weights re-measured on these rollouts (50:50 within-group spread,
#    scripts/measure_distribution_weights.py) and written into
#    configs/framework_grpo_train_v2_round2.yaml, then advantages.
# 3. GRPO, same settings as round 1. The --adapter (start and frozen reference) is the
#    round-1 adapter, i.e. the policy that generated these rollouts.
# 4. Round-2 Student outputs on v2 test / validation -> repair -> rollouts ->
#    refinement -> evaluation, figures (GT / Pre-RL / Round 1 / Round 2) and paired
#    bootstraps (Round 2 vs Pre-RL, Round 2 vs Round 1).
# Round-1 results (v2/grpo, v2/post_rl_*, v2/analysis) are only read.
# Every step leaves round2/state/<name>.done; re-running skips finished steps.
#
# Usage: bash scripts/run_v2_round2.sh   (after scripts/run_v2_grpo.sh)
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (round-1 output
# directory, default $OUTPUT_HOME/tripchain_1030/v2), INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
V2="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"
DATA_V2="${DATA_HOME:-$HOME/data}/tripchain_1030/processed_v2"
OUT="$V2/round2"
BASE_CFG=configs/framework_grpo_train_v2.yaml
TRAIN_CFG=configs/framework_grpo_train_v2_round2.yaml
declare -A CFG=([test]=configs/framework_v2_test.yaml [validation]=configs/framework_v2_validation.yaml)
declare -A SIZE=([test]=5307 [validation]=2653)
declare -A GT=([test]=$DATA_V2/test/ground_truth_tripmode.jsonl
               [validation]=$DATA_V2/validation/ground_truth_tripmode.jsonl)
R1="$V2/grpo"; G="$OUT/grpo"; UPD="$OUT/grpo_train/updates"
A1="$G/student1/final"; A2="$G/student2/final"
N_TRAIN=18574
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"
mkdir -p "$STATE" "$LOGS" "$G/student1" "$G/student2" "$UPD" "$OUT/analysis"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}

# Round-2 train config: v2 train config with the round-1 adapters (weights set in step 2).
if [ ! -e "$TRAIN_CFG" ]; then
  sed -e "s#student1_adapter: .*#student1_adapter: $R1/student1/final#" \
      -e "s#student2_adapter: .*#student2_adapter: $R1/student2/final#" \
      -e "1i # GRPO round 2 (scripts/run_v2_round2.sh): rollouts from the round-1 adapters.\n# distribution weights are rewritten by that script from the round-2 rollouts." \
      "$BASE_CFG" > "$TRAIN_CFG"
fi
[ -d "$R1/student1/final" ] && [ -d "$R1/student2/final" ] || { echo "missing round-1 adapters in $R1"; exit 2; }

# 1. Train rollouts.
ROLLS=()
s1() { $PY scripts/run_student_inference_2gpu.py student1 --config "$TRAIN_CFG" --gpus "$1" --batch-size 32 \
         --limit "$N_TRAIN" --seed "$2" --output "$3" --resume; }
s2() { $PY scripts/run_student_inference_2gpu.py student2 --config "$TRAIN_CFG" --gpus "$1" --batch-size 32 \
         --limit "$N_TRAIN" --seed "$2" --student1-input "$3" --output "$4" --resume; }
train_branch() {
  local gpu="$1" seed="$2" dir="$OUT/grpo_train/s1_seed$2"
  mkdir -p "$dir"
  step "train_s1_$seed" s1 "$gpu" "$seed" "$dir/student1.jsonl"
  for sample in 1 2; do
    mkdir -p "$dir/s2_sample$sample"
    step "train_s2_${seed}_$sample" s2 "$gpu" "$((seed * 100 + sample))" "$dir/student1.jsonl" \
      "$dir/s2_sample$sample/student2.jsonl"
    ( step "roll_train_${seed}_$sample" flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$TRAIN_CFG" \
        --student1 "$dir/student1.jsonl" --student2 "$dir/s2_sample$sample/student2.jsonl" \
        --output "$dir/s2_sample$sample/rollouts.jsonl" --limit "$N_TRAIN" --workers 16 \
        --rollout-id "r2_train_s1-${seed}_s2-$sample" --resume ) &
    ROLLS+=("$!")
  done
  wait "${ROLLS[@]}"
}
echo "v2 GRPO round 2 -> $OUT ($(date))"
train_branch 0 42 & B0=$!
train_branch 1 43 & B1=$!
wait "$B0" && wait "$B1"
ROLLOUTS=("$OUT"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl)

# 2. Weights, then advantages.
step measure_weights sh -c "$PY scripts/measure_distribution_weights.py ${ROLLOUTS[*]} --config $TRAIN_CFG > $OUT/weights.json"
step set_weights $PY - "$TRAIN_CFG" "$OUT/weights.json" <<'EOF'
import json, re, sys
path, weights = sys.argv[1], json.load(open(sys.argv[2]))["weights"]
text = open(path).read()
for key in ("sequence", "trip_mode", "first_departure"):
    text, n = re.subn(rf"(\n  distribution:(?:\n    .*)*?\n    {key}: )[0-9.]+", rf"\g<1>{weights[key]:.2f}", text)
    assert n == 1, key
open(path, "w").write(text)
print({k: round(v, 2) for k, v in weights.items()})
EOF
step prep_updates $PY scripts/prepare_policy_updates.py "${ROLLOUTS[@]}" \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore

# 3. GRPO from the round-1 adapters.
train() {
  CUDA_VISIBLE_DEVICES="$2" $PY scripts/train_group_relative.py --updates "$UPD/$1_updates.jsonl" \
    --adapter "$R1/$1/final" --output "$G/$1" --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
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
           --rollout-id post_rl_round2 --resume; }
refine() { flock "$LOCK" $PY scripts/refine_infeasible.py --config "${CFG[$1]}" \
             --input "$OUT/post_rl_$1/rollouts_initial.jsonl" --output "$OUT/post_rl_$1/rollouts.jsonl" \
             --student1-adapter "$A1" --student2-adapter "$A2" --level1-attempts 1 --workers 4 --gpus "$2"; }
mkdir -p "$OUT/post_rl_test" "$OUT/post_rl_validation"
lane1() { step grpo_student1 train student1 1; step post_test_s1 gen1 test 1; step post_validation_s1 gen1 validation 1; }
lane1 & L1=$!
step grpo_student2 train student2 0
wait "$L1"

# 4. Evaluation.
step post_test_s2 gen2 test 0,1
( step post_test_repair repair test 0 && step roll_post_test roll test ) & T=$!
step post_validation_s2 gen2 validation 1
step post_validation_repair repair validation 1
wait "$T"
( step roll_post_validation roll validation ) & V=$!
step refine_post_test refine test 0,1
wait "$V"
step refine_post_validation refine validation 0,1
for split in test validation; do
  d="$OUT/post_rl_$split"
  step "eval_post_$split" $PY scripts/evaluate_population.py --ground-truth "${GT[$split]}" \
    --student1 "$d/student1.jsonl" --student2 "$d/student2.jsonl" --rollouts "$d/rollouts.jsonl" \
    --output-dir "$OUT/analysis/post_rl_$split" --limit "${SIZE[$split]}"
  step "eval_post_${split}_noref" $PY scripts/evaluate_population.py --ground-truth "${GT[$split]}" \
    --student1 "$d/student1.jsonl" --student2 "$d/student2.jsonl" --rollouts "$d/rollouts_initial.jsonl" \
    --output-dir "$OUT/analysis/post_rl_${split}_noref" --limit "${SIZE[$split]}"
  step "boot_pre_$split" $PY scripts/bootstrap_pair.py --truth "${GT[$split]}" \
    --a "$V2/pre_rl_$split/rollouts_refined.jsonl" --b "$d/rollouts.jsonl" \
    --labels Pre-RL,Round2 --output "$OUT/analysis/bootstrap_pre_vs_round2_$split.json"
  step "boot_r1_$split" $PY scripts/bootstrap_pair.py --truth "${GT[$split]}" \
    --a "$V2/post_rl_$split/rollouts.jsonl" --b "$d/rollouts.jsonl" \
    --labels Round1,Round2 --output "$OUT/analysis/bootstrap_round1_vs_round2_$split.json"
  step "figures_$split" $PY scripts/visualize_population.py --config "${CFG[$split]}" \
    --sft-dir "$DATA_V2/sft" --output "$OUT/analysis/distributions_$split" \
    --population "Pre-RL=$V2/pre_rl_$split/rollouts_refined.jsonl" \
    --population "Round 1=$V2/post_rl_$split/rollouts.jsonl" \
    --population "Round 2=$d/rollouts.jsonl"
done
echo "v2 GRPO round 2 finished ($(date))"
}
main "$@"; exit $?
