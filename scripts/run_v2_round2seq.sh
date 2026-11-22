#!/usr/bin/env bash
# GRPO round 2 from SeqBoost (configs/framework_grpo_train_v2_round2seq.yaml), validation only.
# 1. Train rollouts with the SeqBoost adapters on all 18,574 v2 train people:
#    3 Student1 branches (seeds 42/43/44) x 2 Student2 samples. Each generation uses both GPUs
#    (data-parallel shards); environment rollouts run on CPU in the background (flock).
# 2. Updates (sequence 1.64, trip_mode 0.94, trip_distance 1.04, first_departure 0.86),
#    then Student2 GRPO on GPU0 and Student1 GRPO on GPU1, both starting from (and
#    referenced to) the SeqBoost adapters that generated the rollouts.
# 3. Validation: Student1 -> Student2 -> repair -> rollouts -> refinement -> evaluation,
#    bootstraps against the no-end-home baselines and SeqBoost, figures.
# Every step leaves round2seq/state/<name>.done; re-running skips finished steps.
# Usage: bash scripts/run_v2_round2seq.sh
#
# Needs GPUs 0 and 1. Run after scripts/run_v2_seqboost.sh. The train config reads the
# SeqBoost adapters from ${OUTPUT_ROOT}/v2/seqboost/grpo, so V2_ROOT must be that v2
# directory (the default).
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (v2 output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
V2="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"
DATA_V2="${DATA_HOME:-$HOME/data}/tripchain_1030/processed_v2"
OUT="$V2/round2seq"; TRAIN_CFG=configs/framework_grpo_train_v2_round2seq.yaml
CFG=configs/framework_v2_validation.yaml; N=2653; N_TRAIN=18574
GT=$DATA_V2/validation/ground_truth_tripmode.jsonl
SB="$V2/seqboost/grpo"; UPD="$OUT/updates"; A1="$OUT/grpo/student1/final"; A2="$OUT/grpo/student2/final"
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"; D="$OUT/post_rl_validation"
mkdir -p "$STATE" "$LOGS" "$UPD" "$OUT/grpo/student1" "$OUT/grpo/student2" "$OUT/analysis" "$D"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1
for cfg in "$CFG" "$TRAIN_CFG"; do
  grep -q "^  require_end_home: false" "$cfg" || { echo "require_end_home is not off in $cfg"; exit 2; }
done
$PY -c "
import sys
from pathlib import Path
sys.path.insert(0, '.')
from tripchain.common import load_config
paths, seqboost = load_config(sys.argv[1])['paths'], Path(sys.argv[2])
sys.exit(any(Path(paths[f'{s}_adapter']) != seqboost / s / 'final' for s in ('student1', 'student2')))" \
  "$TRAIN_CFG" "$SB" || { echo "$TRAIN_CFG does not point at the SeqBoost adapters in $SB"; exit 2; }
[ -d "$SB/student1/final" ] && [ -d "$SB/student2/final" ] || { echo "missing SeqBoost adapters"; exit 2; }

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
echo "v2 round 2 (SeqBoost, 3 branches) -> $OUT ($(date))"

# 1. Train rollouts.
S1SEEDS=(42 43 44)
for seed in "${S1SEEDS[@]}"; do
  dir="$OUT/grpo_train/s1_seed$seed"; mkdir -p "$dir"
  step "train_s1_$seed" $PY scripts/run_student_inference_2gpu.py student1 --config "$TRAIN_CFG" --gpus 0,1 \
    --batch-size 32 --limit "$N_TRAIN" --seed "$seed" --output "$dir/student1.jsonl" --resume
done
ROLLS=()
for seed in "${S1SEEDS[@]}"; do for sample in 1 2; do
  dir="$OUT/grpo_train/s1_seed$seed"; mkdir -p "$dir/s2_sample$sample"
  step "train_s2_${seed}_$sample" $PY scripts/run_student_inference_2gpu.py student2 --config "$TRAIN_CFG" --gpus 0,1 \
    --batch-size 32 --limit "$N_TRAIN" --seed "$((seed * 100 + sample))" --student1-input "$dir/student1.jsonl" \
    --output "$dir/s2_sample$sample/student2.jsonl" --resume
  ( step "roll_train_${seed}_$sample" flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$TRAIN_CFG" \
      --student1 "$dir/student1.jsonl" --student2 "$dir/s2_sample$sample/student2.jsonl" \
      --output "$dir/s2_sample$sample/rollouts.jsonl" --limit "$N_TRAIN" --workers 16 \
      --rollout-id "r2seq_train_s1-${seed}_s2-$sample" --resume ) &
  ROLLS+=("$!")
done; done
for pid in "${ROLLS[@]}"; do wait "$pid"; done

# 2. Updates; Student2 GRPO on GPU0 and Student1 on GPU1 from the SeqBoost adapters.
step prep_updates $PY scripts/prepare_policy_updates.py "$OUT"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore
grpo() {  # student gpu
  CUDA_VISIBLE_DEVICES="$2" $PY scripts/train_group_relative.py --updates "$UPD/$1_updates.jsonl" \
    --adapter "$SB/$1/final" --output "$OUT/grpo/$1" --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
    --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 \
  && rm -f "$OUT/grpo/$1"/checkpoint-*/optimizer.pt
}
( step grpo_student2 grpo student2 0 ) & S2=$!
step grpo_student1 grpo student1 1
step post_validation_s1 $PY scripts/run_student_inference_2gpu.py student1 --config "$CFG" --adapter "$A1" \
  --gpus 1 --batch-size 32 --limit $N --seed 42 --output "$D/student1.jsonl" --resume
wait "$S2"

# 3. Validation with both GPUs.
step post_validation_s2 $PY scripts/run_student_inference_2gpu.py student2 --config "$CFG" --adapter "$A2" \
  --gpus 0,1 --batch-size 32 --limit $N --seed 42 --student1-input "$D/student1.jsonl" --output "$D/student2.jsonl" --resume
step post_validation_repair $PY scripts/repair_student2_failures.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" --seed 42 --gpu 0
step roll_post_validation flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --output "$D/rollouts_initial.jsonl" \
  --limit $N --workers 16 --rollout-id post_rl_round2seq --resume
step refine_post_validation flock "$LOCK" $PY scripts/refine_infeasible.py --config "$CFG" \
  --input "$D/rollouts_initial.jsonl" --output "$D/rollouts.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" \
  --level1-attempts 1 --workers 4 --gpus 0,1
step eval_post_validation $PY scripts/evaluate_population.py --ground-truth "$GT" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --rollouts "$D/rollouts.jsonl" \
  --output-dir "$OUT/analysis/post_rl_validation" --limit $N
NE="$V2/noendhome"
for ref in "Pre-RL:$NE/pre_rl_validation/rollouts.jsonl" "Round1:$NE/round1_validation/rollouts.jsonl" \
           "SeqBoost:$V2/seqboost/post_rl_validation/rollouts.jsonl"; do
  label="${ref%%:*}"
  step "boot_${label}_validation" $PY scripts/bootstrap_pair.py --truth "$GT" \
    --a "${ref#*:}" --b "$D/rollouts.jsonl" --labels "$label,Round2" --workers 8 \
    --output "$OUT/analysis/bootstrap_${label}_vs_round2_validation.json"
done
step figures_validation $PY scripts/visualize_population.py --config "$CFG" \
  --sft-dir "$DATA_V2/sft" --output "$OUT/analysis/distributions_validation" \
  --population "Pre-RL=$NE/pre_rl_validation/rollouts.jsonl" \
  --population "Round 1=$NE/round1_validation/rollouts.jsonl" \
  --population "SeqBoost=$V2/seqboost/post_rl_validation/rollouts.jsonl" \
  --population "Round 2=$D/rollouts.jsonl"
echo "v2 round 2 (SeqBoost, 3 branches) finished ($(date))"
}
main "$@"; exit $?
