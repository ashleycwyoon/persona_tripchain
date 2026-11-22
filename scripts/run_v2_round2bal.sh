#!/usr/bin/env bash
# Balanced GRPO round 2 (configs/framework_grpo_train_v2_round2bal.yaml), validation only.
# Reuses the round2seq train rollouts (SeqBoost adapters, 3 Student1 branches x 2 Student2
# samples, all 18,574 v2 train people): the policy that generated them is the one trained
# here, so only the rewards change. prepare_policy_updates.py --rescore applies the lower
# distribution weights (sequence 1.20, trip_mode 0.80, first_departure 0.60, trip_distance 1.04).
# 1. Updates; Student2 GRPO on GPU0 and Student1 on GPU1 from (and referenced to) SeqBoost.
# 2. Validation: Student1 -> Student2 -> repair -> rollouts -> refinement -> evaluation,
#    bootstraps against Pre-RL, Round 1, SeqBoost and round2seq, figures.
# Every step leaves round2bal/state/<name>.done; re-running skips finished steps.
# Usage: bash scripts/run_v2_round2bal.sh
#
# Needs GPUs 0 and 1. Run after scripts/run_v2_round2seq.sh. The train config reads the
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
OUT="$V2/round2bal"; TRAIN_CFG=configs/framework_grpo_train_v2_round2bal.yaml
CFG=configs/framework_v2_validation.yaml; N=2653
GT=$DATA_V2/validation/ground_truth_tripmode.jsonl
SB="$V2/seqboost/grpo"; R2S="$V2/round2seq"; UPD="$OUT/updates"
A1="$OUT/grpo/student1/final"; A2="$OUT/grpo/student2/final"
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
grep -q "^    sequence: 1.20$" "$TRAIN_CFG" || { echo "train config not balanced"; exit 2; }
ROLLS=("$R2S"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl)
[ "${#ROLLS[@]}" -eq 6 ] || { echo "expected 6 round2seq rollout files, got ${#ROLLS[@]}"; exit 2; }
for seed in 42 43 44; do for sample in 1 2; do
  [ -e "$R2S/state/roll_train_${seed}_$sample.done" ] || { echo "round2seq roll_train_${seed}_$sample not done"; exit 2; }
done; done

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
echo "v2 balanced round 2 -> $OUT ($(date))"

# 1. Updates from the round2seq rollouts with the balanced weights; GRPO from SeqBoost.
step prep_updates $PY scripts/prepare_policy_updates.py "${ROLLS[@]}" \
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

# 2. Validation with both GPUs.
step post_validation_s2 $PY scripts/run_student_inference_2gpu.py student2 --config "$CFG" --adapter "$A2" \
  --gpus 0,1 --batch-size 32 --limit $N --seed 42 --student1-input "$D/student1.jsonl" --output "$D/student2.jsonl" --resume
step post_validation_repair $PY scripts/repair_student2_failures.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" --seed 42 --gpu 0
step roll_post_validation flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --output "$D/rollouts_initial.jsonl" \
  --limit $N --workers 16 --rollout-id post_rl_round2bal --resume
step refine_post_validation flock "$LOCK" $PY scripts/refine_infeasible.py --config "$CFG" \
  --input "$D/rollouts_initial.jsonl" --output "$D/rollouts.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" \
  --level1-attempts 1 --workers 4 --gpus 0,1
step eval_post_validation $PY scripts/evaluate_population.py --ground-truth "$GT" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --rollouts "$D/rollouts.jsonl" \
  --output-dir "$OUT/analysis/post_rl_validation" --limit $N
NE="$V2/noendhome"
for ref in "Pre-RL:$NE/pre_rl_validation/rollouts.jsonl" "Round1:$NE/round1_validation/rollouts.jsonl" \
           "SeqBoost:$V2/seqboost/post_rl_validation/rollouts.jsonl" \
           "Round2seq:$R2S/post_rl_validation/rollouts.jsonl"; do
  label="${ref%%:*}"
  step "boot_${label}_validation" $PY scripts/bootstrap_pair.py --truth "$GT" \
    --a "${ref#*:}" --b "$D/rollouts.jsonl" --labels "$label,Round2bal" --workers 8 \
    --output "$OUT/analysis/bootstrap_${label}_vs_round2bal_validation.json"
done
step figures_validation $PY scripts/visualize_population.py --config "$CFG" \
  --sft-dir "$DATA_V2/sft" --output "$OUT/analysis/distributions_validation" \
  --population "Pre-RL=$NE/pre_rl_validation/rollouts.jsonl" \
  --population "SeqBoost=$V2/seqboost/post_rl_validation/rollouts.jsonl" \
  --population "Round 2 (seq)=$R2S/post_rl_validation/rollouts.jsonl" \
  --population "Round 2 (bal)=$D/rollouts.jsonl"
echo "v2 balanced round 2 finished ($(date))"
}
main "$@"; exit $?
