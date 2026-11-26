#!/usr/bin/env bash
# Student1 retrained with cell-centered distribution rewards, on the context + context-Teacher run.
# scripts/prepare_policy_updates.py now centers the distribution terms within the demographic
# cell instead of within the person, so a person whose samples all share one over-represented
# sequence still gets a learning signal. The run of scripts/run_v2ctxt.sh was trained before
# that change. This run repeats only its Student1 GRPO with the new updates:
# 1. Updates from the existing v2ctxt train rollouts (re-scored, nothing is generated again);
#    Student1 GRPO from the v2ctxt SFT adapter on GPU1. Student2 is the v2ctxt GRPO adapter.
# 2. Validation: Student1 (seeds 42, 43, 44) -> Student2 -> repair -> rollouts -> refinement
#    -> evaluation, paired bootstraps, sequence scores (scripts/score_sequences.py), figures.
# Every step leaves <NAME>/state/<name>.done; re-running skips finished steps.
# Usage: bash scripts/run_v2ctxt_s1cell.sh                 (sequence weight 1.64, v2/v2ctxt_s1cell)
#        NAME=v2ctxt_s1cell_w0.53 TRAIN_CFG=configs/framework_grpo_train_v2ctxt_s1w0.53.yaml \
#          bash scripts/run_v2ctxt_s1cell.sh               (weight 0.53: individual : distribution 50:50)
# A run with another NAME is also compared with v2/v2ctxt_s1cell. GRPO_GPU (default 1) is the
# training GPU and VAL_GPUS (default 0,1) the validation GPUs, e.g. GRPO_GPU=1 VAL_GPUS=1 to
# leave GPU 0 to another job. LABEL replaces the legend name "Cell-centered Student1 w<weight>".
#
# Run after RUN_GRPO=1 scripts/run_v2ctxt.sh, with GraphHopper running. Needs GPUs 0 and 1.
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (v2 output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), DATA_HOME, OUTPUT_HOME, INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
RUNS="${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030"; DATA="${DATA_HOME:-$HOME/data}/tripchain_1030"
NAME="${NAME:-v2ctxt_s1cell}"; BASE=v2ctxt_s1cell
V2="${V2_ROOT:-$RUNS/v2}"; SRC="$V2/v2ctxt"; OUT="$V2/$NAME"; MS="$V2/s1_multiseed"
TRAIN_CFG="${TRAIN_CFG:-configs/framework_grpo_train_v2ctxt.yaml}"; CFG=configs/framework_v2ctxt_validation.yaml
N=2653; GT=$DATA/processed_v2/validation/ground_truth_tripmode.jsonl
SFT1="$RUNS/runs/student1_v2ctxt_sft/final"
UPD="$OUT/updates"; A1="$OUT/grpo/student1/final"; A2="$SRC/grpo/student2/final"
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"; D="$OUT/post_rl_validation"
ROLLS=("$SRC"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl)
[ "${#ROLLS[@]}" -eq 4 ] && [ -e "${ROLLS[0]}" ] || { echo "expected 4 v2ctxt train rollout files, got ${#ROLLS[@]}"; exit 2; }
for need in "$SFT1/adapter_config.json" "$A2/adapter_config.json" "$SRC/post_rl_validation/rollouts.jsonl" \
            "$V2/seqboost/post_rl_validation/rollouts.jsonl" "$MS/v2ctxt_seed44.jsonl" "$MS/seqboost_seed44.jsonl"; do
  [ -e "$need" ] || { echo "missing $need"; exit 2; }
done
rm -f "$STATE/FAILED"; mkdir -p "$STATE" "$LOGS" "$UPD" "$OUT/grpo/student1" "$OUT/analysis" "$D" "$MS"
VAL_GPUS="${VAL_GPUS:-0,1}"; G0="${VAL_GPUS%%,*}"; G1="${VAL_GPUS##*,}"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; touch "$STATE/FAILED"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
s1() { $PY scripts/run_student_inference_2gpu.py student1 --config "$CFG" --gpus "$1" --batch-size 32 \
         --limit "$N" --seed "$2" --output "$3" --resume --adapter "$A1"; }
W=$(sed -n 's/^    sequence: //p' "$TRAIN_CFG"); MODELS=(); POPS=(); LABEL="${LABEL:-Cell-centered Student1 w$W}"
if [ "$NAME" != "$BASE" ]; then
  for need in "$V2/$BASE/post_rl_validation/rollouts.jsonl" "$MS/${BASE}_seed44.jsonl"; do
    [ -e "$need" ] || { echo "missing $need"; exit 2; }
  done
  MODELS=(--model "Cell-centered Student1 w1.64=$V2/$BASE/post_rl_validation/student1.jsonl,$MS/${BASE}_seed43.jsonl,$MS/${BASE}_seed44.jsonl")
  POPS=(--population "Cell-centered Student1 w1.64=$V2/$BASE/post_rl_validation/rollouts.jsonl")
fi
echo "$NAME ($TRAIN_CFG) -> $OUT ($(date))"

# 1. Updates and Student1 GRPO.
step prep_updates $PY scripts/prepare_policy_updates.py "${ROLLS[@]}" \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore
grep -q '"distribution_baseline": "demographic_cell"' "$LOGS/prep_updates.log" \
  || { echo "updates were not made with cell-centered distribution rewards"; exit 2; }
grpo_student1() {
  CUDA_VISIBLE_DEVICES="${GRPO_GPU:-1}" $PY scripts/train_group_relative.py --updates "$UPD/student1_updates.jsonl" \
    --adapter "$SFT1" --output "$OUT/grpo/student1" --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
    --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 \
  && rm -f "$OUT/grpo/student1"/checkpoint-*/optimizer.pt
}
step grpo_student1 grpo_student1

# 2. Validation. Seeds 43 and 44 run on the last validation GPU while the seed-42 chain uses the first.
step post_validation_s1 s1 "$VAL_GPUS" 42 "$D/student1.jsonl"
( step ms_43 s1 "$G1" 43 "$MS/${NAME}_seed43.jsonl" && step ms_44 s1 "$G1" 44 "$MS/${NAME}_seed44.jsonl" ) & SEEDS=$!
step post_validation_s2 $PY scripts/run_student_inference_2gpu.py student2 --config "$CFG" --gpus "$G0" --batch-size 32 \
  --limit "$N" --seed 42 --student1-input "$D/student1.jsonl" --output "$D/student2.jsonl" --resume --adapter "$A2"
wait "$SEEDS"
step post_validation_repair $PY scripts/repair_student2_failures.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" --seed 42 --gpu "$G0"
step roll_post_validation flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$CFG" --student1 "$D/student1.jsonl" \
  --student2 "$D/student2.jsonl" --output "$D/rollouts_initial.jsonl" --limit "$N" --workers 16 \
  --rollout-id "post_rl_$NAME" --resume
step refine_post_validation flock "$LOCK" $PY scripts/refine_infeasible.py --config "$CFG" \
  --input "$D/rollouts_initial.jsonl" --output "$D/rollouts.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" \
  --level1-attempts 1 --workers 4 --gpus "$VAL_GPUS"
step eval_post_validation $PY scripts/evaluate_population.py --ground-truth "$GT" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --rollouts "$D/rollouts.jsonl" \
  --output-dir "$OUT/analysis/post_rl_validation" --limit $N
REFS=("SeqBoost:$V2/seqboost/post_rl_validation/rollouts.jsonl" "PersonCentered:$SRC/post_rl_validation/rollouts.jsonl")
[ "$NAME" = "$BASE" ] || REFS+=("CellCenteredW1.64:$V2/$BASE/post_rl_validation/rollouts.jsonl")
for ref in "${REFS[@]}"; do
  label="${ref%%:*}"
  step "boot_${label}_validation" $PY scripts/bootstrap_pair.py --truth "$GT" \
    --a "${ref#*:}" --b "$D/rollouts.jsonl" --labels "$label,CellCentered" --workers 8 \
    --output "$OUT/analysis/bootstrap_${label}_vs_cellcentered_validation.json"
done
step sequence_scores $PY scripts/score_sequences.py --config "$CFG" --reference "Person-centered GRPO" \
  --model "Pre-RL=$SRC/pre_rl_validation/student1.jsonl,$MS/v2ctxtpre_seed43.jsonl,$MS/v2ctxtpre_seed44.jsonl" \
  --model "SeqBoost=$V2/seqboost/post_rl_validation/student1.jsonl,$MS/seqboost_seed43.jsonl,$MS/seqboost_seed44.jsonl" \
  --model "Person-centered GRPO=$SRC/post_rl_validation/student1.jsonl,$MS/v2ctxt_seed43.jsonl,$MS/v2ctxt_seed44.jsonl" \
  "${MODELS[@]}" --model "$LABEL=$D/student1.jsonl,$MS/${NAME}_seed43.jsonl,$MS/${NAME}_seed44.jsonl" \
  --output "$OUT/analysis/sequence_scores_validation.json"
step figures_validation $PY scripts/visualize_population.py --config "$CFG" \
  --sft-dir "$DATA/processed_v2ctxt/sft" --output "$OUT/analysis/distributions_validation" \
  --population "SeqBoost=$V2/seqboost/post_rl_validation/rollouts.jsonl" \
  --population "Context+Teacher Pre-RL=$SRC/pre_rl_validation/rollouts.jsonl" \
  --population "Person-centered GRPO=$SRC/post_rl_validation/rollouts.jsonl" \
  "${POPS[@]}" --population "$LABEL=$D/rollouts.jsonl"
echo "$NAME finished ($(date))"
}
main "$@"; exit $?
