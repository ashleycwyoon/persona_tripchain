#!/usr/bin/env bash
# Student2 retrained with cell-centered distribution rewards at individual : distribution = 50:50,
# on the context + context-Teacher run. The Student2 of scripts/run_v2ctxt.sh was trained while
# the distribution terms were centered within the group; with the cell-centered terms its
# weights (0.94 / 0.43 / 1.04) leave the individual rewards 10% of the advantage variance.
# configs/framework_grpo_train_v2ctxt_s2w.yaml holds the re-measured weights.
# 1. Updates from the existing v2ctxt train rollouts (re-scored, nothing is generated again);
#    Student2 GRPO from the v2ctxt SFT adapter.
# 2. Validation on the Student1 answers of the v2ctxt run (same seed-42 file, so only Student2
#    differs): Student2 -> repair -> rollouts -> refinement -> evaluation, paired bootstraps, figures.
# Every step leaves <NAME>/state/<name>.done; re-running skips finished steps.
# Usage: bash scripts/run_v2ctxt_s2cell.sh
#        NAME=v2ctxt_s1s2cell S1_RUN=v2ctxt_s1cell bash scripts/run_v2ctxt_s2cell.sh
# The second form trains nothing: it evaluates the Student2 adapter of v2/v2ctxt_s2cell on the
# Student1 answers (and with the Student1 adapter) of v2/$S1_RUN, into v2/$NAME.
#
# Run after RUN_GRPO=1 scripts/run_v2ctxt.sh, with GraphHopper running.
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (v2 output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), DATA_HOME, OUTPUT_HOME, INFERENCE_CPU_THREADS (default 4),
# GRPO_GPU (training GPU, default 0), VAL_GPUS (validation GPUs, default 0,1), S1_WEIGHT (legend only).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
RUNS="${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030"; DATA="${DATA_HOME:-$HOME/data}/tripchain_1030"
NAME="${NAME:-v2ctxt_s2cell}"; BASE=v2ctxt_s2cell; S1_RUN="${S1_RUN:-v2ctxt}"
V2="${V2_ROOT:-$RUNS/v2}"; SRC="$V2/v2ctxt"; S1="$V2/$S1_RUN"; OUT="$V2/$NAME"
TRAIN_CFG=configs/framework_grpo_train_v2ctxt_s2w.yaml; CFG=configs/framework_v2ctxt_validation.yaml
N=2653; GT=$DATA/processed_v2/validation/ground_truth_tripmode.jsonl
SFT2="$RUNS/runs/student2_v2ctxt_sft/final"
UPD="$OUT/updates"; A1="$S1/grpo/student1/final"; A2="$V2/$BASE/grpo/student2/final"
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"; D="$OUT/post_rl_validation"
ROLLS=("$SRC"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl)
[ "${#ROLLS[@]}" -eq 4 ] && [ -e "${ROLLS[0]}" ] || { echo "expected 4 v2ctxt train rollout files, got ${#ROLLS[@]}"; exit 2; }
for need in "$SFT2/adapter_config.json" "$A1/adapter_config.json" "$S1/post_rl_validation/student1.jsonl" \
            "$SRC/post_rl_validation/rollouts.jsonl" "$S1/post_rl_validation/rollouts.jsonl" "$V2/seqboost/post_rl_validation/rollouts.jsonl"; do
  [ -e "$need" ] || { echo "missing $need"; exit 2; }
done
rm -f "$STATE/FAILED"; mkdir -p "$STATE" "$LOGS" "$OUT/analysis" "$D"
VAL_GPUS="${VAL_GPUS:-0,1}"; G0="${VAL_GPUS%%,*}"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; touch "$STATE/FAILED"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
echo "$NAME (Student1 of $S1_RUN) -> $OUT ($(date))"

# 1. Updates and Student2 GRPO.
if [ "$NAME" = "$BASE" ]; then
mkdir -p "$UPD" "$OUT/grpo/student2"
step prep_updates $PY scripts/prepare_policy_updates.py "${ROLLS[@]}" \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore
grep -q '"distribution_baseline": "demographic_cell"' "$LOGS/prep_updates.log" \
  || { echo "updates were not made with cell-centered distribution rewards"; exit 2; }
grpo_student2() {
  CUDA_VISIBLE_DEVICES="${GRPO_GPU:-0}" $PY scripts/train_group_relative.py --updates "$UPD/student2_updates.jsonl" \
    --adapter "$SFT2" --output "$OUT/grpo/student2" --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
    --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 \
  && rm -f "$OUT/grpo/student2"/checkpoint-*/optimizer.pt
}
step grpo_student2 grpo_student2
else
  [ -e "$A2/adapter_config.json" ] || { echo "missing $A2 (run scripts/run_v2ctxt_s2cell.sh first)"; exit 2; }
fi

# 2. Validation on the Student1 answers of the $S1_RUN run.
[ -e "$D/student1.jsonl" ] || cp "$S1/post_rl_validation/student1.jsonl" "$D/student1.jsonl"
step post_validation_s2 $PY scripts/run_student_inference_2gpu.py student2 --config "$CFG" --gpus "$VAL_GPUS" --batch-size 32 \
  --limit "$N" --seed 42 --student1-input "$D/student1.jsonl" --output "$D/student2.jsonl" --resume --adapter "$A2"
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
REFS=("SeqBoost:$V2/seqboost/post_rl_validation/rollouts.jsonl" "PersonCentered:$SRC/post_rl_validation/rollouts.jsonl" \
      "PreRL:$SRC/pre_rl_validation/rollouts.jsonl")
POPS=(); LABEL="Cell-centered Student2"
if [ "$S1_RUN" != v2ctxt ]; then   # same Student1 answers with the earlier Student2
  REFS+=("SameStudent1:$S1/post_rl_validation/rollouts.jsonl")
  W="${S1_WEIGHT:-1.64}"   # sequence weight the Student1 of $S1_RUN was trained with (figure legend only)
  POPS=(--population "Cell-centered Student1 w$W=$S1/post_rl_validation/rollouts.jsonl"); LABEL="Cell-centered Student1 w$W + Student2"
fi
for ref in "${REFS[@]}"; do
  label="${ref%%:*}"
  step "boot_${label}_validation" $PY scripts/bootstrap_pair.py --truth "$GT" \
    --a "${ref#*:}" --b "$D/rollouts.jsonl" --labels "$label,CellCenteredS2" --workers 8 \
    --output "$OUT/analysis/bootstrap_${label}_vs_cellcentered_s2_validation.json"
done
step figures_validation $PY scripts/visualize_population.py --config "$CFG" \
  --sft-dir "$DATA/processed_v2ctxt/sft" --output "$OUT/analysis/distributions_validation" \
  --population "SeqBoost=$V2/seqboost/post_rl_validation/rollouts.jsonl" \
  --population "Context+Teacher Pre-RL=$SRC/pre_rl_validation/rollouts.jsonl" \
  --population "Person-centered GRPO=$SRC/post_rl_validation/rollouts.jsonl" \
  "${POPS[@]}" --population "$LABEL=$D/rollouts.jsonl"
echo "$NAME finished ($(date))"
}
main "$@"; exit $?
