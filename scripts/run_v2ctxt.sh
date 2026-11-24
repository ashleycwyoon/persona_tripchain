#!/usr/bin/env bash
# Context + context-Teacher run: as scripts/run_v2ctx.sh (the Students read household / student /
# work-status context), but the SFT targets come from a Teacher that also saw the context
# (scripts/teacher_ctx.py). The v2ctx targets were written without it and contradicted it
# (55% of the people in multi-member households were described as living alone).
# 0. Teacher (both GPUs, several hours): scripts/run_teacher_2gpu.py through scripts/teacher_ctx_python
#    on all v2 train + validation people; scripts/build_sft_v2ctxt.py -> processed_v2ctxt/sft
#    (user messages identical to processed_v2ctx/sft apart from the Teacher text, checked).
# 1-4. As run_v2ctx.sh with the v2ctxt configs. Stops before GRPO unless RUN_GRPO=1.
# Every step leaves v2ctxt/state/<name>.done; re-running skips finished steps and resumes the rest.
# Usage: bash scripts/run_v2ctxt.sh               (steps 0-2)
#        RUN_GRPO=1 bash scripts/run_v2ctxt.sh    (steps 3-4; finished steps are skipped)
#
# Run after scripts/build_context_v2.py and scripts/run_v2ctx.sh (its Pre-RL rollouts are a
# comparison population). The Teacher model is downloaded to $HF_HUB_CACHE when that is set.
# Needs GPUs 0 and 1. Run after scripts/run_v2_seqboost.sh (comparison populations and
# v2/noendhome) with GraphHopper running.
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (v2 output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), DATA_HOME, OUTPUT_HOME, INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
RUNS="${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030"; DATA="${DATA_HOME:-$HOME/data}/tripchain_1030"
V2="${V2_ROOT:-$RUNS/v2}"; OUT="$V2/v2ctxt"
TRAIN_CFG=configs/framework_grpo_train_v2ctxt.yaml; CFG=configs/framework_v2ctxt_validation.yaml
N=2653; N_TRAIN=18574; GT=$DATA/processed_v2/validation/ground_truth_tripmode.jsonl
SFT1="$RUNS/runs/student1_v2ctxt_sft/final"; SFT2="$RUNS/runs/student2_v2ctxt_sft/final"
UPD="$OUT/updates"; A1="$OUT/grpo/student1/final"; A2="$OUT/grpo/student2/final"
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"
PRE="$OUT/pre_rl_validation"; D="$OUT/post_rl_validation"
rm -f "$STATE/FAILED"; mkdir -p "$STATE" "$LOGS" "$UPD" "$OUT/grpo/student1" "$OUT/grpo/student2" "$OUT/analysis" "$PRE" "$D" "$V2/s1_multiseed"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1
grep -q "processed_v2ctx/train/demographics" "$TRAIN_CFG" || { echo "train config not on v2ctx"; exit 2; }
grep -q "processed_v2ctx/validation/demographics" "$CFG" || { echo "validation config not on v2ctx"; exit 2; }
grep -q "^    sequence: 1.64$" "$TRAIN_CFG" || { echo "train config not SeqBoost weights"; exit 2; }

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; touch "$STATE/FAILED"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
# Lanes wait on each other's markers; a failure anywhere stops the waiting lanes too.
wait_for() { while [ ! -e "$STATE/$1.done" ]; do [ -e "$STATE/FAILED" ] && return 1; sleep 30; done; }
s1() { $PY scripts/run_student_inference_2gpu.py student1 --config "$1" --gpus "$2" --batch-size 32 \
         --limit "$3" --seed "$4" --output "$5" --resume "${@:6}"; }
s2() { $PY scripts/run_student_inference_2gpu.py student2 --config "$1" --gpus "$2" --batch-size 32 \
         --limit "$3" --seed "$4" --student1-input "$5" --output "$6" --resume "${@:7}"; }
roll() { flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$1" --student1 "$2" --student2 "$3" \
           --output "$4" --limit "$5" --workers 16 --rollout-id "$6" --resume; }
echo "v2ctxt -> $OUT ($(date))"

# 0. Teacher with context, then SFT data.
TIN=$DATA/processed_v2ctx/teacher_input
TPREFIX=$RUNS/teacher_ctx/teacher_ctx; mkdir -p "$(dirname "$TPREFIX")"
teacher() { $PY scripts/run_teacher_2gpu.py --python "$PWD/scripts/teacher_ctx_python" --gpus 0,1 \
              --model Qwen/Qwen2.5-32B-Instruct-AWQ ${HF_HUB_CACHE:+--download-dir "$HF_HUB_CACHE"} --batch-size 32 \
              --gpu-memory-utilization 0.95 --max-new-tokens 900 --max-retries 10 --retry-temperature 0.2 \
              --demographics "$TIN/demographics.jsonl" --activity "$TIN/activity_sequence.jsonl" \
              --spatial "$TIN/location_dwell_mode_preference.jsonl" --output-prefix "$TPREFIX" --resume; }
step teacher_inputs $PY scripts/teacher_ctx.py --build-inputs
step teacher teacher
step build_sft $PY scripts/build_sft_v2ctxt.py --teacher-prefix "$TPREFIX"

# 1. SFT.
sft1() { $PY scripts/train_student1.py \
           --config configs/student1_v2ctxt_sft.yaml --gpu 0; }
sft2() { $PY scripts/train_student2_tripmode.py --config configs/student2_v2ctxt_sft.yaml --gpu 1; }

# 2. Train rollouts (4 inference lanes) and Pre-RL validation.
branch_s2() {  # gpu seed sample
  local dir="$OUT/grpo_train/s1_seed$2"; mkdir -p "$dir/s2_sample$3"
  wait_for "train_s1_$2"; wait_for sft_s2
  step "train_s2_$2_$3" s2 "$TRAIN_CFG" "$1" "$N_TRAIN" "$(($2 * 100 + $3))" "$dir/student1.jsonl" "$dir/s2_sample$3/student2.jsonl"
  step "roll_train_$2_$3" roll "$TRAIN_CFG" "$dir/student1.jsonl" "$dir/s2_sample$3/student2.jsonl" \
    "$dir/s2_sample$3/rollouts.jsonl" "$N_TRAIN" "train_s1-$2_s2-$3"
}
lane0() { step sft_s1 sft1; mkdir -p "$OUT/grpo_train/s1_seed42"
          step train_s1_42 s1 "$TRAIN_CFG" 0 "$N_TRAIN" 42 "$OUT/grpo_train/s1_seed42/student1.jsonl"; branch_s2 0 42 1; }
lane0b() { branch_s2 0 42 2; }
lane1() { step sft_s2 sft2; wait_for sft_s1; mkdir -p "$OUT/grpo_train/s1_seed43"
          step train_s1_43 s1 "$TRAIN_CFG" 1 "$N_TRAIN" 43 "$OUT/grpo_train/s1_seed43/student1.jsonl"; branch_s2 1 43 1; }
lane1b() { branch_s2 1 43 2; }
lane_pre() {
  wait_for sft_s1; step pre_s1 s1 "$CFG" 0 "$N" 42 "$PRE/student1.jsonl"
  for seed in 43 44; do step "pre_ms_$seed" s1 "$CFG" 0 "$N" "$seed" "$V2/s1_multiseed/v2ctxtpre_seed$seed.jsonl"; done
  wait_for sft_s2; step pre_s2 s2 "$CFG" 0 "$N" 42 "$PRE/student1.jsonl" "$PRE/student2.jsonl"
  step pre_repair $PY scripts/repair_student2_failures.py --config "$CFG" --student1 "$PRE/student1.jsonl" \
    --student2 "$PRE/student2.jsonl" --seed 42 --gpu 0
  step roll_pre roll "$CFG" "$PRE/student1.jsonl" "$PRE/student2.jsonl" "$PRE/rollouts_initial.jsonl" "$N" pre_rl_v2ctxt
  step refine_pre flock "$LOCK" $PY scripts/refine_infeasible.py --config "$CFG" --input "$PRE/rollouts_initial.jsonl" \
    --output "$PRE/rollouts.jsonl" --level1-attempts 1 --workers 4 --gpus 0,1
  step eval_pre_validation $PY scripts/evaluate_population.py --ground-truth "$GT" \
    --student1 "$PRE/student1.jsonl" --student2 "$PRE/student2.jsonl" --rollouts "$PRE/rollouts.jsonl" \
    --output-dir "$OUT/analysis/pre_rl_validation" --limit $N
}
lane0 & P1=$!; lane0b & P2=$!; lane1 & P3=$!; lane1b & P4=$!; lane_pre & P5=$!
for p in $P1 $P2 $P3 $P4 $P5; do wait "$p"; done
[ "${RUN_GRPO:-0}" = 1 ] || { echo "v2ctxt stopped before GRPO ($(date)); RUN_GRPO=1 to continue"; exit 0; }

# 3. Updates and GRPO.
step prep_updates $PY scripts/prepare_policy_updates.py "$OUT"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore
grpo() {  # student gpu adapter
  CUDA_VISIBLE_DEVICES="$2" $PY scripts/train_group_relative.py --updates "$UPD/$1_updates.jsonl" \
    --adapter "$3" --output "$OUT/grpo/$1" --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
    --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 \
  && rm -f "$OUT/grpo/$1"/checkpoint-*/optimizer.pt
}
( step grpo_student2 grpo student2 0 "$SFT2" ) & S2=$!
step grpo_student1 grpo student1 1 "$SFT1"
step post_validation_s1 s1 "$CFG" 1 "$N" 42 "$D/student1.jsonl" --adapter "$A1"
( step ms_43 s1 "$CFG" 1 "$N" 43 "$V2/s1_multiseed/v2ctxt_seed43.jsonl" --adapter "$A1" ) & M43=$!
step ms_44 s1 "$CFG" 1 "$N" 44 "$V2/s1_multiseed/v2ctxt_seed44.jsonl" --adapter "$A1"
wait "$M43"; wait "$S2"

# 4. Validation.
step post_validation_s2 s2 "$CFG" 0,1 "$N" 42 "$D/student1.jsonl" "$D/student2.jsonl" --adapter "$A2"
step post_validation_repair $PY scripts/repair_student2_failures.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" --seed 42 --gpu 0
step roll_post_validation roll "$CFG" "$D/student1.jsonl" "$D/student2.jsonl" "$D/rollouts_initial.jsonl" "$N" post_rl_v2ctxt
step refine_post_validation flock "$LOCK" $PY scripts/refine_infeasible.py --config "$CFG" \
  --input "$D/rollouts_initial.jsonl" --output "$D/rollouts.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" \
  --level1-attempts 1 --workers 4 --gpus 0,1
for which in post; do
  d="$OUT/${which}_rl_validation"
  step "eval_${which}_validation" $PY scripts/evaluate_population.py --ground-truth "$GT" \
    --student1 "$d/student1.jsonl" --student2 "$d/student2.jsonl" --rollouts "$d/rollouts.jsonl" \
    --output-dir "$OUT/analysis/${which}_rl_validation" --limit $N
done
NE="$V2/noendhome"
for ref in "Pre-RL:$NE/pre_rl_validation/rollouts.jsonl" "SeqBoost:$V2/seqboost/post_rl_validation/rollouts.jsonl" \
           "CtxPre-RL:$V2/v2ctx/pre_rl_validation/rollouts.jsonl" "CtxTPre-RL:$PRE/rollouts.jsonl"; do
  label="${ref%%:*}"
  step "boot_${label}_validation" $PY scripts/bootstrap_pair.py --truth "$GT" \
    --a "${ref#*:}" --b "$D/rollouts.jsonl" --labels "$label,CtxT" --workers 8 \
    --output "$OUT/analysis/bootstrap_${label}_vs_ctxt_validation.json"
done
step boot_SeqBoost_vs_CtxPre_validation $PY scripts/bootstrap_pair.py --truth "$GT" \
  --a "$V2/seqboost/post_rl_validation/rollouts.jsonl" --b "$PRE/rollouts.jsonl" --labels "SeqBoost,CtxTPre-RL" --workers 8 \
  --output "$OUT/analysis/bootstrap_SeqBoost_vs_ctxtpre_validation.json"
step figures_validation $PY scripts/visualize_population.py --config "$CFG" \
  --sft-dir "$DATA/processed_v2ctxt/sft" --output "$OUT/analysis/distributions_validation" \
  --population "Pre-RL=$NE/pre_rl_validation/rollouts.jsonl" \
  --population "SeqBoost=$V2/seqboost/post_rl_validation/rollouts.jsonl" \
  --population "Context Pre-RL=$V2/v2ctx/pre_rl_validation/rollouts.jsonl" \
  --population "Context+Teacher Pre-RL=$PRE/rollouts.jsonl" \
  --population "Context+Teacher GRPO=$D/rollouts.jsonl"
echo "v2ctxt finished ($(date))"
}
main "$@"; exit $?
