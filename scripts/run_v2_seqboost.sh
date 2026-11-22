#!/usr/bin/env bash
# SeqBoost: Student1 sequence-pattern weight 1.64 plus the trip-distance Student2 reward
# (1.04), configs/framework_grpo_train_v2_seqboost.yaml. Both Students are retrained
# from the v2 SFT adapters; evaluation on the validation split only.
# - require_end_home is off: the v2 train rollouts that failed END_ACTIVITY_NOT_HOME
#   are rolled again under the new environment and merged (all others are reused),
# - Student1 (GPU1) and Student2 (GPU0) are trained side by side from the v2 SFT adapters,
# - then validation: Student1 -> Student2 (both GPUs) -> repair -> rollouts -> refinement
#   -> evaluation, bootstraps and figures against the no-end-home baselines (v2/noendhome).
# Every step leaves seqboost/state/<name>.done; re-running skips finished steps.
# Usage: bash scripts/run_v2_seqboost.sh
#
# Needs GPUs 0 and 1. Run after scripts/run_v2_grpo.sh (the v2 train rollouts in
# v2/grpo_train) and scripts/run_v2_noendhome_baselines.sh.
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (v2 output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
main() {
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
V2="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"
DATA_V2="${DATA_HOME:-$HOME/data}/tripchain_1030/processed_v2"
OUT="$V2/seqboost"; TRAIN_CFG=configs/framework_grpo_train_v2_seqboost.yaml
CFG=configs/framework_v2_validation.yaml; N=2653
GT=$DATA_V2/validation/ground_truth_tripmode.jsonl
UPD="$OUT/updates"; A1="$OUT/grpo/student1/final"; A2="$OUT/grpo/student2/final"
STATE="$OUT/state"; LOGS="$OUT/logs"; LOCK="$V2/.rollout.lock"; D="$OUT/post_rl_validation"
mkdir -p "$STATE" "$LOGS" "$UPD" "$OUT/grpo/student1" "$OUT/grpo/student2" "$OUT/analysis" "$D"
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1
for cfg in "$CFG" "$TRAIN_CFG"; do
  grep -q "^  require_end_home: false" "$cfg" || { echo "require_end_home is not off in $cfg"; exit 2; }
done

step() {
  local name="$1"; shift
  [ -e "$STATE/$name.done" ] && { echo "[skip] $name"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $name"
  "$@" > "$LOGS/$name.log" 2>&1 || { echo "FAILED $name: $LOGS/$name.log"; return 1; }
  touch "$STATE/$name.done"; echo "[$(date +%m-%d\ %H:%M)] done  $name"
}
echo "v2 seqboost -> $OUT ($(date))"

# 1. Re-roll the END_ACTIVITY_NOT_HOME train rollouts and merge.
for s1 in 42 43; do for s2 in 1 2; do
  src="$V2/grpo_train/s1_seed$s1/s2_sample$s2"; dst="$OUT/grpo_train/s1_seed$s1/s2_sample$s2"; mkdir -p "$dst/reroll"
  step "subset_${s1}_$s2" $PY -c "
import json, sys
src, s1file, dst = sys.argv[1:]
rows = [json.loads(l) for l in open(src + '/rollouts.jsonl') if l.strip()]
bad = {str(r['person_id']) for r in rows
       if str(((r.get('decision_trace') or {}).get('environment') or {}).get('failure_reason', '')).startswith('ValueError:END_ACTIVITY_NOT_HOME')}
for name, path in (('student1', s1file), ('student2', src + '/student2.jsonl')):
    with open(f'{dst}/reroll/{name}.jsonl', 'w') as out:
        for l in open(path):
            if l.strip() and str(json.loads(l)['person_id']) in bad:
                out.write(l)
print('re-roll', len(bad))" "$src" "$V2/grpo_train/s1_seed$s1/student1.jsonl" "$dst"
  n=$(wc -l < "$dst/reroll/student1.jsonl")
  step "reroll_${s1}_$s2" flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$TRAIN_CFG" \
    --student1 "$dst/reroll/student1.jsonl" --student2 "$dst/reroll/student2.jsonl" \
    --output "$dst/reroll/rollouts.jsonl" --limit "$n" --workers 16 --rollout-id "train_s1-${s1}_s2-$s2" --resume
  step "merge_${s1}_$s2" $PY -c "
import json, sys
src, dst = sys.argv[1:]
new = {str(r['person_id']): r for r in (json.loads(l) for l in open(dst + '/reroll/rollouts.jsonl') if l.strip())}
kept = replaced = 0
with open(dst + '/rollouts.jsonl', 'w') as out:
    for l in open(src + '/rollouts.jsonl'):
        if not l.strip(): continue
        r = json.loads(l); pid = str(r['person_id'])
        if pid in new: r = new.pop(pid); replaced += 1
        else: kept += 1
        out.write(json.dumps(r, ensure_ascii=False) + '\n')
assert not new, f'{len(new)} re-rolled people missing from the source'
print('kept', kept, 'replaced', replaced)" "$src" "$dst"
done; done

# 2. Updates (rescored with the seqboost config); Student2 GRPO on GPU0 and Student1 on GPU1.
step prep_updates $PY scripts/prepare_policy_updates.py "$OUT"/grpo_train/s1_seed*/s2_sample*/rollouts.jsonl \
  --output-dir "$UPD" --min-group-size 2 --config "$TRAIN_CFG" --rescore
SFT2="$($PY -c "import sys; sys.path.insert(0,'.'); from tripchain.common import load_config; print(load_config('$TRAIN_CFG')['paths']['student2_adapter'])")"
( step grpo_student2 sh -c "CUDA_VISIBLE_DEVICES=0 $PY scripts/train_group_relative.py --updates $UPD/student2_updates.jsonl \
  --adapter $SFT2 --output $OUT/grpo/student2 --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
  --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 && rm -f $OUT/grpo/student2/checkpoint-*/optimizer.pt" ) & S2=$!
SFT1="$($PY -c "import sys; sys.path.insert(0,'.'); from tripchain.common import load_config; print(load_config('$TRAIN_CFG')['paths']['student1_adapter'])")"
step grpo_student1 sh -c "CUDA_VISIBLE_DEVICES=1 $PY scripts/train_group_relative.py --updates $UPD/student1_updates.jsonl \
  --adapter $SFT1 --output $OUT/grpo/student1 --epochs 1.0 --max-steps -1 --learning-rate 1e-5 \
  --reference-penalty 0.02 --save-steps 0 --batch-size 2 --grad-accum 16 && rm -f $OUT/grpo/student1/checkpoint-*/optimizer.pt"
step post_validation_s1 $PY scripts/run_student_inference_2gpu.py student1 --config "$CFG" --adapter "$A1" \
  --gpus 1 --batch-size 32 --limit $N --seed 42 --output "$D/student1.jsonl" --resume

# 3. Wait for Student2 training (GPU0).
wait "$S2"

# 4. Validation with both GPUs.
step post_validation_s2 $PY scripts/run_student_inference_2gpu.py student2 --config "$CFG" --adapter "$A2" \
  --gpus 0,1 --batch-size 32 --limit $N --seed 42 --student1-input "$D/student1.jsonl" --output "$D/student2.jsonl" --resume
step post_validation_repair $PY scripts/repair_student2_failures.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" --seed 42 --gpu 0
step roll_post_validation flock "$LOCK" $PY scripts/run_rollouts_parallel.py --config "$CFG" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --output "$D/rollouts_initial.jsonl" \
  --limit $N --workers 16 --rollout-id post_rl_seqboost --resume
step refine_post_validation flock "$LOCK" $PY scripts/refine_infeasible.py --config "$CFG" \
  --input "$D/rollouts_initial.jsonl" --output "$D/rollouts.jsonl" --student1-adapter "$A1" --student2-adapter "$A2" \
  --level1-attempts 1 --workers 4 --gpus 0,1
step eval_post_validation $PY scripts/evaluate_population.py --ground-truth "$GT" \
  --student1 "$D/student1.jsonl" --student2 "$D/student2.jsonl" --rollouts "$D/rollouts.jsonl" \
  --output-dir "$OUT/analysis/post_rl_validation" --limit $N
NE="$V2/noendhome"
for ref in "Pre-RL:$NE/pre_rl_validation/rollouts.jsonl" "Round1:$NE/round1_validation/rollouts.jsonl" \
           "DistMode0.94:$NE/distmode_w0.94_validation/rollouts.jsonl"; do
  label="${ref%%:*}"
  step "boot_${label}_validation" $PY scripts/bootstrap_pair.py --truth "$GT" \
    --a "${ref#*:}" --b "$D/rollouts.jsonl" --labels "$label,SeqBoost" --workers 8 \
    --output "$OUT/analysis/bootstrap_${label}_vs_seqboost_validation.json"
done
step figures_validation $PY scripts/visualize_population.py --config "$CFG" \
  --sft-dir "$DATA_V2/sft" --output "$OUT/analysis/distributions_validation" \
  --population "Pre-RL=$NE/pre_rl_validation/rollouts.jsonl" \
  --population "Round 1=$NE/round1_validation/rollouts.jsonl" \
  --population "Distance mode 0.94=$NE/distmode_w0.94_validation/rollouts.jsonl" \
  --population "SeqBoost=$D/rollouts.jsonl"
echo "v2 seqboost finished ($(date))"
}
main "$@"; exit $?
