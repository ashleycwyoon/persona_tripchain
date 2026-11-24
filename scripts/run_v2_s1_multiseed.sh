#!/usr/bin/env bash
# Extra Student1 validation answers (seeds 43, 44) for Pre-RL, SeqBoost, round2seq and round2bal.
# Together with the seed-42 answers of each run they give three samples per person, the input
# of per-person proper scoring rules (Brier / log score of the ground-truth activity sequence).
# The scoring itself is not part of this repository yet.
# Output: v2/s1_multiseed/<model>_seed<seed>.jsonl. Finished outputs are skipped; partial ones resume.
# Usage: bash scripts/run_v2_s1_multiseed.sh
#
# Needs GPUs 0 and 1 (two jobs per GPU). Run after scripts/run_v2_round2bal.sh.
# Environment: PYTHON (interpreter, default "python"), V2_ROOT (v2 output directory,
# default $OUTPUT_HOME/tripchain_1030/v2), INFERENCE_CPU_THREADS (default 4).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python}"
V2="${V2_ROOT:-${OUTPUT_HOME:-$HOME/outputs}/tripchain_1030/v2}"; OUT="$V2/s1_multiseed"; CFG=configs/framework_v2_validation.yaml; N=2653
export INFERENCE_CPU_THREADS="${INFERENCE_CPU_THREADS:-4}" PYTHONUNBUFFERED=1
mkdir -p "$OUT"
declare -A ADP=([prerl]="" [seqboost]="--adapter $V2/seqboost/grpo/student1/final"
                [round2seq]="--adapter $V2/round2seq/grpo/student1/final" [round2bal]="--adapter $V2/round2bal/grpo/student1/final")
run() {  # gpu model seed
  local o="$OUT/$2_seed$3.jsonl"
  [ -e "$o.done" ] && { echo "[skip] $2 $3"; return 0; }
  echo "[$(date +%m-%d\ %H:%M)] start $2 seed$3 gpu$1"
  $PY scripts/run_student_inference_2gpu.py student1 --config "$CFG" ${ADP[$2]} --gpus "$1" --batch-size 32 \
    --limit $N --seed "$3" --output "$o" --resume > "$o.log" 2>&1 || { echo "FAILED $2 $3: $o.log"; return 1; }
  touch "$o.done"; echo "[$(date +%m-%d\ %H:%M)] done  $2 seed$3"
}
# One job leaves most of a GPU idle, so each GPU runs two queues (4 queues in total).
( for j in "prerl 43" "round2seq 43"; do run 0 $j; done ) & P0=$!
( for j in "seqboost 43" "round2bal 43"; do run 0 $j; done ) & P1=$!
( for j in "prerl 44" "round2seq 44"; do run 1 $j; done ) & P2=$!
( for j in "seqboost 44" "round2bal 44"; do run 1 $j; done ) & P3=$!
wait $P0; wait $P1; wait $P2; wait $P3
echo "s1 multiseed finished ($(date))"
