#!/bin/bash
# Baseline reproduction eval — Qwen3-1.7B, 3 benchmarks (AIME24 / AIME25 / HMMT25), Avg@12.
# EXP_DIR must match the training output_dir + run_config from run_opsd_1b_baseline.sh.
# Run from the script's own dir so evaluate_math.py resolves regardless of CWD.
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_MODEL="Qwen/Qwen3-1.7B"
EXP_DIR="/workspace/outputs/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"

# Overridable so eval can run on whatever is free and target specific checkpoints:
#   EVAL_GPUS=0,1,2 STEPS="25 50 75 100" RUN_BASE=1 bash run_eval_baseline.sh
# Defaults: checkpoint-100 only, plus the base model for reference.
#
# Qwen3-1.7B is ~3.4GB in bf16 and fits on one 80GB GPU, so tensor parallelism buys no
# memory here and only adds two all-reduces per layer per decode step. TP therefore
# defaults to 1 and the GPUs are spent on data parallelism instead: one eval process per
# GPU, jobs round-robined across them. Set TP>1 to fall back to a single process
# spanning all of EVAL_GPUS.
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
TP="${TP:-1}"
STEPS="${STEPS:-100}"
RUN_BASE="${RUN_BASE:-1}"
LOG_DIR="${LOG_DIR:-eval_logs}"

mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPU_ARR <<<"$EVAL_GPUS"
NUM_GPUS=${#GPU_ARR[@]}

# $1 = GPU(s) for this process, $2 = dataset, $3 = "base" or a checkpoint step
run_one() {
    local gpus="$1" ds="$2" step="$3"
    local log="$LOG_DIR/${ds}_${step}.log"
    local ckpt_args=()
    [[ "$step" != "base" ]] && ckpt_args=(--checkpoint_dir "$EXP_DIR/checkpoint-$step")

    echo ">> [gpu $gpus] $ds step=$step -> $log"
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="$gpus" python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "$ds" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size "$TP" \
        "${ckpt_args[@]}" >"$log" 2>&1
}

# One job per (checkpoint, benchmark). The benchmarks differ ~40% in generated tokens,
# and a plain round-robin over 3 GPUs would pin one benchmark per GPU, so rotate the
# benchmark order per checkpoint to spread the expensive one across workers.
DATASETS=(aime24 aime25 hmmt25)
STEP_LIST=()
[[ "$RUN_BASE" = "1" ]] && STEP_LIST+=(base)
for step in $STEPS; do STEP_LIST+=("$step"); done

JOBS=()
rot=0
for step in "${STEP_LIST[@]}"; do
    for ((d = 0; d < ${#DATASETS[@]}; d++)); do
        JOBS+=("${DATASETS[$(((d + rot) % ${#DATASETS[@]}))]}:$step")
    done
    rot=$((rot + 1))
done

echo ">> ${#JOBS[@]} eval runs | steps='${STEP_LIST[*]}' | gpus=$EVAL_GPUS | TP=$TP | logs=$LOG_DIR/"

if [[ "$TP" -gt 1 ]]; then
    # Tensor-parallel: one process at a time spanning all listed GPUs.
    for job in "${JOBS[@]}"; do
        run_one "$EVAL_GPUS" "${job%%:*}" "${job##*:}"
    done
else
    # Data-parallel: one worker per GPU, each running its share of the jobs serially.
    for ((i = 0; i < NUM_GPUS; i++)); do
        (
            for ((j = i; j < ${#JOBS[@]}; j += NUM_GPUS)); do
                job="${JOBS[$j]}"
                run_one "${GPU_ARR[$i]}" "${job%%:*}" "${job##*:}"
            done
        ) &
    done
    wait
fi

echo ">> all eval runs finished. results: eval_results/  logs: $LOG_DIR/"
