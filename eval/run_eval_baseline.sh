#!/bin/bash
# Baseline reproduction eval — Qwen3-1.7B, 3 benchmarks (AIME24 / AIME25 / HMMT25), Avg@12.
# EXP_DIR must match the training output_dir + run_config from run_opsd_1b_baseline.sh.
# Run from the script's own dir so evaluate_math.py resolves regardless of CWD.
cd "$(dirname "${BASH_SOURCE[0]}")"
BASE_MODEL="Qwen/Qwen3-1.7B"
EXP_DIR="/workspace/outputs/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"

# Overridable so eval can run on whatever is free and target specific checkpoints:
#   EVAL_GPUS=0,1 TP=2 STEPS=100 RUN_BASE=0 bash run_eval_baseline.sh
# Defaults: eval only checkpoint-100, plus the base model for reference.
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
TP="${TP:-4}"
STEPS="${STEPS:-100}"
RUN_BASE="${RUN_BASE:-1}"

# base model performance (per benchmark)
if [ "$RUN_BASE" = "1" ]; then
for ds in aime24 aime25 hmmt25; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$EVAL_GPUS python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "$ds" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size $TP
    wait
done
fi

# trained checkpoints
for step in $STEPS; do
    for ds in aime24 aime25 hmmt25; do
        NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$EVAL_GPUS python evaluate_math.py \
            --base_model "$BASE_MODEL" \
            --dataset "$ds" \
            --val_n 12 \
            --temperature 1.0 \
            --tensor_parallel_size $TP \
            --checkpoint_dir "$EXP_DIR/checkpoint-$step"
        wait
    done
done
