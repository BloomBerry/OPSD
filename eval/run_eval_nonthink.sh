#!/bin/bash
# Run from the script's own dir so evaluate_math.py resolves regardless of CWD.
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_MODEL="/data0/shared/Qwen3-4B"

# GPUs / tensor-parallel size overridable: EVAL_GPUS=4,5 TP=2 bash run_eval_nonthink.sh
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
TP="${TP:-4}"

# Evaluate the both-nonthink 4B model (student & teacher both non-thinking during training)
# at checkpoint-100 on AIME24, in non-thinking inference mode.
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$EVAL_GPUS python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size $TP \
    --no_thinking \
    --checkpoint_dir /data0/siyanz/opsd/qwen34b_gen1024_both_nonthink_fixteacher_temp11_forwardbeta0_clip1e-6/checkpoint-100
wait
