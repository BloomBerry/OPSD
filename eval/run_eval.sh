#!/bin/bash
# Run from the script's own dir so evaluate_math.py resolves regardless of CWD.
cd "$(dirname "${BASH_SOURCE[0]}")"

BASE_MODEL="/data0/shared/Qwen3-1.7B"
EXP_DIR="/data1/opsd/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"

# GPUs / tensor-parallel size overridable: EVAL_GPUS=4,5 TP=2 bash run_eval.sh
EVAL_GPUS="${EVAL_GPUS:-0,1,2,3}"
TP="${TP:-4}"

# evaluate base model performance
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$EVAL_GPUS python evaluate_math.py \
    --base_model "$BASE_MODEL" \
    --dataset "aime24" \
    --val_n 12 \
    --temperature 1.0 \
    --tensor_parallel_size $TP
wait

# after trained, evaluate the performance of the trained model.
for step in 25 50 75 100; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=$EVAL_GPUS python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "aime24" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size $TP \
        --checkpoint_dir "$EXP_DIR/checkpoint-$step"
done
