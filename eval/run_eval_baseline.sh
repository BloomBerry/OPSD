#!/bin/bash
# Baseline reproduction eval — Qwen3-1.7B, 3 benchmarks (AIME24 / AIME25 / HMMT25), Avg@12.
# EXP_DIR must match the training output_dir + run_config from run_opsd_1b_baseline.sh.
BASE_MODEL="Qwen/Qwen3-1.7B"
EXP_DIR="/workspace/outputs/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005"

# base model performance (per benchmark)
for ds in aime24 aime25 hmmt25; do
    NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
        --base_model "$BASE_MODEL" \
        --dataset "$ds" \
        --val_n 12 \
        --temperature 1.0 \
        --tensor_parallel_size 4
    wait
done

# trained checkpoints
for step in 25 50 75 100; do
    for ds in aime24 aime25 hmmt25; do
        NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 python evaluate_math.py \
            --base_model "$BASE_MODEL" \
            --dataset "$ds" \
            --val_n 12 \
            --temperature 1.0 \
            --tensor_parallel_size 4 \
            --checkpoint_dir "$EXP_DIR/checkpoint-$step"
        wait
    done
done
