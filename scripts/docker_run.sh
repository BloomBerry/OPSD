#!/usr/bin/env bash
# Launch an OPSD training/eval run inside the opsd:repro Docker image with
# Weights & Biases logging in online mode.
#
# Usage:
#   # 1) put your keys here (or export them in your shell before calling)
#   export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
#   export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx   # optional, for gated models
#
#   # 2) run any script under scripts/ (defaults to the 1B baseline)
#   bash scripts/docker_run.sh scripts/run_opsd_1b_baseline.sh
#
set -euo pipefail

# ---- config -----------------------------------------------------------------
IMAGE="${IMAGE:-opsd:repro}"
TRAIN_SCRIPT="${1:-scripts/run_opsd_1b_baseline.sh}"
GPUS="${GPUS:-all}"                       # e.g. GPUS='"device=0,1"' to pin GPUs
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Persist HF downloads and training outputs on the host across container runs.
HF_CACHE_DIR="${HF_CACHE_DIR:-$REPO_DIR/.hf_cache}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_DIR/outputs}"
WANDB_DIR="${WANDB_DIR:-$REPO_DIR/wandb}"

# ---- checks -----------------------------------------------------------------
if [[ -z "${WANDB_API_KEY:-}" ]]; then
    echo "ERROR: WANDB_API_KEY is not set. Export it first, e.g.:" >&2
    echo "  export WANDB_API_KEY=<your-key-from-https://wandb.ai/authorize>" >&2
    exit 1
fi

mkdir -p "$HF_CACHE_DIR" "$OUTPUTS_DIR" "$WANDB_DIR"

echo ">> image:        $IMAGE"
echo ">> gpus:         $GPUS"
echo ">> train script: $TRAIN_SCRIPT"
echo ">> wandb mode:   online (project OPSD)"

# ---- run --------------------------------------------------------------------
exec docker run --rm -it \
    --gpus "$GPUS" \
    --shm-size=32g \
    --ipc=host \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e WANDB_MODE=online \
    -e WANDB_PROJECT="${WANDB_PROJECT:-OPSD}" \
    ${WANDB_ENTITY:+-e WANDB_ENTITY="$WANDB_ENTITY"} \
    ${HF_TOKEN:+-e HF_TOKEN="$HF_TOKEN"} \
    -e HF_HOME=/workspace/.hf_cache \
    -v "$REPO_DIR":/workspace \
    -v "$HF_CACHE_DIR":/workspace/.hf_cache \
    -v "$OUTPUTS_DIR":/workspace/outputs \
    -v "$WANDB_DIR":/workspace/wandb \
    "$IMAGE" \
    bash "$TRAIN_SCRIPT"
