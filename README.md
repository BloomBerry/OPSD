# Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models


<p align="center">
<a href="https://arxiv.org/pdf/2601.18734v3"><img src="https://img.shields.io/badge/arXiv-2601.18734-b31b1b.svg"></a>
<a href="https://siyan-zhao.github.io/blog/2026/opsd/"><img src="https://img.shields.io/badge/Blog-Post-blue.svg"></a>
</p>

---
## Overview

**On-Policy Self-Distillation (OPSD)** trains a single model to act as both student and teacher by conditioning on different contexts — the student sees only the problem, while the teacher additionally sees the ground-truth solution — and performs token-level distribution matching along the student's own on-policy trajectories.


## Updates

- **Mar 18, 2026**: Released updated code. 

  (1) Fixed chat template and zero2 bugs (see [template issue](https://github.com/huggingface/trl/issues/5241)), we re-ran experiments with updated results (detailed results & ablations updated on arxiv/blog). The fixes yield improved OPSD performance, most notably on Qwen3-1.7B.

  (2) Added a new training stabilization strategy 🚀: per-token point-wise KL clipping. We find style tokens (such as 'wait', 'think') can exhibit 6–15× higher KL divergence than math-related tokens, and dominates the training signal. Clipping stablizes training and improves performance.


-  **Mar 3, 2026**: Initial code release.

## Installation


```bash
conda env create -f environment.yml
conda activate opsd
```

```bash
pip install flash-attn==2.8.3 --no-build-isolation
```
If you encounter difficulties installing flash-attn, you can check the version matching your CUDA and PyTorch versions from the [flash-attention releases page](https://github.com/Dao-AILab/flash-attention/releases).

The code uses `trl`'s experimental GOLD trainer as a base.

## Repository Structure

```
├── opsd_trainer.py          # OPSDTrainer: core self-distillation trainer
├── data_collator.py         # Data collator for self-distillation
├── opsd_train.py            # OPSD training entry point
├── sft_train.py             # SFT baseline training entry point
├── grpo_train.py            # GRPO baseline training entry point
├── accelerate.yaml          # Accelerate config (multi-GPU)
├── Dockerfile               # Pinned image (opsd:repro) for reproduction runs
├── scripts/
│   ├── run_opsd.sh          # Example launch script for OPSD
│   ├── run_sft.sh           # Example launch script for SFT
│   ├── run_grpo.sh          # Example launch script for GRPO
│   ├── run_opsd_1b_baseline.sh  # Qwen3-1.7B baseline reproduction (training)
│   └── docker_run.sh        # Run any script above inside the pinned image
└── eval/
    ├── evaluate_math.py     # Evaluation script (vLLM)
    ├── run_eval.sh          # Example evaluation script
    └── run_eval_baseline.sh # Qwen3-1.7B baseline reproduction (all 15 eval jobs)
```

## Quick Start

Reproduce results on Qwen3-1.7B (🚀 training only takes **~15 minutes** on 4×H100 and peaks within 100 steps):

```bash
bash scripts/run_opsd_1b.sh
```
Evaluation: (evaluation takes ~ 30-50 minutes on 4xh100 for each checkpoint) 
```bash
cd eval
bash run_eval.sh
```

### Evaluation Results across Tasks on Qwen3-1.7B

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 51.5% |
| 25 | 51.4% |
| 50 | 52.8% |
| 75 | 54.4% |
| 100 | 57.2% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 36.7% |
| 25 | 42.5% |
| 50 | 43.9% |
| 75 | 40.6% |
| 100 | 41.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 25 | 24.7% |
| 50 | 27.8% |
| 75 | 26.9% |
| 100 | 29.2% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, thinking mode enabled, max new tokens=38912, top-p=none, top-k disabled, min-p=0, presence penalty=0, num samples=12


## Baseline Reproduction (Docker)

Tooling for reproducing the Qwen3-1.7B table above inside a pinned container.

### Setup

```bash
docker build -t opsd:repro .
export WANDB_API_KEY=<your-key-from-https://wandb.ai/authorize>
```

`scripts/docker_run.sh` runs any script under `scripts/` or `eval/` inside the image:

| Variable | Default | Meaning |
|---|---|---|
| `WANDB_API_KEY` | *(required)* | Runs log to wandb project `OPSD` in online mode |
| `GPUS` | `all` | Passed to `docker --gpus`, e.g. `GPUS='"device=4,5,6,7"'` |
| `IMAGE` | `opsd:repro` | Image to run |
| `HF_CACHE_DIR` | `/data/nas_vol1/huggingface` | HF hub cache on the NAS, mounted read-write at `/hf_cache` |
| `EVAL_GPUS`, `TP`, `STEPS`, `RUN_BASE` | — | Forwarded to the eval script (see below) |

### Training

```bash
GPUS='"device=4,5,6,7"' bash scripts/docker_run.sh scripts/run_opsd_1b_baseline.sh
```

Checkpoints land in `outputs/qwen31b_gen1024_fixteacher_temp11_forwardbeta0_clip005/checkpoint-*`
(`--save_steps 25`). The baseline peaks within 100 steps, so the eval below only needs
checkpoint-25/50/75/100.

### Evaluation

```bash
# default: base + checkpoint-100 on all three benchmarks (6 jobs)
GPUS='"device=4,5,6"' EVAL_GPUS=0,1,2 bash scripts/docker_run.sh eval/run_eval_baseline.sh

# full sweep reproducing the table above (15 jobs)
GPUS='"device=4,5,6"' EVAL_GPUS=0,1,2 STEPS="25 50 75 100" \
    bash scripts/docker_run.sh eval/run_eval_baseline.sh
```

One job per (checkpoint, benchmark), each at Avg@12.

| Variable | Default | Meaning |
|---|---|---|
| `EVAL_GPUS` | `0,1,2,3` | **Container-side** device index, so `0..N-1` for whatever `GPUS` exposes |
| `STEPS` | `100` | Space-separated checkpoint steps, e.g. `"25 50 75 100"` |
| `RUN_BASE` | `1` | Also evaluate the untrained base model; `0` to skip |
| `TP` | `1` | Tensor-parallel size — see below |
| `LOG_DIR` | `eval_logs` | Per-job logs, relative to `eval/` |

`TP` defaults to `1` because Qwen3-1.7B is ~3.4GB in bf16 and fits on one 80GB GPU: tensor
parallelism buys no memory here and only adds two all-reduces per layer per decode step. The
GPUs are used for data parallelism instead — one eval process per GPU, jobs round-robined
across them. Set `TP>1` to fall back to a single process spanning all of `EVAL_GPUS`.

Results land in `eval/eval_results/*.json`, one file per job.

### Performance notes

The first full reproduction took ~20h for its 15 jobs on 2×A100 with `TP=2`, holding a flat
~1,000 output tokens/s. The workload is genuinely heavy — thinking mode generates a median of
~16k (AIME24) to ~23k (HMMT25) tokens per solution, 360 solutions per job — but two settings
were making it worse than it needed to be, and both are now fixed:

- **CUDA graphs are on by default.** `enforce_eager` used to be hardcoded to `True`. Decode for
  a 1.7B model is kernel-launch bound, so graph capture matters a lot. Pass `--enforce_eager`
  to `evaluate_math.py` to revert if you hit graph-capture issues.
- **`TP=1` with one process per GPU** instead of `TP=2` (see above).

Known remaining costs, not addressed:

- LoRA is served at runtime (`max_lora_rank=64`) rather than merged into the base weights,
  adding ~200 kernel launches per decode step. Base-model jobs do not pay this.
- Every job starts a fresh vLLM engine rather than hot-swapping adapters.
- `max_model_len=40960` with sequences reaching ~40k tokens means KV cache capacity, not
  compute, caps the batch size. Check the `GPU KV cache size:` line in `eval/eval_logs/*.log`;
  if preemption warnings show up, raise `--gpu_memory_utilization`.
- No seed is fixed in training or sampling, so run-to-run variation of a few points is expected.

## Non-Thinking Mode

OPSD can also run in non-thinking setting where both the Qwen student and teacher are enabled_thinking=False during training (`--student_thinking False --teacher_thinking False`) and evaluated with non-thinking inference (`--no_thinking`), with faster evaluation time than thinking mode.

Training:
```bash
bash scripts/run_opsd_4b_nonthink.sh
bash scripts/run_opsd_8b_nonthink.sh
```

Evaluation:
```bash
cd eval
bash run_eval_nonthink.sh
```

### Evaluation Results with Non-Thinking Mode across Models

#### Qwen3-8B (`--jsd_token_clip 1e-7`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 26.4% |
| 50 | 49.7% |
| 75 | 45.3% |
| 100 | 38.3% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 19.7% |
| 50 | 35.0% |
| 75 | 26.9% |
| 100 | 27.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 18.3% |
| 75 | 17.5% |
| 100 | 15.3% |

</td>
</tr>
</table>
</div>

#### Qwen3-4B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 23.1% |
| 50 | 20.3% |
| 75 | 27.5% |
| 100 | 31.1% |
| 150 | 32.8% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 21.4% |
| 50 | 21.4% |
| 75 | 20.8% |
| 100 | 21.1% |
| 150 | 21.9% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 10.8% |
| 50 | 11.1% |
| 75 | 13.1% |
| 100 | 16.4% |
| 150 | 14.4% |

</td>
</tr>
</table>
</div>

#### Qwen3-1.7B (`--jsd_token_clip 1e-6`)

<div align="center">
<table>
<tr>
<th align="center">AIME24</th>
<th align="center">AIME25</th>
<th align="center">HMMT25</th>
</tr>
<tr>
<td>

| Step | Avg@12 |
|---|---|
| Base | 11.9% |
| 50 | 15.0% |
| 75 | 13.9% |
| 100 | 12.5% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 9.2% |
| 50 | 6.2% |
| 75 | 8.3% |
| 100 | 8.1% |

</td>
<td>

| Step | Avg@12 |
|---|---|
| Base | 5.0% |
| 25 | 7.2% |
| 50 | 5.8% |
| 75 | 5.0% |

</td>
</tr>
</table>
</div>

> **Evaluation settings:** temperature=1.0, non-thinking mode, num samples=12.



## Key OPSD arguments

| Argument | Default | Description |
|---|---|---|
| `--fixed_teacher` | `False` | Fix the teacher to the initial policy (step 0). Requires --use_peft. Note ❗ If you disable PEFT, the teacher will keep updating at every training step, which may make training unstable. Our main results use the fixed teacher, which is currently implemented with LoRA adapter weights. |
| `--use_tinker_loss` | `False` | Use sampled-token policy-gradient objective instead of full-vocabulary JSD. More memory efficient. Currently no clipped implemented for this variant, could be unstable. |
| `--max_completion_length` | — | Student generation length for distillation. We use 1024 in our main experiments. |
| `--beta` | — | Interpolation weight for the JSD mixture distribution. Beta=0 means forward KL and 1 means reverse KL. |
| `--jsd_token_clip` | 0.05 | Clip the point-wise JSD loss contributions to a maximum value before summing over vocabulary (i.e., clipping is applied point-wise, not to the vocabulary-summed loss). This can improve stability by preventing stylistic tokens from dominating the training signal. Note when clipping is applied, the loss can be negative due to positive KL summand being capped. | 
| `--reason_first` | `False` | Prepend an explicit rationalization to the teacher context before distillation. |
| `--run_config` | `None` | Custom name suffix for the output directory and WandB run. |

### SFT Baseline

See [`scripts/run_sft.sh`](scripts/run_sft.sh).

### GRPO Baseline

See [`scripts/run_grpo.sh`](scripts/run_grpo.sh).

### Acknowledgements
Our implementation builds on [TRL GOLD Trainer](https://huggingface.co/docs/trl/gold_trainer). We sincerely thank [@simran135](https://github.com/simran135) and [@beanie00](https://github.com/beanie00) for identifying the prompt template bugs and the zero-2 issue, respectively!

## Citation
If you find this useful, please consider citing:
```bibtex
@article{zhao2026self,
  title={Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models},
  author={Zhao, Siyan and Xie, Zhihui and Liu, Mengchen and Huang, Jing and Pang, Guan and Chen, Feiyu and Grover, Aditya},
  journal={arXiv preprint arXiv:2601.18734},
  year={2026}
}
```
