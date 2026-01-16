# Tau2-RL-Pipeline

Multi-turn tool-use training pipeline for tau2-bench using [slime](https://github.com/THUDM/slime).

A 4B parameter model achieving **57.1% Pass@4** on tau2-bench (test split): **4x better than the base model** and competitive with models 6-60x larger.

<p align="center">
  <img src="public/performance-chart.jpeg" alt="Tau2 performance comparison" width="600">
</p>

<p align="center">
  <img src="public/slime-pipeline-tau2.jpeg" alt="Tau2 training pipeline" width="700">
</p>

## Results

| Stage | Overall | Airline | Retail | Telecom |
|-------------------------------|---------|---------|--------|---------|
| Baseline (Qwen3-4B-Instruct) | 14.3% | 5.0% | 16.0% | 20.0% |
| SFT | 8.57% | 5.0% | 20.0% | 0.0% |
| SFT1 (RFT) | 27.0% | 20.0% | 50.0% | 7.5% |
| GRPO (Pass@1, greedy) | 32.9% | 15.0% | 76.0% | 4.0% |
| **GRPO (Pass@4, reported)** | **57.1%** | **50.0%** | **76.0%** | **44.0%** |

## Setup

All scripts use `slimerl/slime:latest` container:

```bash
# Pull and start container
docker pull slimerl/slime:latest
docker run --gpus all --rm -it \
  -v "$(pwd)":/workspace/tau2-rl-pipeline \
  -w /workspace/tau2-rl-pipeline \
  slimerl/slime:latest

# Inside container: set environment
export TAU2_ROOT=/workspace/tau2-rl-pipeline
export TAU2_OUT_DIR="${TAU2_ROOT}/outputs"
mkdir -p "${TAU2_OUT_DIR}"

# Install tau2-bench (official)
git clone https://github.com/sierra-research/tau2-bench.git "${TAU2_OUT_DIR}/_external/tau2-bench"
cd "${TAU2_OUT_DIR}/_external/tau2-bench"
git checkout 337326e62d8e0ca74c353b004a9c5d748e0ba914
pip install -e . --no-deps
export TAU2_DATA_DIR="${TAU2_OUT_DIR}/_external/tau2-bench/data"
cd "${TAU2_ROOT}"

# Install runtime deps (pin litellm to avoid conflicts)
pip install gymnasium addict deepdiff fs langfuse plotly pydantic-argparse redis \
  scikit-learn seaborn tenacity watchdog "litellm==1.65.0"

# Configure API keys
cp configs/.env.template configs/.env  # ADD OPENAI_API_KEY
set -a && source configs/.env && set +a
```

## Quick Start: Evaluate

Uses the published checkpoint (~2h on 2xH100):

```bash
# Terminal 1: Policy server (use --tp 1 for single GPU)
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path Jarrodbarnes/Qwen3-4B-tau2-grpo-v1 \
  --host 0.0.0.0 --port 30000 --tp 2 --mem-fraction-static 0.70

# Terminal 2: Run evaluation (requires OPENAI_API_KEY for user simulator)
python3 eval/eval_passk.py \
  --hf-checkpoint Jarrodbarnes/Qwen3-4B-tau2-grpo-v1 \
  --sglang-url http://127.0.0.1:30000/generate \
  --domains airline,retail,telecom --task-split test --num-samples 4 \
  --temperature 0.8 --top-p 1.0 --top-k 20 \
  --output "${TAU2_OUT_DIR}/eval_pass4.json"
```

## Train from Scratch

### Prerequisites

**1. Download base model and SFT training data:**

```bash
huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 \
  --local-dir "${TAU2_OUT_DIR}/models/Qwen3-4B-Instruct-2507"

mkdir -p "${TAU2_OUT_DIR}/data/sft1"
huggingface-cli download Jarrodbarnes/tau2-sft-seed-v3 \
  --local-dir "${TAU2_OUT_DIR}/data/sft1" --repo-type dataset
export TAU2_SFT_DATA_DIR="${TAU2_OUT_DIR}/data/sft1"
export SFT_DATA_JSONL="${TAU2_SFT_DATA_DIR}/tau2_sft_merged_v3_rft.jsonl"
```

**2. Convert to Megatron format:**

```bash
cd /root/slime
source scripts/models/qwen3-4B-Instruct-2507.sh
python3 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint "${TAU2_OUT_DIR}/models/Qwen3-4B-Instruct-2507" \
  --save "${TAU2_OUT_DIR}/models/Qwen3-4B-Instruct-2507_torch_dist" \
  ${MODEL_ARGS[@]}
cd "${TAU2_ROOT}"
```

### Stage 1: SFT

```bash
bash scripts/train_sft.sh
```

For a smaller debug run, set `SFT_DATA_JSONL="${TAU2_SFT_DATA_DIR}/seed_sft_v3.jsonl"`.

### Stage 2: GRPO

**Generate task indices:**

```bash
python3 tau2_rl_pipeline/tasks.py \
  --local_dir "${TAU2_OUT_DIR}/tasks" \
  --domains airline,retail,telecom --splits train
```

**Start user simulator (separate terminal, keep GPUs distinct):**

```bash
GPUS=2,3 bash scripts/start_user_sim.sh
```

**Run GRPO training:**

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 bash scripts/train_grpo.sh
```

Training takes ~2 hours on 8xH100s.

## Resources

**Models (Hugging Face):**
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) - Base model
- [Qwen3-4B-tau2-sft1](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-sft1) - After SFT+RFT
- [Qwen3-4B-tau2-grpo-v1](https://huggingface.co/Jarrodbarnes/Qwen3-4B-tau2-grpo-v1) - Final GRPO checkpoint

**Dataset:** [tau2-sft-seed-v3](https://huggingface.co/datasets/Jarrodbarnes/tau2-sft-seed-v3)

**Training logs:** [WandB project](https://wandb.ai/jbarnes850-near-protocol/tau2-cookbook)

## Configuration

Environment variables (set in `configs/.env`):

| Variable | Default | Description |
|----------|---------|-------------|
| `TAU2_USER_MODEL` | `openai/Qwen/Qwen3-4B-Instruct-2507` | User simulator model |
| `TAU2_USER_API_BASE` | `http://127.0.0.1:30001/v1` | User simulator endpoint |
| `TAU2_USER_TEMPERATURE` | `0.7` | User simulator temperature |
| `TAU2_MAX_STEPS` | `100` | Max steps per episode |
| `TAU2_REWARD_ALPHA` | `0.25` | Partial score weight |
| `TAU2_USE_CURRICULUM` | `1` | Enable curriculum learning |

## Project Structure

```
tau2_rl_pipeline/
  __init__.py         # Package exports
  rollout.py          # Multi-turn GRPO rollout generation
  reward.py           # Reward shaping + curriculum
  env.py              # tau2-bench wrapper utilities
  actions.py          # Action parsing
  prompting.py        # System prompts
  tasks.py            # Task preprocessing
scripts/
  train_sft.sh        # SFT training script
  train_grpo.sh       # GRPO training script
  start_user_sim.sh   # User simulator server
eval/
  eval_passk.py       # Pass@K evaluation
configs/
  .env.template       # Environment template
```

## Troubleshooting

- **SGLang abort/OOM**: Reduce `--mem-fraction-static`, `--max-tokens-per-gpu`, or `--rollout-batch-size`
- **Ray working directory issues**: Scripts submit Ray jobs with `working_dir` set explicitly; avoid running from random directories
- **Telecom is slow / low Pass@K**: Dual-control pushes difficulty into communication. Inspect failures for tool ownership violations, premature `done`, or missing follow-up questions

## Acknowledgments

- [slime](https://github.com/THUDM/slime) - RL training framework
- [tau2-bench](https://github.com/sierra-research/tau2-bench) - Multi-turn agent benchmark
- [Qwen3](https://huggingface.co/Qwen) - Base model family

## License

Apache-2.0
