# TuneAgent: Agentic Operating System Kernel Tuning with Reinforcement Learning

Official resources for "**TuneAgent: Agentic Operating System Kernel Tuning with
Reinforcement Learning**". [Hongyu Lin](https://openreview.net/profile?id=~Hongyu_Lin2), [Yuchen Li](https://openreview.net/profile?id=~Yuchen_Li22), [Haoran Luo](https://openreview.net/profile?id=~Haoran_Luo1), [Zhenghong Lin](https://openreview.net/profile?id=~Zhenghong_Lin1), [Libo Zhang](https://openreview.net/profile?id=~Libo_Zhang1), [Mingjie Xing](https://openreview.net/profile?id=~Mingjie_Xing1), [Yanjun Wu](https://openreview.net/profile?id=~Yanjun_Wu1). **KDD 2026** [[paper](https://arxiv.org/abs/2508.12551)].

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

## Overview

![TuneAgent framework](./figs/Framework.png)

**Linux kernel tuning** is difficult because the kernel configuration space is large,
highly constrained, and workload-sensitive. TuneAgent formulates kernel tuning
as a **constrained reinforcement learning problem** over Kconfig option groups. The
agent observes a tuning target, reasons over candidate kernel options, queries a
kernel knowledge tool when needed, and emits structured configuration decisions
that can be applied to a Linux `.config`.

TuneAgent contains three main designs from the paper:

- **Constraint-aware kernel environment.** Kernel options are organized into
  type-aware groups: `Bool`, `Menu`, `Choice`, and `Value`.
- **Rule-based rewards.** Training combines `R_format`, `R_answer`, and
  `R_perf` to encourage structured reasoning, valid configuration actions, and
  performance-aware exploration.
- **Two-phase GRPO training.** A warm-up phase learns format and semantic
  correctness, followed by a performance-aware exploration phase.

## Experimental Results

The camera-ready paper reports that TuneAgent improves both kernel performance
and configuration validity.

| Setting | Main reported result |
| --- | --- |
| UnixBench overall score | TuneAgent-7B reaches `662.2`, `+35.0` over the default heuristic baseline |
| Overall improvement | Up to `5.6%` |
| Configuration validity | Up to `93.8%` for TuneAgent-7B |
| Nginx | Up to `51.8%` improvement |
| PostgreSQL | About `8.6%-9.4%` improvement |
| Redis | About `1.5%-3.8%` improvement |

Full reproduction requires the curated TuneAgent dataset, trained checkpoints, a
kernel build/boot environment, and the benchmark suites listed below.

## Implementation

The implementation is organized around the paper components:

| Paper component | Repository location |
| --- | --- |
| Configuration-group dataset construction | `examples/data_preprocess/tuneagent.py` |
| Tool-augmented agent interaction | `tuneagent/tool/`, `tuneagent/llm_agent/` |
| Kernel knowledge tool backed by LightRAG | `tuneagent/tool/tools/kernel_knowledge_tool.py`, `inference/RAG.py` |
| Rule rewards `R_format` and `R_answer` | `tuneagent/src/reward_score/tuneagent_score.py` |
| LLM-as-a-Judge reward helper `R_perf` | `tuneagent/src/reward_score/tuneagent_judge_score.py` |
| GRPO/PPO training loop | `tuneagent/src/main_agent.py`, `tuneagent/src/agent_ray_trainer.py` |
| Kconfig traversal and final config generation | `inference/Inference.py`, `inference/ConfigTree.py`, `inference/TuneAgentLLM.py` |
| Paper-scale GRPO launch script | `scripts/run_grpo_tuneagent.sh` |

Repository layout:

```text
TuneAgent/
  tuneagent/
    llm_agent/          # multi-turn generation with tool calls
    src/                # Ray/FSDP training, GRPO/PPO logic, rewards, configs
    tool/               # tool abstraction and TuneAgent knowledge tools
    vllm_infer/         # OpenAI-compatible vLLM client utilities
  examples/
    data_preprocess/    # raw configuration logs to parquet training data
  inference/            # Kconfig traversal and tuned .config generation
  scripts/              # training, sanity-check, serving, and checkpoint scripts
  tests/                # lightweight smoke tests
  figs/                 # figures
  verl/                 # required verl submodule
```

## Installation

TuneAgent is intended for a Linux GPU environment for full training and
inference. The lightweight sanity checks can run without GPUs.

```bash
git clone --recursive <repo-url> TuneAgent
cd TuneAgent

conda create -n tuneagent python=3.10
conda activate tuneagent

pip install -r requirements.txt
pip install -e ./verl
pip install vllm
pip install flash-attn --no-build-isolation
```

Install LightRAG for the kernel knowledge base:

```bash
git clone https://github.com/HKUDS/LightRAG.git
cd LightRAG
git checkout v1.1.1
pip install -e .
cd ../TuneAgent
```

Set API keys if you use OpenAI-compatible judge or knowledge-base calls:

```bash
export OPENAI_API_KEY=<your_api_key>
export OPENAI_BASE_URL=<optional_openai_compatible_base_url>
export WANDB_API_KEY=<optional_wandb_key>
```

## Dataset

The paper uses over 3,000 expert-curated kernel configuration samples covering
CPU scheduling, memory management, file I/O, process management, networking,
locking, and application scenarios.

Expected raw log structure:

```text
data/tuneagent/
  tuneagent_train/*.log
  tuneagent_validate/*.log
```

Each log starts with a tuning target on the first line. Each following line is a
JSON object:

```json
{"question": "Bool\t<config group text>", "answer": [{"config": "CONFIG_X", "value": 2}]}
```

Supported question types are `Bool`, `Menu`, `Choice`, and `Value`.

Preprocess logs into parquet files:

```bash
python examples/data_preprocess/tuneagent.py \
  --local_dir ./data/tuneagent
```

The command writes:

```text
data/tuneagent/train.parquet
data/tuneagent/validation.parquet
```

The curated paper dataset is not included in this checkout. Place released data
under `data/tuneagent/` or pass `DATA_DIR=/path/to/data` to the training script.

## Quick Start

### 1. Sanity Check

This verifies the TuneAgent reward parser without requiring GPUs, datasets,
kernel source code, LightRAG, or checkpoints.

```bash
python scripts/run_sanity_check.py
python -m pytest tests
```

### 2. GRPO Training

Paper-aligned GRPO training:

```bash
DATA_DIR=./data/tuneagent \
BASE_MODEL=Qwen/Qwen2.5-3B-Instruct \
PROJECT_NAME=tuneagent \
EXPERIMENT_NAME=grpo-qwen25-3b \
bash scripts/run_grpo_tuneagent.sh
```

Common overrides:

```bash
N_GPUS=4 \
TRAIN_BATCH_SIZE=128 \
TENSOR_MODEL_PARALLEL_SIZE=4 \
LOGGER="['console','wandb']" \
bash scripts/run_grpo_tuneagent.sh
```

Train the 7B variant:

```bash
BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
bash scripts/run_grpo_tuneagent.sh
```

`scripts/run_ppo_tuneagent.sh` is retained for PPO-style baseline experiments.
The main TuneAgent method in the paper uses GRPO.

### 3. Kernel Knowledge Base

The knowledge tool expects a LightRAG working directory. By default:

```text
data/lightrag_knowledge_base/
```

Override it with:

```bash
export LIGHTRAG_WORKING_DIR=/path/to/lightrag_knowledge_base
export LIGHTRAG_SEARCH_MODE=hybrid
export LIGHTRAG_LLM_FUNC=gpt-4o-mini
```

If the knowledge base is absent, the tool disables itself and returns an error
string. This is acceptable for smoke tests but not for paper-scale experiments.

## Evaluation

The paper evaluates TuneAgent with:

- UnixBench for CPU, memory, file I/O, pipe, shell, system call, process, and
  overall system score.
- ApacheBench for Nginx.
- Sysbench for PostgreSQL.
- Redis Benchmark for Redis.
- Kernel compile-and-boot checks for configuration validity.

Recommended full reproduction workflow:

```bash
python examples/data_preprocess/tuneagent.py --local_dir ./data/tuneagent
DATA_DIR=./data/tuneagent bash scripts/run_grpo_tuneagent.sh
python inference/Inference.py /path/to/linux-6.2.16 \
  --target "Improve overall UnixBench performance" \
  --config-path tuneagent/src/config \
  --config-name agent_trainer_inference \
  --output outputs/unixbench.config
```

Then compile, boot, and benchmark the tuned kernel against the default kernel
configuration.

## Inference

Download a Linux kernel source tree and provide a baseline `.config`:

```bash
wget https://www.kernel.org/pub/linux/kernel/v6.x/linux-6.2.16.tar.gz
tar -zxf linux-6.2.16.tar.gz
cp /path/to/baseline.config linux-6.2.16/.config
```

Generate a tuned configuration:

```bash
python inference/Inference.py /path/to/linux-6.2.16 \
  --target "Improve system memory throughput" \
  --config-path tuneagent/src/config \
  --config-name agent_trainer_inference \
  --output outputs/tuneagent.config \
  --mode hybrid \
  --use-knowledge 1
```

Apply the generated configuration before kernel compilation:

```bash
cp outputs/tuneagent.config /path/to/linux-6.2.16/.config
```

## Limitations

- The curated paper dataset and trained checkpoints are not included in this
  checkout.
- Full reproduction requires GPU training, kernel compile/boot infrastructure,
  and benchmark workloads.
- Existing checkpoints or parquet files produced before the camera-ready naming
  cleanup may need to be regenerated with the `tuneagent` data-source name.
- `R_perf` depends on an OpenAI-compatible judge model and is not exercised by
  the lightweight sanity check.
- Static tests cannot prove kernel bootability; paper-scale validity requires
  real kernel builds and boots.

## BibTeX

```bibtex
@inproceedings{lin2026tuneagent,
  title     = {TuneAgent: Agentic Operating System Kernel Tuning with Reinforcement Learning},
  author    = {Lin, Hongyu and Li, Yuchen and Luo, Haoran and Lin, Zhenghong and Zhang, Libo and Xing, Mingjie and Wu, Yanjun},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year      = {2026},
  series    = {KDD '26},
  location  = {Jeju Island, Republic of Korea},
  publisher = {ACM},
  doi       = {10.1145/3770855.3817987}
}
```
