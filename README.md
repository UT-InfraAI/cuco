<p align="center">
  <img src="docs/images/cuco.png" alt="CUCo Logo" width="220">
</p>

# CUCo: An Agentic Framework for Compute and Communication Co-design

[![arXiv](https://img.shields.io/badge/arXiv-2603.02376-b31b1b.svg)](https://arxiv.org/abs/2603.02376) [![Website](https://img.shields.io/badge/Website-ut--infraai.github.io/cuco-blue.svg)](https://ut-infraai.github.io/cuco/) [![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

CUCo is a training-free, agent-driven framework that automatically generates high-performance CUDA kernels jointly orchestrating computation and communication. By co-optimizing these traditionally disjoint components, CUCo unlocks optimization opportunities unavailable to existing approaches, reducing end-to-end latency by up to 1.57x over host-driven baselines.

## Overview

CUCo consists of three inter-twined components:

1. **Design Space Specification** — A structured, declarative set of communication primitives (backend, placement, sync scope, issuer granularity, chunk size) that grounds agent reasoning in valid collective semantics.

2. **Fast-Path Agent** — A correctness-first pipeline that converts host-driven NCCL code into device-initiated (GIN/LSA) equivalents through a three-step process: CUDA code analysis, host-to-device transformation via an LLM-judge loop, and evolve-block annotation.

3. **Slow-Path Agent** — An LLM-driven evolutionary search that optimizes the fast-path baseline through island-based populations, phase-dependent explore/exploit mutation, cascaded evaluation, and a shared candidate database with meta-summarization.

### How It Works

<p align="center">
  <img src="docs/images/workflow.png" alt="CUCo Workflow" width="700">
</p>

Given a host-driven CUDA+NCCL kernel, CUCo's fast-path agent first analyzes the communication pattern, converts host-side collectives to device-initiated GIN/LSA primitives, and annotates mutable regions with EVOLVE-BLOCK markers. The slow-path agent then treats the annotated kernel as generation 0 and runs an evolutionary search: each generation, an LLM mutates the code within the evolve blocks, the candidate is compiled, run, and scored, and the result feeds back into the next iteration. Over 10-20 generations, this loop discovers optimizations like compute-communication overlap, kernel fusion, and pipelined transfers that are difficult to find manually.

### Key Results

CUCo was evaluated on four representative workloads spanning different compute-communication patterns. In each case, CUCo's evolved kernels significantly outperform the host-driven NCCL baselines.

<table>
  <tr>
    <td align="center"><b>DeepSeek-V3 MoE</b><br><sub>Dispatch-Compute-Combine</sub></td>
    <td align="center"><b>KV Cache Transfer</b><br><sub>Prefill-Decode Pipeline</sub></td>
  </tr>
  <tr>
    <td><img src="docs/images/moe_ratio_sweep.png" alt="MoE Ratio Sweep" width="450"></td>
    <td><img src="docs/images/kv_transfer.png" alt="KV Cache Transfer" width="450"></td>
  </tr>
  <tr>
    <td align="center"><b>Flash Attention</b><br><sub>Attention with AllGather</sub></td>
    <td align="center"><b>GEMM + AllGather</b><br><sub>Matmul with Collective</sub></td>
  </tr>
  <tr>
    <td><img src="docs/images/flash_attention.png" alt="Flash Attention" width="450"></td>
    <td><img src="docs/images/gemm_allgather.png" alt="GEMM AllGather" width="450"></td>
  </tr>
</table>

## Documentation

<table>
  <tr>
    <th>Guide</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><a href="docs/getting-started.md">Getting Started</a></td>
    <td>Installation, first run, end-to-end walkthrough</td>
  </tr>
  <tr>
    <td><a href="docs/architecture.md">Architecture</a></td>
    <td>System design, module map, data flow</td>
  </tr>
  <tr>
    <td><a href="docs/adding-a-workload.md">Adding a New Workload</a></td>
    <td>Step-by-step guide to onboard your own kernel</td>
  </tr>
  <tr>
    <td><a href="docs/fast-path-agent.md">Fast-Path Agent</a></td>
    <td>Host-to-device transformation pipeline</td>
  </tr>
  <tr>
    <td><a href="docs/slow-path-agent.md">Slow-Path Agent</a></td>
    <td>Evolutionary search deep dive</td>
  </tr>
  <tr>
    <td><a href="docs/configuration.md">Configuration Reference</a></td>
    <td>All config parameters (EvolutionConfig, TransformConfig, etc.)</td>
  </tr>
  <tr>
    <td><a href="docs/llm-backends.md">LLM Backends</a></td>
    <td>Provider setup (Anthropic, Bedrock, OpenAI, Gemini, DeepSeek)</td>
  </tr>
  <tr>
    <td><a href="docs/evaluation.md">Writing Evaluations</a></td>
    <td>Custom evaluate.py for your workload</td>
  </tr>
  <tr>
    <td><a href="docs/visualization.md">Visualization</a></td>
    <td>Web UI, plotting tools, database queries</td>
  </tr>
</table>

## Extensibility

While the included example uses CUDA and NCCL device APIs, CUCo's core framework is workload-agnostic. Run `cuco_init /path/to/kernel.cu` to scaffold a new workload with all required files pre-configured for your cluster. The evaluation script (`evaluate.py`), prompt customization (`run_evo.py`), and API documentation file are all user-defined — you can adapt CUCo for any kernel, library, or optimization target where an LLM can generate code and a script can score it. See [Adding a New Workload](docs/adding-a-workload.md) for details.

## Repository Layout

```
cuco/                   Core framework
├── core/               Evolution runner, sampler, novelty judge, summarizer
├── database/           Candidate database, complexity analysis, island management
├── edit/               Diff/full-rewrite application, async editing
├── llm/                LLM client, model backends (Anthropic, OpenAI, Gemini, DeepSeek)
├── prompts/            Mutation prompt templates (base, diff, full, cross, novelty, meta)
├── transform/          Fast-path agent: CUDA analyzer, host-to-device transformer
├── plots/              Visualization utilities (lineage trees, pareto fronts, improvement plots)
├── webui/              Interactive evolution visualization UI
├── launch/             Local and Slurm launch backends
├── templates/          Templates for evaluate.py, .gitignore (used by cuco_init)
├── site_config.py      Cluster auto-detection and ~/.cuco/site.yaml management
├── init_workload.py    Workload scaffolding logic (used by cuco_init)
├── run_workload.py     Workload launcher logic (used by cuco_run)
├── cuco_init           CLI: scaffold new workloads or run cluster setup
├── cuco_run            CLI: launch evolution for a named workload
├── cuco_launch         Entry point for launching evolution runs
└── cuco_visualize      Entry point for the visualization UI
workloads/
└── ds_v3_moe/          DeepSeek-V3 MoE dispatch-compute-combine workload
    ├── ds_v3_moe.cu    Seed CUDA kernel (host-driven baseline)
    ├── evaluate.py     Build, run, and fitness evaluation logic
    ├── run_evo.py      Launch slow-path evolutionary search
    ├── run_transform.py Launch fast-path host-to-device transformation
    ├── nccl_api_docs.py NCCL device API documentation for agent context
    └── results_ds_v3_moe/ Evolution results (generations, scores, logs)
pyproject.toml          Package configuration and dependencies
uv.lock                 Locked dependency versions
```

## Setup

### Prerequisites

- Python >= 3.10
- CUDA 13.1+ with NCCL 2.28.9+ (for device-initiated communication)
- NVIDIA GPUs with NVLink (intra-node) or RoCE (inter-node)
- LLM API credentials (Anthropic Bedrock, OpenAI, etc.)

### Installation

```bash
# Clone the repository
git clone https://github.com/UT-InfraAI/cuco.git
cd cuco

# Create virtual environment and install
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or with [uv](https://docs.astral.sh/uv/) (recommended):

```bash
uv venv
source .venv/bin/activate
uv sync
```

### Configuration

Create a `.env` file in the repository root with your LLM API credentials:

```bash
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

## Usage

### Quick Start: Add Your Own Workload

CUCo provides `cuco_init` to scaffold a new workload from any seed CUDA kernel:

```bash
# One-time cluster setup (auto-detects CUDA, NCCL, MPI, GPUs)
cuco_init --setup

# Scaffold a new workload from your seed kernel
cuco_init /path/to/my_kernel.cu

# Run evolution (50 generations by default)
cuco_run my_kernel --generations 50
```

`cuco_init` creates a ready-to-run workload directory under `workloads/` with all required files (`evaluate.py`, `run_evo.py`, `run_transform.py`, etc.) pre-configured using your cluster settings from `~/.cuco/site.yaml`. See [Adding a New Workload](docs/adding-a-workload.md) for details.

### Fast-Path Agent (Host-to-Device Transformation)

The fast-path agent converts a host-driven NCCL program into a device-initiated equivalent:

```bash
cd workloads/ds_v3_moe
python run_transform.py
```

This runs the three-step pipeline (CUDA analysis, host-to-device transformation, evolve-block annotation) and outputs the transformed kernel to `_transform_host_output/`.

### Slow-Path Agent (Evolutionary Search)

The slow-path agent optimizes the transformed kernel through LLM-driven evolution:

```bash
cd workloads/ds_v3_moe
python run_evo.py --num_generations=18
```

Evolution results (candidate programs, scores, logs) are saved to `results_ds_v3_moe/`.

### Visualization

Launch the interactive web UI to explore the evolution tree:

```bash
cuco_visualize --db workloads/ds_v3_moe/results_ds_v3_moe/evolution_db.sqlite
```

## Citation

```bibtex
@misc{hu2026cucoagenticframeworkcompute,
      title={CUCo: An Agentic Framework for Compute and Communication Co-design}, 
      author={Bodun Hu and Yoga Sri Varshan V and Saurabh Agarwal and Aditya Akella},
      year={2026},
      eprint={2603.02376},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2603.02376}, 
}
```

## License

Apache 2.0
