# System Architecture

This document describes CUCo's architecture, module layout, and data flow. For the theoretical foundations, see the [paper](https://arxiv.org/abs/2603.02376).

## End-to-End Pipeline

CUCo transforms a host-driven CUDA+NCCL program into an optimized device-initiated kernel through two sequential agents:

```
                        ┌──────────────────────────────────────────┐
                        │            FAST-PATH AGENT               │
  Host-driven           │                                          │
  CUDA + NCCL  ───────► │  Analyze ──► Transform ──► Annotate      │
  seed kernel           │  (regex)     (LLM loop)    (EVOLVE-BLOCK)│
                        └────────────────┬─────────────────────────┘
                                         │  Correct, conservative
                                         │  device-side kernel
                                         ▼
                        ┌──────────────────────────────────────────┐
                        │            SLOW-PATH AGENT               │
                        │                                          │
                        │  Island-based evolutionary search        │
                        │  LLM mutation ──► Cascade evaluation     │
                        │  Explore/exploit phases                  │
                        │  Meta-summarizer feedback loop           │
                        └────────────────┬─────────────────────────┘
                                         │
                                         ▼
                                  Optimized kernel
                                  (best candidate)
```

### Fast-Path Agent

The fast-path agent prioritizes **correctness**. Starting from a host-driven program (standard NCCL collectives called from the CPU), it produces a device-initiated equivalent through a three-step pipeline:

1. **CUDA Analysis** (`CUDAAnalyzer`) — Regex-based extraction of NCCL collectives, buffer allocations, kernel launches, and their data dependencies. Produces a communication dependency graph.
2. **Host-to-Device Transformation** (`HostToDeviceTransformer`) — An LLM-driven build/verify loop that rewrites host collectives into device-initiated forms (GIN or LSA). Operates in two stages:
   - **Stage A**: Add device-side infrastructure (ncclMemAlloc, window registration, device communicator) while keeping host collectives.
   - **Stage B**: Replace host-side NCCL calls with device-side kernel(s).
3. **Evolve-Block Annotation** — Mark mutable code regions with `EVOLVE-BLOCK-START` / `EVOLVE-BLOCK-END` markers. Frozen regions (MPI/NCCL init, main, verification) cannot be modified by the slow-path agent.

### Slow-Path Agent

The slow-path agent prioritizes **performance**. It takes the fast-path output as generation zero and runs an island-based evolutionary search:

1. **Parent Selection** — Choose a candidate from the population (power-law, weighted, or beam search).
2. **LLM Mutation** — Propose a code modification: diff patch, full rewrite, or crossover from the archive.
3. **Cascade Evaluation** — Screen candidates through three levels:
   - L1: Compile (nvcc)
   - L2: Run and verify correctness (mpirun + "Verification: PASS")
   - L3: Benchmark and score (fitness = `10000 / (1 + time_ms)`)
4. **Database Update** — Store the candidate (including failures) with metrics, LLM feedback, and code embedding.
5. **Meta-Summarization** — Periodically distill cross-generation patterns into actionable recommendations.

## Module Map

```
cuco/
├── core/                      # Slow-path evolutionary search
│   ├── runner.py              # EvolutionConfig, EvolutionRunner (main loop)
│   ├── sampler.py             # PromptSampler (diff/full/cross prompt assembly)
│   ├── summarizer.py          # MetaSummarizer (cross-generation learning)
│   ├── novelty_judge.py       # NoveltyJudge (embedding + LLM novelty filter)
│   └── wrap_eval.py           # Generic evaluation wrapper for Hydra
│
├── transform/                 # Fast-path host-to-device transformation
│   ├── cuda_analyzer.py       # CUDAAnalyzer (regex-based NCCL/CUDA extraction)
│   ├── transformer.py         # TransformConfig, HostToDeviceTransformer
│   └── pipeline.py            # PreTransformPipeline (ordered conditional steps)
│
├── database/                  # Candidate storage and selection
│   ├── dbase.py               # DatabaseConfig, Program, ProgramDatabase (SQLite)
│   ├── parents.py             # Parent selection strategies
│   ├── inspirations.py        # Archive/top-k inspiration sampling
│   ├── islands.py             # Island assignment, migration, multi-seed
│   ├── complexity.py          # Code complexity metrics (radon, custom C++)
│   └── display.py             # Rich-based database display
│
├── llm/                       # LLM abstraction layer
│   ├── client.py              # get_client_llm() — provider routing
│   ├── llm.py                 # LLMClient, AsyncLLMClient, cost tracking
│   ├── query.py               # query() — dispatches to provider backends
│   ├── embedding.py           # EmbeddingClient (OpenAI, Gemini, Bedrock)
│   ├── dynamic_sampling.py    # Bandit-based model selection (UCB)
│   └── models/                # Per-provider implementations
│       ├── anthropic.py       # Anthropic / Bedrock
│       ├── openai.py          # OpenAI / Azure
│       ├── deepseek.py        # DeepSeek
│       ├── gemini.py          # Google Gemini
│       ├── claude_cli.py      # Claude Code CLI (subprocess)
│       ├── pricing.py         # Model registries and pricing tables
│       └── result.py          # QueryResult dataclass
│
├── prompts/                   # Mutation prompt templates
│   ├── prompts_base.py        # BASE_SYSTEM_MSG, performance formatting
│   ├── prompts_diff.py        # SEARCH/REPLACE diff mutation
│   ├── prompts_full.py        # Full-rewrite mutation (5 variants)
│   ├── prompts_cross.py       # Crossover mutation
│   ├── prompts_init.py        # Initial program generation
│   ├── prompts_meta.py        # Meta-summarization (3-step pipeline)
│   └── prompts_novelty.py     # Novelty assessment
│
├── edit/                      # Code patch application
│   ├── apply_diff.py          # apply_diff_patch() — EVOLVE-BLOCK aware
│   ├── apply_full.py          # apply_full_patch() — full rewrites
│   ├── async_apply.py         # Async variants
│   └── summary.py             # Diff summarization, immutable redaction
│
├── launch/                    # Job execution backends
│   ├── scheduler.py           # JobScheduler, JobConfig variants
│   ├── local.py               # Local subprocess execution
│   └── slurm.py               # Slurm (Docker/Conda) execution
│
├── plots/                     # Visualization utilities
│   ├── plot_lineage_tree.py   # Evolution lineage tree (NetworkX)
│   ├── plot_improvement.py    # Best score over generations
│   ├── plot_pareto.py         # 2D Pareto front
│   ├── plot_similarity.py     # Embedding similarity heatmap
│   └── code_path_anim.py      # Code evolution video (MoviePy)
│
├── webui/                     # Interactive web UI
│   ├── visualization.py       # HTTP server + JSON API
│   └── viz_tree.html          # Single-page D3.js frontend
│
├── utils/                     # Shared helpers
│   ├── utils_hydra.py         # Hydra config loading, evolve markers
│   ├── general.py             # General utilities
│   └── load_df.py             # DataFrame loading from results
│
├── launch_hydra.py            # Hydra entry point (@hydra.main)
├── eval_hydra.py              # Hydra evaluation launcher
├── logo.py                    # Gradient logo
├── cuco_launch                # Bash entry point for Hydra
└── cuco_visualize             # Python entry point for web UI
```

## Key Abstractions

### Program

A `Program` (defined in `database/dbase.py`) is the central data object. Each candidate kernel that enters the system becomes a Program with:

- **Identity**: unique `id`, `code` (source text), `language`
- **Lineage**: `parent_id`, `archive_inspiration_ids`, `top_k_inspiration_ids`, `island_idx`, `generation`
- **Metrics**: `combined_score`, `public_metrics` (timing), `private_metrics`, `text_feedback`, `correct` (bool)
- **Embeddings**: `embedding` (for novelty/similarity), `embedding_pca_2d/3d`, `embedding_cluster_id`
- **Metadata**: `complexity`, `code_diff`, `migration_history`

### ProgramDatabase

SQLite-backed persistent store for all evaluated candidates. Provides:
- `add()` / `get()` — CRUD for programs
- `sample()` — parent + archive inspirations + top-k inspirations
- `get_best_program()` / `get_top_programs()` — fitness-ordered retrieval
- `compute_similarity()` — cosine similarity against stored embeddings
- Island management, archive maintenance, and embedding-guided retrieval

### EvolutionRunner

The main orchestrator (`core/runner.py`). Manages:
- Pre-transform pipeline (optional fast-path)
- Generation 0 initialization from seed
- Parallel job submission and completion
- Patch generation (via `PromptSampler` + `LLMClient`)
- Novelty filtering (via `NoveltyJudge`)
- Meta-summarization (via `MetaSummarizer`)

### HostToDeviceTransformer

The fast-path workhorse (`transform/transformer.py`). Runs an LLM-driven build/verify loop:
- Sends current code + error feedback to the LLM
- Compiles with nvcc
- Runs with mpirun
- LLM judge analyzes failures and provides corrective feedback
- Repeats until verification passes or iteration budget exhausts

### PromptSampler

Assembles mutation prompts (`core/sampler.py`) by combining:
- Task system message (workload-specific constraints, API knowledge, hardware context)
- Mutation format instructions (diff / full / cross)
- Parent code + evaluation history
- Archive inspirations + top-k programs
- Meta-recommendations

## Data Flow

```
1. User provides:
   - Seed kernel (.cu file with host NCCL)
   - evaluate.py (build, run, score)
   - nccl_api_docs.py (API reference for LLM context)

2. Fast-path (optional):
   CUDAAnalyzer ──► HostToDeviceTransformer ──► insert_evolve_markers
   Output: device-initiated .cu with EVOLVE-BLOCK markers

3. Evolution loop (per generation):
   ProgramDatabase.sample() ──► PromptSampler.sample()
         │                              │
         │ parent + inspirations        │ system + user prompt
         │                              ▼
         │                        LLMClient.query()
         │                              │
         │                              │ code patch
         │                              ▼
         │                    apply_diff_patch / apply_full_patch
         │                              │
         │                              │ candidate .cu file
         │                              ▼
         │                    JobScheduler.submit_async()
         │                              │
         │                              │ evaluate.py
         │                              ▼
         │                    metrics.json + correct.json
         │                              │
         └──────────── ProgramDatabase.add() ◄──┘

4. Periodic: MetaSummarizer distills patterns into recommendations
5. Final: best candidate retrieved from database
```
