# Fast-Path Agent

The fast-path agent converts host-driven CUDA+NCCL programs into device-initiated equivalents (GIN or LSA). It prioritizes **correctness over performance**, producing a conservative but verified baseline that the slow-path agent can then optimize.

## Pipeline Overview

```
  Host-driven .cu            CUDA Analysis           Host-to-Device            Evolve-Block
  (ncclAlltoAll, etc.)  ──►  (CUDAAnalyzer)  ──►    Transformation    ──►    Annotation
                              │                       (LLM + Judge)           (LLM + regex)
                              │                       │                       │
                              ▼                       ▼                       ▼
                         AnalysisReport          Verified device        Annotated seed
                         (comm graph)            .cu file               with EVOLVE-BLOCK
```

The pipeline is implemented in two layers:
- **PreTransformPipeline** (`cuco/transform/pipeline.py`) — orchestrates the ordered steps
- **HostToDeviceTransformer** (`cuco/transform/transformer.py`) — runs the LLM build/verify loop

## Step 1: CUDA Analysis

**Module**: `cuco/transform/cuda_analyzer.py`
**Class**: `CUDAAnalyzer`

The analyzer extracts a communication dependency graph from the input source using regex-based pattern matching. It identifies:

- **NCCL collectives**: `ncclAllReduce`, `ncclAllGather`, `ncclAlltoAll`, `ncclSend`, `ncclRecv`, `ncclGroupStart/End`
- **Memory allocations**: `cudaMalloc` vs. `ncclMemAlloc` (device APIs require the latter)
- **Kernel launches**: names, grid/block dimensions, stream assignments
- **Synchronization points**: `cudaStreamSynchronize`, `cudaDeviceSynchronize`, `MPI_Barrier`
- **Communication graph**: For each collective, which buffers are sent/received, which kernels produce/consume them

### Output: AnalysisReport

```python
analyzer = CUDAAnalyzer("my_kernel.cu")
report = analyzer.analyze()

# Check if host-side collectives exist
if report.has_host_communication():
    # Feed to the transformer
    llm_context = report.format_for_llm()
```

`format_for_llm()` produces a structured text summary that the rewriter LLM uses to understand the communication pattern, data dependencies, and which buffers need migration from `cudaMalloc` to `ncclMemAlloc`.

## Step 2: Host-to-Device Transformation

**Module**: `cuco/transform/transformer.py`
**Class**: `HostToDeviceTransformer`

This is the core of the fast-path agent. It runs an LLM-driven feedback loop:

```
                    ┌─────────────────────┐
                    │  LLM Rewrite        │
                    │  (code generation)  │
                    └────────┬────────────┘
                             │ candidate .cu
                             ▼
                    ┌─────────────────────┐
                    │  Build (nvcc)       │
                    └────────┬────────────┘
                             │
                      success│ fail → error feedback ──► back to LLM
                             ▼
                    ┌─────────────────────┐
                    │  Run (mpirun)       │
                    └────────┬────────────┘
                             │
                      pass   │ fail → diagnostic rerun ──► back to LLM
                             ▼
                    ┌─────────────────────┐
                    │  LLM Judge          │
                    │  (analyzes result)  │
                    └────────┬────────────┘
                             │
                    verified │ issues found ──► corrective feedback ──► back to LLM
                             ▼
                         Success!
```

### Two-Stage Mode

By default (`two_stage=True`), transformation is split into two stages:

**Stage A: Infrastructure Setup**

The LLM adds device-side NCCL infrastructure while keeping host collectives intact:
- Replace `cudaMalloc` with `ncclMemAlloc` for communication buffers
- Create and register NCCL windows (`ncclCommWindowRegister`)
- Configure device communicator requirements (`ncclDevCommRequirements`)
- Instantiate the device communicator (`ncclDevCommCreate`)
- Set up cooperative kernel launch infrastructure

The host-side NCCL collectives (`ncclAlltoAll`, etc.) remain unchanged. This isolates infrastructure errors from communication logic errors.

**Stage B: Collective Replacement**

With infrastructure in place, the LLM replaces host collectives with device-initiated equivalents:
- For GIN: `ncclAlltoAll` → `gin.put()` + `gin.flush()` + `gin.waitSignal()`
- For LSA: `ncclAlltoAll` → `ncclGetLsaPointer()` + direct stores + `ncclLsaBarrierSession.sync()`

All directives are set conservatively:
- CTA-level issuance (`ncclCoopCta`) to avoid warp-level divergence
- Fully deferred placement to minimize ordering complexity
- Global synchronization scope for cross-rank visibility
- Coarse transfer granularity

### Single-Stage Mode

Set `two_stage=False` for simpler programs where infrastructure and collective replacement can be done together.

### Diagnostic Rerun

When a runtime failure occurs, the transformer can inject `cudaDeviceSynchronize()` after each kernel launch to isolate which kernel is causing the fault. This provides much more targeted feedback to the LLM than a generic crash report.

### Convergence

The combined loop typically converges in **2-4 iterations per stage**. Stage A usually succeeds in 1-2 iterations (infrastructure is more formulaic). Stage B may need 3-4 iterations due to the complexity of synchronization semantics.

## Step 3: Evolve-Block Annotation

**Function**: `insert_evolve_markers()` in `transformer.py`

After transformation, the code is annotated with mutable-region markers:

```cuda
// EVOLVE-BLOCK-START
// ... kernel definitions and pipeline logic ...
// EVOLVE-BLOCK-END
```

The annotator uses LLM analysis with regex fallback:
1. **LLM pass**: Ask the LLM to identify which regions are safe to mutate
2. **Regex fallback**: If LLM fails, use pattern matching to find kernel definitions and pipeline sections

Frozen regions are explicitly excluded:
- MPI/NCCL initialization and teardown
- Verification and output formatting
- Main function structure

## Step 4: Warmup Injection (optional)

The pipeline can optionally inject communication warmup before the timed section. The first GIN/LSA/NCCL call triggers lazy RDMA/NIC initialization (10-50 ms). Warmup rounds amortize this cost.

The warmup step:
1. Asks the LLM to add 2 rounds of dummy communication calls before timing starts
2. Builds and verifies the result compiles and runs correctly
3. Skips if warmup is already detected in the code

## PreTransformPipeline

**Module**: `cuco/transform/pipeline.py`
**Class**: `PreTransformPipeline`

Orchestrates the four steps in order, with skip logic:

```python
pipeline = PreTransformPipeline(
    config=transform_config,
    steps=["analyze", "host_to_device", "evolve_markers", "warmup"],
)
result = pipeline.run(source_path="my_kernel.cu", output_dir="_transform_output")
```

<table>
<tr><th>Step</th><th>Method</th><th>Skipped when</th></tr>
<tr><td><code>analyze</code></td><td>Regex (Python)</td><td>Never — always runs</td></tr>
<tr><td><code>host_to_device</code></td><td>LLM loop</td><td>No host NCCL collectives detected</td></tr>
<tr><td><code>evolve_markers</code></td><td>LLM + regex</td><td>EVOLVE-BLOCK markers already present</td></tr>
<tr><td><code>warmup</code></td><td>LLM + build/verify</td><td>Warmup section already detected</td></tr>
</table>

Each step produces a `PipelineStepResult` with timing, cost, and error information.

## Agent Mode vs. Structured Loop

CUCo offers two modes for the fast-path transformation:

### Agent Mode (default)

Uses Claude Code CLI (`claude -p`) with full file system autonomy. The agent:
- Reads the source file
- Iteratively edits, builds, and runs until verification passes
- Has access to Bash, Read, Write, and Edit tools

```bash
python run_transform.py  # Agent mode (default)
```

Advantages: More flexible, can handle unexpected edge cases.
Disadvantages: Higher cost, less predictable.

### Structured Loop Mode

Uses the `HostToDeviceTransformer` with a fixed rewrite-build-judge cycle:

```bash
python run_transform.py --no-agent
```

Advantages: Deterministic, lower cost, easier to debug.
Disadvantages: Less flexible when the transformation requires creative solutions.

## Configuration

See [Configuration Reference](configuration.md) for all `TransformConfig` parameters. The most important ones:

<table>
  <tr>
    <th>Parameter</th>
    <th>Default</th>
    <th>Description</th>
  </tr>
  <tr>
    <td><code>api_type</code></td>
    <td><code>"gin"</code></td>
    <td>Target API: <code>"gin"</code> or <code>"lsa"</code></td>
  </tr>
  <tr>
    <td><code>two_stage</code></td>
    <td><code>True</code></td>
    <td>Split into infrastructure + replacement stages</td>
  </tr>
  <tr>
    <td><code>max_iterations</code></td>
    <td>5</td>
    <td>Max iterations (single-stage mode)</td>
  </tr>
  <tr>
    <td><code>stage_a_max_iterations</code></td>
    <td>5</td>
    <td>Max iterations for infrastructure stage</td>
  </tr>
  <tr>
    <td><code>stage_b_max_iterations</code></td>
    <td>10</td>
    <td>Max iterations for replacement stage</td>
  </tr>
  <tr>
    <td><code>rewrite_model</code></td>
    <td>Sonnet 4.6</td>
    <td>LLM for code generation</td>
  </tr>
  <tr>
    <td><code>judge_model</code></td>
    <td><code>""</code> (same)</td>
    <td>LLM for judge feedback</td>
  </tr>
  <tr>
    <td><code>reference_code</code></td>
    <td><code>""</code></td>
    <td>Working device-side example to show the LLM</td>
  </tr>
  <tr>
    <td><code>nccl_api_docs</code></td>
    <td><code>""</code></td>
    <td>NCCL API documentation string</td>
  </tr>
  <tr>
    <td><code>verification_pass_str</code></td>
    <td><code>"Verification: PASS"</code></td>
    <td>Expected output for success</td>
  </tr>
</table>
