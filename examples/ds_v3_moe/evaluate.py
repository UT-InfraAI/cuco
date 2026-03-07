#!/usr/bin/env python3
"""
Generic evaluation harness for CUDA kernel evolution.

Pipeline:
1. Build the evolved .cu file.
2. Run with mpirun, check "Verification: PASS", parse "Time: X.XXXX ms".
3. Run a second time for stability (take the better of two runs).
4. Provide structured feedback: factual code analysis + LLM suggestions.
5. Write metrics.json and correct.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_CUDA_EVOLVE_DIR = Path(__file__).resolve().parent.parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(_CUDA_EVOLVE_DIR / ".env")
except ImportError:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Build and run configuration
# ---------------------------------------------------------------------------
NVCC = "/usr/local/cuda-13.1/bin/nvcc"
NCCL_INCLUDE = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/include"
NCCL_STATIC_LIB = "/usr/local/nccl_2.28.9-1+cuda13.0_x86_64/lib/libnccl_static.a"
CUDA_LIB64 = "/usr/local/cuda-13.1/lib64"
MPI_INCLUDE = "/usr/lib/x86_64-linux-gnu/openmpi/include"
MPI_INCLUDE_OPENMPI = "/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi"
MPI_LIB = "/usr/lib/x86_64-linux-gnu/openmpi/lib"
MPI_NP = 2

NUM_RUNS = 2
RUN_TIMEOUT = 120

TIME_PATTERN = re.compile(r"^Time:\s*([\d.]+)\s*ms", re.MULTILINE)
RANK_HEADER_PATTERN = re.compile(
    r"RESULTS \(Rank (\d+).*?receives (\d+) tokens\).*?\nTime:\s*([\d.]+)\s*ms",
    re.DOTALL,
)
VERIFICATION_PASS_STR = "Verification: PASS"

# ---------------------------------------------------------------------------
# LLM feedback configuration
# ---------------------------------------------------------------------------
LLM_FEEDBACK_ENABLED = True
FEEDBACK_LLM_MODEL = os.environ.get(
    "FEEDBACK_LLM_MODEL",
    "bedrock/us.anthropic.claude-opus-4-6-v1",
)
FEEDBACK_MAX_TOKENS = 1024

_EXAMPLE_DIR = Path(__file__).resolve().parent
if str(_EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLE_DIR))

_NCCL_GIN_CONTEXT = ""
_NCCL_LSA_CONTEXT = ""
try:
    from nccl_api_docs import (
        NCCL_HEADER_GIN_H,
        NCCL_HEADER_CORE_H,
        NCCL_HEADER_COOP_H,
        NCCL_HEADER_BARRIER_H,
    )
    _NCCL_GIN_CONTEXT = "\n\n".join([NCCL_HEADER_GIN_H, NCCL_HEADER_CORE_H, NCCL_HEADER_COOP_H])
    _NCCL_LSA_CONTEXT = "\n\n".join([NCCL_HEADER_CORE_H, NCCL_HEADER_COOP_H, NCCL_HEADER_BARRIER_H])
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Feedback prompt — built per-program based on detected API
# ---------------------------------------------------------------------------
_FEEDBACK_BASE = """You are an expert CUDA and NCCL device-side kernel reviewer.
You will be given a CUDA kernel program, its runtime results, and a factual
code analysis. Your job: concise, actionable diagnosis with 2-3 suggestions.

## Goal

Minimize kernel runtime while maintaining correctness ("Verification: PASS").
Only code inside EVOLVE-BLOCK markers may be changed.

## Compute-Communication Strategies

- **Fused kernel**: Single persistent kernel interleaving compute tiles with
  communication (e.g., warp specialization, tile-and-send). Best when the
  workload is iterative with fine-grained produce/consume cycles.
- **Multi-stream overlap**: Compute and communication on separate CUDA streams,
  synchronized via events. Best when compute phases are large and self-contained.
- **Split put/wait**: Separate PUT (initiate) and WAIT (complete) kernels with
  useful compute in between. Maximizes the overlap window.
- **Sequential**: All operations serialized on one stream. No overlap.

These are composable — a program can combine strategies. The right choice
depends on the workload's data-dependency structure.

## Key Principles

- With overlap, total time ≈ max(compute_path, comm_path), not their sum.
- Focus on the CRITICAL PATH — improving something that already runs
  concurrently does not help wall-clock time.
- Communication kernels should use minimal GPU resources (1 block, 32 threads).
- Event-based sync (cudaEventRecord + cudaStreamWaitEvent) instead of
  cudaStreamSynchronize which blocks the host.

## Hard Constraints on Suggestions
- Do NOT suggest adding external libraries (cuBLAS, cuDNN, cuSPARSE, Thrust,
  CUB, etc.). The build environment links ONLY cuda, nccl, and mpi. Any
  suggestion requiring additional library linkage will fail at compile time.
- All improvements must use hand-written CUDA kernels, NCCL device APIs, and
  standard CUDA runtime APIs only.
- Compute kernel internals are FIXED — tile sizes, thread-block dimensions,
  and arithmetic logic must NOT be changed. Treat their runtimes as given
  constants. Do NOT suggest optimizing compute kernel internals.
- The ONLY levers for reducing total time are: overlapping compute with
  communication (fusing kernels, multi-stream, split put/wait, pipelining tiles with puts),
  reordering phases to shorten the critical path, reducing synchronization
  overhead, and minimizing idle time across ranks.

IMPORTANT: Be concise (under 200 words). Focus on:
1. What strategy is this program using and what is the biggest bottleneck?
2. Given that compute kernel runtimes are fixed, what overlap or pipeline
   change would have the highest impact on wall-clock time?
Do NOT repeat the code back. Do NOT suggest rewriting from scratch."""

_INSTRUMENT_PROMPT = """You are a CUDA profiling expert. Given a CUDA program, insert
cudaEventRecord and cudaEventElapsedTime calls to measure each distinct phase
between the existing ev_start and ev_stop timing markers.

Rules:
- Create a cudaEvent_t pair for each phase boundary.
- After ev_stop, print each phase as:
    printf("Phase: <descriptive_name> = %.4f ms\\n", elapsed);
  Use short descriptive names like "quantize", "gin_dispatch", "dequantize",
  "gemm1", "swiglu", "gemm2", "gin_combine", etc.
- Do NOT change any kernel logic, launch configs, or buffer operations.
- Do NOT remove the existing ev_start / ev_stop markers or their elapsed-time
  print. The phase timers are ADDITIONAL instrumentation.
- Keep verification and all other program logic intact.
- cudaEventRecord takes a stream argument — use 0 (default stream) unless the
  phase runs on a named stream, in which case use that stream.
- For phases on separate streams, record events on each stream and synchronize
  appropriately before calling cudaEventElapsedTime.
- Return the COMPLETE modified file (not a diff)."""

_GIN_RULES = """
## GIN API Rules (this program uses GIN)

- ncclGin_SignalInc{index} on the last put to a peer. ncclGin_None{} on earlier puts.
- flush(Coop) must match the threads that issued puts.
- __syncthreads() required between buffer writes and GIN puts.
- ginSignalCount must be >= number of distinct signal indices used.
- ginContextCount should match number of independent communication phases.
- GIN puts offload to NIC DMA — uses ~0 SMs. Ideal for overlap with compute.
- GIN wait kernels should use minimal threads (1 block, 32 threads).
"""

_LSA_RULES = """
## LSA API Rules (this program uses LSA)

- ncclGetLsaPointer(window, byte_offset, peer_rank) for peer memory access.
- ncclLsaBarrierSession for synchronization: memory_order_release after writes,
  memory_order_relaxed before reads.
- lsaBarrierCount must be >= number of CTA blocks creating barrier sessions.
- LSA uses GPU SMs for data movement (load/store), so it contends with compute.
- Best for intranode (NVLink). For internode, GIN is typically better.
"""


def _build_feedback_prompt(api_mode: str) -> str:
    """Build API-specific feedback prompt based on detected communication API."""
    prompt = _FEEDBACK_BASE

    if api_mode == "gin":
        prompt += "\n" + _GIN_RULES
        if _NCCL_GIN_CONTEXT:
            prompt += "\n\n## NCCL GIN Headers\n\n" + _NCCL_GIN_CONTEXT
    elif api_mode == "lsa":
        prompt += "\n" + _LSA_RULES
        if _NCCL_LSA_CONTEXT:
            prompt += "\n\n## NCCL LSA Headers\n\n" + _NCCL_LSA_CONTEXT
    elif api_mode == "both":
        prompt += "\n" + _GIN_RULES + "\n" + _LSA_RULES
        if _NCCL_GIN_CONTEXT:
            prompt += "\n\n## NCCL GIN Headers\n\n" + _NCCL_GIN_CONTEXT
        if _NCCL_LSA_CONTEXT:
            prompt += "\n\n## NCCL LSA Headers\n\n" + _NCCL_LSA_CONTEXT
    else:
        prompt += "\n(No device-side communication API detected — host NCCL only.)"

    return prompt


# ---------------------------------------------------------------------------
# LLM calling
# ---------------------------------------------------------------------------

def _call_llm(system_prompt: str, user_msg: str, max_tokens: int = FEEDBACK_MAX_TOKENS) -> str:
    if not LLM_FEEDBACK_ENABLED:
        return ""

    model_name = FEEDBACK_LLM_MODEL

    if model_name.startswith("claude-cli/"):
        return _call_claude_cli(model_name, system_prompt, user_msg)
    return _call_anthropic(model_name, system_prompt, user_msg, max_tokens)


def _call_claude_cli(model_name: str, system_prompt: str, user_msg: str) -> str:
    import subprocess as _sp
    model_map = {"claude-cli/opus": "opus", "claude-cli/sonnet": "sonnet", "claude-cli/haiku": "haiku"}
    model_alias = model_map.get(model_name, "haiku")
    cmd = [
        "claude", "-p",
        "--model", model_alias,
        "--output-format", "text",
        "--dangerously-skip-permissions",
        "--no-session-persistence",
        "--tools", "",
        "--system-prompt", system_prompt,
    ]
    env = os.environ.copy()
    for p in ["/usr/bin", "/usr/local/bin", os.path.expanduser("~/.local/bin")]:
        if p not in env.get("PATH", ""):
            env["PATH"] = p + ":" + env.get("PATH", "")
    try:
        result = _sp.run(cmd, input=user_msg, capture_output=True, text=True, timeout=120, env=env)
        return (result.stdout or "").strip()
    except Exception as exc:
        logger.warning(f"Claude CLI feedback failed: {exc}")
        return ""


def _call_anthropic(model_name: str, system_prompt: str, user_msg: str, max_tokens: int) -> str:
    try:
        import anthropic
        if model_name.startswith("bedrock/"):
            actual_model = model_name.split("/", 1)[1]
            client = anthropic.AnthropicBedrock(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
                aws_region=os.environ.get("AWS_REGION_NAME"),
            )
        else:
            actual_model = model_name
            client = anthropic.Anthropic()

        with client.messages.stream(
            model=actual_model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_msg}],
            max_tokens=max_tokens,
            temperature=0.0,
        ) as stream:
            response = stream.get_final_message()

        if response.content and len(response.content) > 0:
            return response.content[0].text.strip()
        return ""
    except Exception as exc:
        logger.warning(f"Anthropic feedback failed: {exc}")
        return ""


# ---------------------------------------------------------------------------
# Standard evaluation functions
# ---------------------------------------------------------------------------

def get_build_command(work_dir: Path, source_name: str = "main.cu", binary_name: str = "main") -> list[str]:
    return [
        NVCC, "-o", binary_name, str(work_dir / source_name),
        f"-I{NCCL_INCLUDE}", NCCL_STATIC_LIB,
        "-rdc=true", "-arch=sm_80",
        f"-L{CUDA_LIB64}", "-lcudart", "-lcudadevrt", "-lpthread",
        f"-I{MPI_INCLUDE}", f"-I{MPI_INCLUDE_OPENMPI}",
        f"-L{MPI_LIB}", "-lmpi",
    ]


HOSTFILE = str(Path(__file__).resolve().parent / "build" / "hostfile")

# GPU architecture lookup: sm_XX -> (name, SMs, HBM bandwidth, L2 cache, shared mem/SM)
_GPU_ARCH_INFO = {
    "sm_70": ("V100 (Volta)",       80,  "~900 GB/s",  "6 MB",   "96 KB"),
    "sm_75": ("T4 (Turing)",        40,  "~300 GB/s",  "4 MB",   "64 KB"),
    "sm_80": ("A100 (Ampere)",     108,  "~2 TB/s",   "40 MB", "164 KB"),
    "sm_86": ("A10/A40 (Ampere)",   84,  "~600 GB/s",  "6 MB",   "100 KB"),
    "sm_89": ("L40/L4 (Ada)",      142,  "~864 GB/s", "96 MB", "100 KB"),
    "sm_90": ("H100 (Hopper)",     132,  "~3.35 TB/s","50 MB", "228 KB"),
    "sm_100": ("B200 (Blackwell)", 192,  "~8 TB/s",   "96 MB", "228 KB"),
}


def _extract_arch_from_build() -> str:
    """Extract the -arch=sm_XX flag from get_build_command."""
    for flag in get_build_command(Path(".")):
        if flag.startswith("-arch="):
            return flag.split("=", 1)[1]
    return "unknown"


def _extract_version(path: str, prefix: str) -> str:
    """Extract version string from a path like /usr/local/cuda-13.1/..."""
    import re as _re
    m = _re.search(rf'{prefix}[-_]?([\d.]+)', path)
    return m.group(1) if m else "unknown"


def _detect_interconnect() -> Tuple[str, str, bool]:
    """Detect interconnect type from _run_binary env vars. Returns (network_type, detail, is_internode)."""
    build_cmd_str = " ".join(get_build_command(Path(".")))
    run_cmd = [
        "mpirun", "--hostfile", HOSTFILE,
        "-np", str(MPI_NP), "--map-by", "node",
        "-x", "NCCL_GIN_ENABLE=1",
        "-x", f"NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA', 'mlx5_1')}",
    ]
    run_str = " ".join(run_cmd)

    is_internode = "--map-by" in run_str and "node" in run_str
    if "NCCL_IB_HCA" in run_str or "mlx5" in run_str:
        return "InfiniBand", "Mellanox ConnectX (mlx5)", is_internode
    return "NVLink (intra-node)", "", is_internode


def get_hardware_context() -> str:
    """Build hardware context string dynamically from evaluate.py configuration."""
    arch = _extract_arch_from_build()
    gpu_info = _GPU_ARCH_INFO.get(arch)
    cuda_ver = _extract_version(NVCC, "cuda")
    nccl_ver = _extract_version(NCCL_INCLUDE, "nccl")
    net_type, net_detail, is_internode = _detect_interconnect()

    if gpu_info:
        gpu_name, sm_count, hbm_bw, l2_size, smem_size = gpu_info
    else:
        gpu_name, sm_count, hbm_bw, l2_size, smem_size = (
            f"Unknown ({arch})", "?", "?", "?", "?"
        )

    topology = "Inter-node (separate machines)" if is_internode else "Intra-node (same machine)"
    net_line = f"- **Network**: {net_type}"
    if net_detail:
        net_line += f" ({net_detail})"

    if is_internode:
        peer_line = (
            "- **Peer connectivity**: Ranks are NOT NVLink-connected (different nodes).\n"
            "  LSA (load/store access) is unavailable across nodes. GIN (GPU-Initiated\n"
            "  Networking) is the correct backend for this topology."
        )
    else:
        peer_line = (
            "- **Peer connectivity**: Ranks are NVLink-connected (same node).\n"
            "  Both LSA (direct load/store) and GIN are available. LSA avoids\n"
            "  NIC overhead for intra-node transfers."
        )

    sm_count_str = str(sm_count)
    budget_lines = (
        f"- With {sm_count_str} SMs, a communication kernel using 1 block (1 SM) leaves "
        f"{sm_count - 1 if isinstance(sm_count, int) else '?'} SMs\n"
        f"  for compute — negligible overhead."
    )
    if is_internode:
        budget_lines += (
            "\n- GIN puts offload to NIC DMA (zero SM cost after initiation). This means\n"
            "  GIN communication and GPU compute can run truly in parallel without\n"
            "  contending for SMs or HBM bandwidth.\n"
            "- For split put/wait: the PUT kernel occupies 1 SM briefly to initiate\n"
            "  transfers, then returns. The WAIT kernel polls NIC signals on 1 SM.\n"
            "  Both are lightweight."
        )
    else:
        budget_lines += (
            "\n- LSA transfers use GPU load/store units and share NVLink bandwidth\n"
            "  with compute peer-memory accesses. Barrier sync requires dedicated CTAs.\n"
            "- GIN (if used intra-node) still offloads to NIC DMA, but NVLink-based\n"
            "  LSA may have lower latency for small transfers."
        )

    return f"""
## Hardware Context

This program is compiled and executed on the following hardware. Use this
information to guide kernel design decisions (block counts, overlap strategy,
backend selection, resource budgeting).

### GPU Architecture
- **GPU**: NVIDIA {gpu_name}
- **Architecture**: {arch}
- **SMs**: {sm_count_str} streaming multiprocessors
- **HBM bandwidth**: {hbm_bw}
- **L2 cache**: {l2_size}
- **Shared memory per SM**: {smem_size} (configurable)
- **Max threads per SM**: 2048
- **Max threads per block**: 1024
- **Warp size**: 32

### Topology & Interconnect
- **Ranks**: {MPI_NP} (one GPU per node, mapped by node)
- **Topology**: {topology}
{net_line}
{peer_line}

### Build Environment
- **CUDA**: {cuda_ver}
- **NCCL**: {nccl_ver} (device API with GIN/LSA support)
- **Compilation**: nvcc -rdc=true -arch={arch} (relocatable device code)
- **MPI**: OpenMPI (mpirun with hostfile, {MPI_NP} processes)

### Resource Budgeting Implications
{budget_lines}
"""


def get_run_env() -> Dict[str, str]:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{CUDA_LIB64}:{MPI_LIB}:{env.get('LD_LIBRARY_PATH', '')}"
    env["CUDA_VISIBLE_DEVICES"] = "1"
    return env


def parse_all_rank_times(stdout: str) -> Dict[int, Tuple[float, int]]:
    """Parse per-rank (time_ms, tokens) from 'RESULTS (Rank N ... receives M tokens)' blocks."""
    result: Dict[int, Tuple[float, int]] = {}
    for m in RANK_HEADER_PATTERN.finditer(stdout):
        rank_id = int(m.group(1))
        tokens = int(m.group(2))
        time_ms = float(m.group(3))
        result[rank_id] = (time_ms, tokens)
    return result


def compute_weighted_time(rank_times: Dict[int, Tuple[float, int]]) -> Optional[float]:
    """Token-weighted average time across ranks."""
    if not rank_times:
        return None
    total_tokens = sum(tokens for _, tokens in rank_times.values())
    if total_tokens == 0:
        return None
    return sum(t * tokens / total_tokens for t, tokens in rank_times.values())


def parse_time_ms(stdout: str) -> Optional[float]:
    """Parse timing: prefer token-weighted average from multi-rank output,
    fall back to first 'Time:' line for single-rank or legacy output."""
    rank_times = parse_all_rank_times(stdout)
    if rank_times:
        return compute_weighted_time(rank_times)
    m = TIME_PATTERN.search(stdout)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def score_from_time_ms(time_ms: float) -> float:
    if time_ms <= 0:
        return 0.0
    return 10000.0 / (1.0 + time_ms)


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _get_best_score_so_far(results_path: Path) -> Optional[float]:
    try:
        evo_root = results_path.parent.parent
        best_score = None
        for metrics_file in evo_root.glob("gen_*/results/metrics.json"):
            if metrics_file.parent == results_path:
                continue
            try:
                data = json.loads(metrics_file.read_text(encoding="utf-8"))
                score = data.get("combined_score", 0.0)
                if best_score is None or score > best_score:
                    best_score = score
            except Exception:
                continue
        return best_score
    except Exception:
        return None


def _run_binary(work_dir: Path, binary_path: Path, env: Dict[str, str]) -> Tuple[int, str, str]:
    result = subprocess.run(
        [
            "mpirun", "--hostfile", HOSTFILE,
            "-np", str(MPI_NP),
            "--map-by", "node",
            "-x", "LD_LIBRARY_PATH",
            "-x", "CUDA_VISIBLE_DEVICES",
            "-x", "NCCL_GIN_ENABLE=1",
            "-x", "NCCL_SOCKET_IFNAME=enp75s0f1np1",
            "-x", "NCCL_IB_HCA=mlx5_1",
            "-x", "NCCL_IB_GID_INDEX=3",
            "--mca", "btl_tcp_if_include", "enp75s0f1np1",
            "--mca", "oob_tcp_if_include", "enp75s0f1np1",
            str(binary_path),
        ],
        cwd=str(work_dir), env=env, capture_output=True, text=True,
        timeout=RUN_TIMEOUT,
    )
    return result.returncode, result.stdout or "", result.stderr or ""


# ---------------------------------------------------------------------------
# Code analysis — factual, not prescriptive
# ---------------------------------------------------------------------------

def _detect_api_mode(code: str) -> str:
    """Detect which communication API the code uses: 'gin', 'lsa', 'both', or 'host'."""
    has_gin = bool(re.search(r'ncclGin\b|gin\.put|gin\.flush|gin\.waitSignal|ginContextCount', code))
    has_lsa = bool(re.search(r'ncclLsaBarrierSession|ncclGetLsaPointer|lsaBarrierCount', code))

    if has_gin and has_lsa:
        return "both"
    if has_gin:
        return "gin"
    if has_lsa:
        return "lsa"
    return "host"


# ---------------------------------------------------------------------------
# Per-phase profiling via LLM instrumentation
# ---------------------------------------------------------------------------

_PHASE_PATTERN = re.compile(r"^Phase:\s*(.+?)\s*=\s*([\d.]+)\s*ms", re.MULTILINE)


def _extract_cuda_from_response(response: str) -> str:
    """Strip markdown code fences from an LLM response to get raw CUDA code."""
    fenced = re.search(r"```(?:cuda|cpp|c\+\+|c)?\s*\n(.*?)```", response, re.DOTALL)
    if fenced:
        return fenced.group(1)
    return response


def _try_instrument(code_text: str) -> Optional[str]:
    """Ask an LLM to add per-phase cudaEvent timers to *code_text*.

    Returns the instrumented CUDA source as a string, or None on any failure.
    The caller is responsible for building and running the result.
    """
    try:
        response = _call_llm(_INSTRUMENT_PROMPT, code_text, max_tokens=16384)
        if not response:
            return None
        instrumented = _extract_cuda_from_response(response)
        if len(instrumented) < 200:
            return None
        return instrumented
    except Exception as exc:
        logger.debug("Phase-timing instrumentation LLM call failed: %s", exc)
        return None


def _parse_phase_lines(stdout: str) -> str:
    """Extract ``Phase: name = X.XXXX ms`` lines from program stdout.

    Returns a formatted multi-line string, or "" if no phases found.
    """
    phases = _PHASE_PATTERN.findall(stdout)
    if not phases:
        return ""
    return "\n".join(f"  {name}: {float(ms):.4f} ms" for name, ms in phases)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main(program_path: str, results_dir: str) -> None:
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    program_file = Path(program_path).resolve()
    if not program_file.exists():
        write_json(results_path / "metrics.json", {
            "combined_score": 0.0, "public": {},
            "private": {"error": f"Program file not found: {program_path}"}
        })
        write_json(results_path / "correct.json", {"correct": False, "error": f"Not found: {program_path}"})
        return

    work_dir = program_file.parent
    source_name = program_file.name
    binary_name = program_file.stem
    source_path = work_dir / source_name
    binary_path = work_dir / binary_name

    metrics: Dict = {}
    correct = False
    error_msg = ""

    try:
        code_text = program_file.read_text(encoding="utf-8")

        # Detect API for feedback prompt
        api_mode = _detect_api_mode(code_text)
        feedback_prompt = _build_feedback_prompt(api_mode)

        # Try LLM instrumentation so phase timers are baked into the build
        instrumented_code = _try_instrument(code_text)
        build_code = instrumented_code if instrumented_code else code_text
        source_path.write_text(build_code, encoding="utf-8")

        # Build
        build_cmd = get_build_command(work_dir, source_name, binary_name)
        build_log = results_path / "build.log"
        start = time.perf_counter()
        result = subprocess.run(build_cmd, cwd=str(work_dir), capture_output=True, text=True)
        build_duration = time.perf_counter() - start
        build_log.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")

        if result.returncode != 0 and instrumented_code is not None:
            logger.debug("Instrumented build failed, falling back to original code")
            instrumented_code = None
            source_path.write_text(code_text, encoding="utf-8")
            start = time.perf_counter()
            result = subprocess.run(build_cmd, cwd=str(work_dir), capture_output=True, text=True)
            build_duration = time.perf_counter() - start
            build_log.write_text((result.stdout or "") + "\n" + (result.stderr or ""), encoding="utf-8")

        if result.returncode != 0:
            build_errors = (result.stderr or result.stdout or "")[:2000]
            error_msg = f"Build failed (exit {result.returncode})\n{build_errors}"
            llm_feedback = _call_llm(feedback_prompt, (
                f"## Evolved Program\n\n```cuda\n{code_text[:8000]}\n```\n\n"
                f"## Outcome\n\nBUILD FAILED (exit {result.returncode}).\n"
                f"Compiler errors:\n{build_errors[:3000]}\n\n"
                "Diagnose the build error and suggest fixes."
            ))
            text_feedback = f"BUILD FAILED. Compiler errors:\n{build_errors[:1500]}"
            if llm_feedback:
                text_feedback += f"\n\nLLM Analysis:\n{llm_feedback}"
            write_json(results_path / "metrics.json", {
                "combined_score": 0.0,
                "public": {"build_returncode": result.returncode},
                "private": {"error": error_msg},
                "text_feedback": text_feedback,
            })
            write_json(results_path / "correct.json", {"correct": False, "error": error_msg})
            return

        # Run multiple times for stability
        run_log = results_path / "run.log"
        env = get_run_env()
        all_times: List[float] = []
        best_stdout = ""
        full_run_log = ""

        for run_idx in range(NUM_RUNS):
            start = time.perf_counter()
            returncode, stdout, stderr = _run_binary(work_dir, binary_path, env)
            run_duration = time.perf_counter() - start
            full_run_log += f"=== Run {run_idx + 1} ===\n{stdout}\n{stderr}\n\n"

            if returncode != 0:
                run_output = (stderr or stdout)[:2000]
                error_msg = f"Run failed (exit {returncode})\n{run_output}"
                llm_feedback = _call_llm(feedback_prompt, (
                    f"## Evolved Program\n\n```cuda\n{code_text[:8000]}\n```\n\n"
                    f"## Outcome\n\nRUN FAILED (exit {returncode}).\n{run_output}\n\n"
                    "Diagnose and suggest fixes."
                ))
                text_feedback = f"RUNTIME ERROR (exit {returncode}).\n{run_output[:1500]}"
                if llm_feedback:
                    text_feedback += f"\n\nLLM Analysis:\n{llm_feedback}"
                run_log.write_text(full_run_log, encoding="utf-8")
                write_json(results_path / "metrics.json", {
                    "combined_score": 0.0,
                    "public": {"run_returncode": returncode},
                    "private": {"error": error_msg},
                    "text_feedback": text_feedback,
                })
                write_json(results_path / "correct.json", {"correct": False, "error": error_msg})
                return

            if VERIFICATION_PASS_STR not in stdout:
                error_msg = f"Verification failed: '{VERIFICATION_PASS_STR}' not found."
                run_excerpt = stdout[-2000:] if stdout else "(no output)"
                llm_feedback = _call_llm(feedback_prompt, (
                    f"## Evolved Program\n\n```cuda\n{code_text[:8000]}\n```\n\n"
                    f"## Outcome\n\nVERIFICATION FAILED.\nOutput:\n{run_excerpt}\n\n"
                    "Diagnose why verification failed."
                ))
                text_feedback = f"VERIFICATION FAILED.\nOutput:\n{run_excerpt[:1500]}"
                if llm_feedback:
                    text_feedback += f"\n\nLLM Analysis:\n{llm_feedback}"
                run_log.write_text(full_run_log, encoding="utf-8")
                write_json(results_path / "metrics.json", {
                    "combined_score": 0.0,
                    "public": {"run_log_excerpt": stdout[-2000:]},
                    "private": {"error": error_msg},
                    "text_feedback": text_feedback,
                })
                write_json(results_path / "correct.json", {"correct": False, "error": error_msg})
                return

            t = parse_time_ms(stdout)
            if t is not None:
                all_times.append(t)
                if not best_stdout or t < min(all_times[:-1], default=float("inf")):
                    best_stdout = stdout

        run_log.write_text(full_run_log, encoding="utf-8")

        if not all_times:
            error_msg = "Could not parse 'Time: X.XXXX ms' from any run."
            write_json(results_path / "metrics.json", {
                "combined_score": 0.0,
                "public": {},
                "private": {"error": error_msg},
                "text_feedback": "Program passed but did not print 'Time: X.XXXX ms'.",
            })
            write_json(results_path / "correct.json", {"correct": False, "error": error_msg})
            return

        time_ms = min(all_times)
        combined_score = score_from_time_ms(time_ms)
        correct = True

        # Parse per-rank breakdown from best run
        rank_times = parse_all_rank_times(best_stdout)
        max_rank_time = max((t for t, _ in rank_times.values()), default=time_ms)

        # Compose text feedback
        best_so_far = _get_best_score_so_far(results_path)

        run_stability = ""
        if len(all_times) > 1:
            spread = max(all_times) - min(all_times)
            spread_pct = (spread / min(all_times)) * 100
            run_stability = (
                f"Run times: {', '.join(f'{t:.4f}' for t in all_times)} ms "
                f"(spread: {spread:.4f} ms, {spread_pct:.1f}%). "
                f"Using best: {time_ms:.4f} ms."
            )

        rank_info = ""
        if rank_times:
            parts = []
            for r in sorted(rank_times.keys()):
                t, tok = rank_times[r]
                parts.append(f"Rank{r}={t:.4f} ms ({tok} tokens)")
            rank_info = f"\nPer-rank: {', '.join(parts)}. Max={max_rank_time:.4f} ms."

        text_feedback = (
            f"Kernel time (token-weighted): {time_ms:.4f} ms. Lower is better. "
            f"Combined score: {combined_score:.2f}.{rank_info}"
        )
        if run_stability:
            text_feedback += f"\n{run_stability}"

        if best_so_far is not None:
            if combined_score > best_so_far:
                text_feedback += f"\nNEW BEST! Previous best score: {best_so_far:.2f}."
            elif combined_score < best_so_far:
                pct = ((best_so_far / combined_score) - 1) * 100
                text_feedback += (
                    f"\nREGRESSION. Best so far: {best_so_far:.2f} "
                    f"(yours: {combined_score:.2f}, {pct:.1f}% worse). "
                    f"Try smaller incremental changes."
                )

        # Parse per-phase timings from run output (present if instrumentation succeeded)
        phase_info = _parse_phase_lines(best_stdout) if instrumented_code else ""

        # LLM optimization suggestions
        outcome = (
            f"SUCCESS. Time: {time_ms:.4f} ms. Score: {combined_score:.2f}.\n"
            f"{run_stability}\n"
            f"Program stdout:\n{best_stdout[-2000:]}"
        )
        phase_section = ""
        if phase_info:
            phase_section = f"## Per-Phase Timing Breakdown\n\n{phase_info}\n\n"
            text_feedback += f"\n\nPer-phase timings:\n{phase_info}"

        llm_feedback = _call_llm(feedback_prompt, (
            f"## Evolved Program\n\n```cuda\n{code_text[:8000]}\n```\n\n"
            f"## Outcome\n\n{outcome[:3000]}\n\n"
            f"{phase_section}"
            "Identify the biggest bottleneck and suggest 1-3 improvements."
        ))
        if llm_feedback:
            text_feedback += f"\n\nLLM Suggestions:\n{llm_feedback}"

        per_rank_metrics = {}
        for r in sorted(rank_times.keys()):
            t, tok = rank_times[r]
            per_rank_metrics[f"rank{r}_time_ms"] = t
            per_rank_metrics[f"rank{r}_tokens"] = tok

        metrics = {
            "combined_score": combined_score,
            "public": {
                "time_ms": time_ms,
                "max_time_ms": max_rank_time,
                **per_rank_metrics,
                "build_duration_sec": build_duration,
                "all_run_times_ms": all_times,
            },
            "private": {
                "stdout_excerpt": best_stdout[-2000:] if len(best_stdout) > 2000 else best_stdout,
            },
            "text_feedback": text_feedback,
        }

    except subprocess.TimeoutExpired:
        error_msg = f"Run timed out ({RUN_TIMEOUT}s)."
        text_feedback = (
            f"TIMEOUT: Program did not finish within {RUN_TIMEOUT} seconds. "
            "Possible causes: (1) deadlock in communication signals "
            "(e.g., waitSignal waiting for unsent signal), "
            "(2) barrier deadlock (not all ranks reach barrier), "
            "(3) infinite kernel loop, "
            "(4) persistent polling kernel stuck on device flag. "
            "Check that every wait has a matching send/put and that all ranks "
            "reach every synchronization point."
        )
        try:
            feedback_prompt = _build_feedback_prompt(_detect_api_mode(code_text))
            llm_fb = _call_llm(feedback_prompt, (
                f"## Program\n\n```cuda\n{code_text[:8000]}\n```\n\n"
                f"TIMEOUT: {RUN_TIMEOUT} seconds. Diagnose deadlock."
            ))
            if llm_fb:
                text_feedback += f"\n\nLLM Analysis:\n{llm_fb}"
        except Exception:
            pass
        metrics = {
            "combined_score": 0.0, "public": {},
            "private": {"error": error_msg},
            "text_feedback": text_feedback,
        }
    except Exception as exc:
        error_msg = f"{exc}\n{traceback.format_exc()}"
        metrics = {
            "combined_score": 0.0, "public": {},
            "private": {"error": error_msg},
            "text_feedback": f"Unexpected evaluation error: {exc}",
        }

    write_json(results_path / "metrics.json", metrics)
    write_json(results_path / "correct.json", {"correct": correct, "error": error_msg or None})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate CUDA kernel program."
    )
    parser.add_argument("--program_path", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
