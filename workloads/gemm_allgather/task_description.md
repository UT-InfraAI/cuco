You are an expert in CUDA kernel optimization and NCCL device-side communication.

You are evolving a **Python-driven GEMM + LSA Allgather workload**.

## Program Structure

**FROZEN** — Python infrastructure: NCCL communicator setup, window registration,
DevComm creation, mp.spawn worker, timing loop, verification.

**EVOLVE-BLOCK** — you control:
```python
# EVOLVE-BLOCK-START
_KERNEL_SRC = r"""
...CUDA kernel source code...
"""
_GEMM_BLOCKS  = 256    # GEMM grid blocks
_GEMM_THREADS = 256    # GEMM threads per block
_LSA_CTA_COUNT = 4     # LSA CTAs (MUST match lsa_barrier_count in DevCommRequirements)
_LSA_THREADS   = 256   # LSA threads per CTA
# EVOLVE-BLOCK-END
```

The frozen launcher uses these exactly:
```python
LaunchConfig(grid=_GEMM_BLOCKS, block=_GEMM_THREADS)       # gemm_kernel
LaunchConfig(grid=_LSA_CTA_COUNT, block=_LSA_THREADS)      # lsa_allgather_kernel
DevCommRequirements(lsa_barrier_count=_LSA_CTA_COUNT)
```

## Workload — Column-parallel linear + allgather

- M=2048 (tokens), K=4096 (hidden), N_LOCAL=2048 (per-rank cols), WORLD_SIZE=2
- Each rank: A[M,K] @ B_local[K,N_LOCAL] → send_buf, then LSA allgather
- After allgather: recv_buf[r*M*N_LOCAL:(r+1)*M*N_LOCAL] == A @ B_all[r]

## Optimization Goal

Minimize **Time: X.XXXX ms**. Lower is better.

## Hard Constraints

- Do NOT touch code outside the EVOLVE-BLOCK.
- No external libraries (cuBLAS, cuDNN, Thrust, CUB, etc.) — only cuda + nccl.
- `_LSA_CTA_COUNT` MUST equal `lsa_barrier_count` (frozen) — mismatch → LSA deadlock.
- Keep extern "C" kernel names `gemm_kernel` and `lsa_allgather_kernel`.
- Program must print "Verification: PASS".

## Kernel Optimization Strategies

### GEMM
- Tiled shared-memory SGEMM (TILE=32): reduce HBM traffic, exploit L1/smem.
- Vectorized loads: float4 for A rows and B columns (requires aligned strides).
- Register blocking: each thread computes a 4×4 output sub-tile.
- Launch is 1D; compute (row,col) from linear thread index.

### LSA Allgather
- Vectorized peer reads: cast `ncclGetLsaPointer` result to `float4*` (4× bandwidth).
- CTA count: _LSA_CTA_COUNT controls parallelism; must stay == lsa_barrier_count.
- Overlap potential: if ranks' GEMMs complete at different times, the LSA barrier
  stalls the faster rank. Fusing GEMM tiles with partial LSA reads eliminates this.

### LSA API Correctness
- bar.sync(coop, memory_order_relaxed) — acquire fence before reads.
- bar.sync(coop, memory_order_release) — release fence after writes.
- ncclGetLsaPointer(sendwin, byte_offset, peer) → pointer into peer's send window.
- blockIdx.x must be < _LSA_CTA_COUNT (each CTA owns one barrier slot).
