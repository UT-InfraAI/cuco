#!/usr/bin/env python3
"""GEMM + LSA Allgather workload (Python-native NCCL device-API).

Each rank computes a local GEMM (A @ B_local) then performs an LSA allgather
to share the result with all other ranks.  After the allgather every rank holds
the concatenated outputs of all ranks.

Workload (column-parallel linear):
    A         : [M, K]         — same input on every rank
    B_local   : [K, N_LOCAL]   — each rank owns a different weight slice
    local_out : [M, N_LOCAL]   — GEMM output written into send_buf
    full_out  : [WORLD_SIZE, M, N_LOCAL] — gathered from all ranks via LSA

Dimensions (default):
    M=2048, K=4096, N_LOCAL=2048, WORLD_SIZE=2

Frozen (Python infrastructure):
    NCCL communicator init, window registration, DevComm setup,
    timing loop, verification, output printing.

Evolved (CUDA kernel source string + launch constants):
    _KERNEL_SRC      — CUDA kernels: GEMM + LSA allgather
    _GEMM_BLOCKS     — number of thread-blocks for GEMM launch
    _GEMM_THREADS    — threads per block for GEMM
    _LSA_CTA_COUNT   — CTAs for LSA allgather (must match lsa_barrier_count)
    _LSA_THREADS     — threads per CTA for LSA allgather
"""

from __future__ import annotations

import sys
import os
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Allow importing from the cuco package regardless of install location
_CUCO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_CUCO_ROOT) not in sys.path:
    sys.path.insert(0, str(_CUCO_ROOT))

from cuco.compile import CompileOptions, compile_cuda
from cuco.nccl import (
    DEVCOMM_NBYTES,
    DevCommRequirements,
    comm_query_properties,
    dev_comm_create,
    dev_comm_destroy,
)

# ---------------------------------------------------------------------------
# Workload dimensions
# ---------------------------------------------------------------------------
WORLD_SIZE = 2
M = 2048
K = 4096
N_LOCAL = 2048          # each rank's weight columns; N_total = N_LOCAL * WORLD_SIZE

NUM_WARMUP = 3
NUM_RUNS   = 5
INIT_METHOD = "tcp://127.0.0.1:29502"

# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-START
# ---------------------------------------------------------------------------

_KERNEL_SRC = r"""
#include <nccl_device.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// -------------------------------------------------------------------------
// Kernel 1: GEMM (naive row-parallel)
//   Each thread computes one output element: C[row, col] = sum_k A[row,k]*B[k,col]
//   Output is written directly into send_buf (the symmetric send window's backing
//   memory) so no extra copy is needed before the allgather.
// -------------------------------------------------------------------------
extern "C" __global__ void gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total  = M * N;

    for (int i = idx; i < total; i += stride) {
        int row = i / N;
        int col = i % N;
        float s = 0.f;
        for (int k = 0; k < K; ++k)
            s += A[row * K + k] * B[k * N + col];
        C[i] = s;
    }
}

// -------------------------------------------------------------------------
// Kernel 2: LSA Allgather
//   After a distributed barrier (all ranks' GEMMs have written to their send
//   windows), each rank reads every peer's send window directly via NVLink
//   and stores the result at the appropriate slot of its local recv_buf.
//   Layout: recv_buf[peer * count .. (peer+1)*count - 1] = peer's GEMM output
// -------------------------------------------------------------------------
extern "C" __global__ void lsa_allgather_kernel(
    ncclWindow_t       sendwin,
    float* __restrict__ recv_buf,
    size_t              count,
    const ncclDevComm* devCommPtr)
{
    const ncclDevComm devComm = *devCommPtr;
    auto coop = cg::this_thread_block();

    // One barrier slot per CTA (blockIdx.x in 0..lsaBarrierCount-1)
    ncclLsaBarrierSession<cg::thread_block> bar(
        coop, devComm, ncclTeamTagLsa{}, blockIdx.x);

    // Acquire fence: wait until all peers have written their send windows
    bar.sync(coop, cuda::memory_order_relaxed);

    int nRanks     = devComm.nRanks;
    int globalTid  = threadIdx.x + blockDim.x * blockIdx.x;
    int totalElems = (int)(count * (size_t)nRanks);
    int stride     = blockDim.x * gridDim.x;

    for (int i = globalTid; i < totalElems; i += stride) {
        int peer = i / (int)count;
        int elem = i % (int)count;
        float* peer_send = (float*)ncclGetLsaPointer(sendwin, 0, peer);
        recv_buf[peer * (int)count + elem] = peer_send[elem];
    }

    // Release fence: signal that recv_buf is fully written
    bar.sync(coop, cuda::memory_order_release);
}
"""

_GEMM_BLOCKS  = 256    # thread-blocks for GEMM kernel
_GEMM_THREADS = 256    # threads per block for GEMM kernel

_LSA_CTA_COUNT = 4     # CTAs for LSA allgather (must equal lsa_barrier_count)
_LSA_THREADS   = 256   # threads per CTA for LSA allgather

# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-END
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-rank worker
# ---------------------------------------------------------------------------

def _worker(rank, world_size, init_method, cubin_bytes, A_shared, B_all, results, timing):
    """Intra-node worker: runs GEMM + LSA allgather, stores timing and results."""
    import nccl.core as nccl
    from nccl.core.constants import WindowFlag
    from nccl.core.interop.torch import empty as nccl_empty
    from cuda.bindings import runtime as cudart
    from cuda.core import Device, LaunchConfig, ObjectCode, launch
    from cuda.core._utils.cuda_utils import handle_return
    import time as _time

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            init_method=init_method)
    try:
        with ExitStack() as cleanup:
            dev = Device(rank)
            dev.set_current()

            # ---- NCCL communicator ----
            uid_bytes = [nccl.get_unique_id().as_bytes if rank == 0 else None]
            dist.broadcast_object_list(uid_bytes, src=0)
            comm = nccl.Communicator.init(
                world_size, rank, nccl.UniqueId.from_bytes(uid_bytes[0]))
            cleanup.callback(comm.destroy)

            props = comm_query_properties(comm._comm)
            if not props.device_api_support or props.n_lsa_teams == 0:
                raise RuntimeError(
                    f"LSA not supported on rank {rank}: "
                    f"device_api={props.device_api_support}, "
                    f"n_lsa_teams={props.n_lsa_teams}"
                )

            # ---- GPU tensors ----
            A_local = A_shared.cuda(rank)           # [M, K]
            B_local = B_all[rank].cuda(rank)         # [K, N_LOCAL]

            count = M * N_LOCAL
            # send_buf: symmetric memory (registered as window) — GEMM writes here
            send_buf = nccl_empty(count, dtype=torch.float32, device=rank)
            send_buf.zero_()
            # recv_buf: regular CUDA tensor — LSA allgather reads peers' send_bufs here
            recv_buf = torch.zeros(world_size * count, dtype=torch.float32,
                                   device=f"cuda:{rank}")

            send_win = comm.register_window(send_buf, WindowFlag.CollSymmetric)
            if send_win is None:
                raise RuntimeError(f"rank {rank}: ncclCommWindowRegister returned NULL")
            cleanup.callback(send_win.close)

            # ---- DevComm ----
            dcomm = dev_comm_create(
                comm._comm,
                DevCommRequirements(lsa_barrier_count=_LSA_CTA_COUNT),
            )
            cleanup.callback(dev_comm_destroy, comm._comm, dcomm)

            devcomm_buf = torch.zeros(DEVCOMM_NBYTES, dtype=torch.uint8,
                                      device=f"cuda:{rank}")
            handle_return(cudart.cudaMemcpy(
                devcomm_buf.data_ptr(), dcomm.address, DEVCOMM_NBYTES,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            ))

            # ---- Kernels ----
            stream = dev.create_stream()
            cleanup.callback(stream.close)

            code        = ObjectCode.from_cubin(cubin_bytes)
            gemm_k      = code.get_kernel("gemm_kernel")
            allgather_k = code.get_kernel("lsa_allgather_kernel")

            gemm_cfg      = LaunchConfig(grid=_GEMM_BLOCKS, block=_GEMM_THREADS)
            allgather_cfg = LaunchConfig(grid=_LSA_CTA_COUNT, block=_LSA_THREADS)

            def _run_once():
                send_buf.zero_()
                recv_buf.zero_()
                # GEMM: A_local @ B_local -> send_buf
                launch(stream, gemm_cfg, gemm_k,
                       A_local.data_ptr(), B_local.data_ptr(), send_buf.data_ptr(),
                       M, N_LOCAL, K)
                # LSA Allgather: send_buf -> recv_buf via NVLink peer access
                launch(stream, allgather_cfg, allgather_k,
                       send_win.handle, recv_buf.data_ptr(), count,
                       devcomm_buf.data_ptr())
                stream.sync()

            # ---- Warmup ----
            for _ in range(NUM_WARMUP):
                _run_once()
            dist.barrier()

            # ---- Timed runs ----
            t0 = _time.perf_counter()
            for _ in range(NUM_RUNS):
                _run_once()
            t1 = _time.perf_counter()
            timing[rank] = (t1 - t0) * 1000.0 / NUM_RUNS

            # ---- Save result for verification ----
            results[rank].copy_(recv_buf.view(world_size, M, N_LOCAL).cpu())

    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    A       = torch.randn(M, K).share_memory_()
    B_all   = [torch.randn(K, N_LOCAL).share_memory_() for _ in range(WORLD_SIZE)]

    # Compile kernel source
    cubin = compile_cuda(_KERNEL_SRC, CompileOptions(std="c++17"))

    results = torch.zeros(WORLD_SIZE, WORLD_SIZE, M, N_LOCAL).share_memory_()
    timing  = torch.zeros(WORLD_SIZE).share_memory_()

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, INIT_METHOD, cubin, A, B_all, results, timing),
        nprocs=WORLD_SIZE,
        join=True,
    )

    # ---- Timing ----
    time_ms = timing.max().item()
    print(f"Time: {time_ms:.4f} ms")

    # ---- Verification ----
    # Reference: A @ B_all[r] for each r, computed in FP32 on CPU
    ref = torch.stack([A @ B_all[r] for r in range(WORLD_SIZE)])  # [WS, M, N_LOCAL]

    passed = True
    for rank in range(WORLD_SIZE):
        if not results[rank].allclose(ref, atol=1e-2, rtol=1e-2):
            passed = False
            worst = (results[rank] - ref).abs().max().item()
            print(f"Verification: FAIL (rank {rank}, max_err={worst:.4f})")
            break

    if passed:
        print("Verification: PASS")


if __name__ == "__main__":
    main()
