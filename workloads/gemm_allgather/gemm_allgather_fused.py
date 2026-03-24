#!/usr/bin/env python3
"""GEMM + LSA Allgather – fused single-kernel seed.

A single kernel receives all inputs (A, B, send_buf, sendwin, recv_buf, …).
bar.sync(blockIdx.x) is a per-slot barrier — it syncs CTA i across ranks,
not all CTAs globally.  The kernel exploits this: CTA t computes tiles
{t, t+G, …}, then bar.sync(t) waits for those same tiles on every rank.
CTA t then reads from every peer's send_buf at only those tile positions,
which are guaranteed written.  Every (row, col) is owned by exactly one CTA,
so recv_buf is filled completely.  No cooperative launch needed.

Frozen: NCCL comm / window / DevComm setup, timing loop, verification.
        _run_once calls _launch() from the EVOLVE-BLOCK.
Evolved: _KERNEL_SRC, _LSA_CTA_COUNT, _TOTAL_BLOCKS, _BLOCK_THREADS, _launch().
"""

from __future__ import annotations

import sys
import os
from contextlib import ExitStack
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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

WORLD_SIZE = 2
M = 2048
K = 4096
N_LOCAL = 2048
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

#define TILE 32

// Fused GEMM + LSA Allgather — single kernel, no cooperative launch.
//
// bar.sync(blockIdx.x) is a *per-slot* distributed barrier: it syncs CTA i
// on rank 0 with CTA i on all other ranks — not a global barrier.
// We exploit this as follows:
//
//   Phase 1 — CTA t computes tiles {t, t+G, t+2G, ...} (G = gridDim.x)
//             and writes them to send_buf.
//   Phase 2 — bar.sync(t, acquire): waits for CTA t on every rank to
//             finish Phase 1 for those same tile slots.  After the sync,
//             every peer's send_buf is written at exactly this CTA's tiles.
//   Phase 3 — CTA t reads from every peer's send_buf, but ONLY at the tile
//             positions it owns.  Those are the positions the barrier
//             just guaranteed are written.
//   Phase 4 — bar.sync(t, release).
//
// Every (row, col) is owned by exactly one CTA, so recv_buf is filled
// completely with no races.
extern "C" __global__ void fused_gemm_allgather(
    const float* __restrict__ A,          // [M, K]
    const float* __restrict__ B,          // [K, N]
    float*       __restrict__ send_buf,   // [M, N]  — GEMM output / send window
    ncclWindow_t               sendwin,
    float*       __restrict__ recv_buf,   // [nRanks * M * N]
    size_t                     count,     // M * N
    const ncclDevComm*         devCommPtr,
    int M, int N, int K)
{
    const ncclDevComm devComm = *devCommPtr;
    auto block = cg::this_thread_block();

    // --- 1. Tiled SGEMM → send_buf ----------------------------------------
    // 1-D block of TILE*TILE threads treated as a 2-D tile.
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int tx = threadIdx.x % TILE;
    const int ty = threadIdx.x / TILE;
    const int tiles_N     = (N + TILE - 1) / TILE;
    const int total_tiles = ((M + TILE - 1) / TILE) * tiles_N;

    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int tr  = t / tiles_N;
        const int tc  = t % tiles_N;
        const int row = tr * TILE + ty;
        const int col = tc * TILE + tx;

        float acc = 0.f;
        for (int k = 0; k < (K + TILE - 1) / TILE; ++k) {
            As[ty][tx] = (row < M && k * TILE + tx < K)
                         ? A[row * K + k * TILE + tx] : 0.f;
            Bs[ty][tx] = (col < N && k * TILE + ty < K)
                         ? B[(k * TILE + ty) * N + col] : 0.f;
            __syncthreads();
            for (int i = 0; i < TILE; ++i)
                acc += As[ty][i] * Bs[i][tx];
            __syncthreads();
        }
        if (row < M && col < N)
            send_buf[row * N + col] = acc;
    }

    // --- 2. LSA acquire barrier -------------------------------------------
    ncclLsaBarrierSession<cg::thread_block> bar(
        block, devComm, ncclTeamTagLsa{}, blockIdx.x);
    bar.sync(block, cuda::memory_order_relaxed);

    // --- Phase 3: gather this CTA's tile slots from all peers -------------
    // Safe because bar.sync(blockIdx.x) guaranteed that every peer's CTA
    // blockIdx.x (which owns the same tile slots) has finished writing them.
    const int nRanks = devComm.nRanks;
    for (int t = blockIdx.x; t < total_tiles; t += gridDim.x) {
        const int tr  = t / tiles_N;
        const int tc  = t % tiles_N;
        const int row = tr * TILE + ty;
        const int col = tc * TILE + tx;

        if (row < M && col < N) {
            const int idx = row * N + col;
            for (int peer = 0; peer < nRanks; ++peer) {
                const float* peer_send = (const float*)ncclGetLsaPointer(sendwin, 0, peer);
                recv_buf[(size_t)peer * count + idx] = peer_send[idx];
            }
        }
    }

    // --- Phase 4: LSA release barrier -------------------------------------
    bar.sync(block, cuda::memory_order_release);
}
"""

# _LSA_CTA_COUNT: number of CTAs — both GEMM tile workers and allgather CTAs.
# Must equal lsa_barrier_count in DevCommRequirements (frozen below).
_LSA_CTA_COUNT = 64
_BLOCK_THREADS = 1024  # TILE * TILE = 32 * 32


def _launch(stream, code, A_ptr, B_ptr, send_buf_ptr,
            send_win_handle, recv_buf_ptr, count, devcomm_ptr,
            M, N, K):
    from cuda.core import LaunchConfig, launch
    cfg = LaunchConfig(grid=_LSA_CTA_COUNT, block=_BLOCK_THREADS)
    k = code.get_kernel("fused_gemm_allgather")
    launch(stream, cfg, k,
           A_ptr, B_ptr, send_buf_ptr,
           send_win_handle, recv_buf_ptr, count,
           devcomm_ptr, M, N, K)

# ---------------------------------------------------------------------------
# EVOLVE-BLOCK-END
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-rank worker  (frozen)
# ---------------------------------------------------------------------------

def _worker(rank, world_size, init_method, cubin_bytes, A_shared, B_all, results, timing):
    import nccl.core as nccl
    from nccl.core.constants import WindowFlag
    from nccl.core.interop.torch import empty as nccl_empty
    from cuda.bindings import runtime as cudart
    from cuda.core import Device, ObjectCode
    from cuda.core._utils.cuda_utils import handle_return
    import time as _time

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size,
                            init_method=init_method)
    try:
        with ExitStack() as cleanup:
            dev = Device(rank)
            dev.set_current()

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

            A_local = A_shared.cuda(rank)
            B_local = B_all[rank].cuda(rank)

            count = M * N_LOCAL
            send_buf = nccl_empty(count, dtype=torch.float32, device=rank)
            send_buf.zero_()
            recv_buf = torch.zeros(world_size * count, dtype=torch.float32,
                                   device=f"cuda:{rank}")

            send_win = comm.register_window(send_buf, WindowFlag.CollSymmetric)
            if send_win is None:
                raise RuntimeError(f"rank {rank}: ncclCommWindowRegister returned NULL")
            cleanup.callback(send_win.close)

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

            stream = dev.create_stream()
            cleanup.callback(stream.close)

            code = ObjectCode.from_cubin(cubin_bytes)

            def _run_once():
                send_buf.zero_()
                recv_buf.zero_()
                _launch(stream, code,
                        A_local.data_ptr(), B_local.data_ptr(), send_buf.data_ptr(),
                        send_win.handle, recv_buf.data_ptr(), count,
                        devcomm_buf.data_ptr(), M, N_LOCAL, K)
                stream.sync()

            for _ in range(NUM_WARMUP):
                _run_once()
            dist.barrier()

            t0 = _time.perf_counter()
            for _ in range(NUM_RUNS):
                _run_once()
            t1 = _time.perf_counter()
            timing[rank] = (t1 - t0) * 1000.0 / NUM_RUNS

            results[rank].copy_(recv_buf.view(world_size, M, N_LOCAL).cpu())

    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Main entry point  (frozen)
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)

    A     = torch.randn(M, K).share_memory_()
    B_all = [torch.randn(K, N_LOCAL).share_memory_() for _ in range(WORLD_SIZE)]

    cubin = compile_cuda(_KERNEL_SRC, CompileOptions(std="c++17"))

    results = torch.zeros(WORLD_SIZE, WORLD_SIZE, M, N_LOCAL).share_memory_()
    timing  = torch.zeros(WORLD_SIZE).share_memory_()

    mp.spawn(
        _worker,
        args=(WORLD_SIZE, INIT_METHOD, cubin, A, B_all, results, timing),
        nprocs=WORLD_SIZE,
        join=True,
    )

    time_ms = timing.max().item()
    print(f"Time: {time_ms:.4f} ms")

    ref = torch.stack([A @ B_all[r] for r in range(WORLD_SIZE)])

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
