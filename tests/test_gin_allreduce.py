"""Integration test for GIN-only allreduce across 2 nodes (internode).

GIN (GPU-Initiated Networking) uses one-sided put operations over the network.
Unlike LSA which works intranode, GIN is designed for internode communication.

Usage (2-node, 1 GPU per node):
  # On mew2 (rank 0):
  MASTER_ADDR=mew2 MASTER_PORT=29600 WORLD_SIZE=2 RANK=0 \
    python -m pytest tests/test_gin_allreduce.py -s

  # On mew3 (rank 1):
  MASTER_ADDR=mew2 MASTER_PORT=29600 WORLD_SIZE=2 RANK=1 \
    python -m pytest tests/test_gin_allreduce.py -s
"""

from __future__ import annotations

import os
from contextlib import ExitStack

import pytest

from cuco.compile import CompileOptions, compile_cuda

WORLD_SIZE = 2
GIN_SKIP_REASON = "GIN not available (deviceApiSupport=False or ginType==0)"

# Kernel launch config — CTA count must equal railGinBarrierCount
_CTA_COUNT = 1
_THREADS_PER_CTA = 512
_COUNT = 1 << 20  # float32 elements per rank (~4 MB)

# GIN allreduce kernel:
#   Each rank puts its send buffer to every peer's recv buffer (at rank-specific
#   offset), waits for all remote puts via signals, then locally reduces the
#   received chunks into the output.
_GIN_ALLREDUCE_SRC = r"""
#include <nccl_device.h>

extern "C" __global__ void gin_allreduce_kernel(
    ncclWindow_t sendwin,
    ncclWindow_t recvwin,
    float*       recvbuf,
    float*       output,
    size_t       count,
    const ncclDevComm* devCommPtr)
{
    const ncclDevComm devComm = *devCommPtr;
    const int rank   = devComm.rank;
    const int nRanks = devComm.nRanks;
    const size_t bytes = count * sizeof(float);

    int ginContext = 0;
    unsigned int signalIndex = 0;
    ncclGin gin { devComm, ginContext };

    // GIN barrier: synchronize before starting puts
    ncclGinBarrierSession<ncclCoopCta> bar {
        ncclCoopCta(),
        gin,
        ncclTeamWorld(devComm),
        devComm.railGinBarrier,
        blockIdx.x
    };
    bar.sync(ncclCoopCta(), cuda::memory_order_relaxed, ncclGinFenceLevel::Relaxed);

    // Read initial signal value (rolling)
    uint64_t signalValue = gin.readSignal(signalIndex);

    // Each rank puts its send buffer to every peer's recv buffer at offset rank*bytes
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int nthreads = blockDim.x * gridDim.x;
    for (int r = tid; r < nRanks; r += nthreads) {
        gin.put(ncclTeamWorld(devComm), r,
                recvwin, rank * bytes,    // dst: peer's recv buf at our rank's slot
                sendwin, 0,               // src: our entire send buf
                bytes,
                ncclGin_SignalInc{signalIndex});
    }

    // Wait for all ranks to complete their puts to us
    gin.waitSignal(ncclCoopCta(), signalIndex, signalValue + nRanks);
    gin.flush(ncclCoopCta());

    // Local reduction: sum all received chunks into output
    int globalTid    = threadIdx.x + blockIdx.x * blockDim.x;
    int globalStride = blockDim.x * gridDim.x;
    for (size_t i = globalTid; i < count; i += globalStride) {
        float v = 0.f;
        for (int peer = 0; peer < nRanks; peer++) {
            float* src = (float*)((char*)recvbuf + peer * bytes);
            v += src[i];
        }
        output[i] = v;
    }

    // Final barrier: ensure all ranks are done before returning
    bar.sync(ncclCoopCta(), cuda::memory_order_release, ncclGinFenceLevel::Relaxed);
}
"""


@pytest.fixture(scope="session")
def gin_cubin():
    try:
        return compile_cuda(_GIN_ALLREDUCE_SRC, CompileOptions(std="c++17"))
    except RuntimeError as exc:
        pytest.skip(f"nvcc failed: {exc}")


def _require_gin_dependencies():
    pytest.importorskip("nccl.core", reason="nccl4py not installed")
    pytest.importorskip(
        "cuco.nccl",
        reason="cuco.nccl not built (run: pip install -e '.[nccl]')",
    )


def _get_env_config():
    """Read internode config from environment variables."""
    master_addr = os.environ.get("MASTER_ADDR")
    master_port = os.environ.get("MASTER_PORT", "29600")
    world_size = int(os.environ.get("WORLD_SIZE", "0"))
    rank = int(os.environ.get("RANK", "-1"))
    if not master_addr or world_size < 2 or rank < 0:
        pytest.skip(
            "Internode test requires MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK "
            "env vars (e.g. MASTER_ADDR=mew2 WORLD_SIZE=2 RANK=0)"
        )
    init_method = f"tcp://{master_addr}:{master_port}"
    return init_method, world_size, rank


def _init_process_group(rank, world_size, init_method):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)  # 1 GPU per node
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        init_method=init_method,
    )
    return dist


def _cleanup_process_group(dist):
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def test_gin_allreduce(gin_cubin):
    """Run GIN allreduce over 2 internode ranks; compare against dist.all_reduce."""
    import nccl.core as nccl
    from nccl.core.constants import WindowFlag
    from nccl.core.interop.torch import empty as nccl_empty
    import torch
    from cuda.bindings import runtime as cudart
    from cuda.core import Device, LaunchConfig, ObjectCode, launch
    from cuda.core._utils.cuda_utils import handle_return

    from cuco.nccl import (
        DEVCOMM_NBYTES,
        DevCommRequirements,
        comm_query_properties,
        dev_comm_create,
        dev_comm_destroy,
    )

    _require_gin_dependencies()
    init_method, world_size, rank = _get_env_config()

    dist = _init_process_group(rank, world_size, init_method)
    try:
        with ExitStack() as cleanup:
            dev = Device(0)  # 1 GPU per node
            dev.set_current()

            # Create NCCL communicator
            uid_bytes = [nccl.get_unique_id().as_bytes if rank == 0 else None]
            dist.broadcast_object_list(uid_bytes, src=0)
            comm = nccl.Communicator.init(
                world_size,
                rank,
                nccl.UniqueId.from_bytes(uid_bytes[0]),
            )
            cleanup.callback(comm.destroy)

            # Check GIN support
            props = comm_query_properties(comm._comm)
            print(f"[rank {rank}] CommProperties: device_api={props.device_api_support}, "
                  f"gin_type={props.gin_type}, n_lsa_teams={props.n_lsa_teams}")
            if not props.device_api_support or props.gin_type == 0:
                pytest.skip(GIN_SKIP_REASON)

            # Allocate symmetric buffers
            # recv buffer needs nRanks * count to hold each peer's contribution
            send_buf = nccl_empty(_COUNT, dtype=torch.float32, device=0)
            recv_buf = nccl_empty(_COUNT * world_size, dtype=torch.float32, device=0)
            output_buf = torch.zeros(_COUNT, dtype=torch.float32, device="cuda:0")
            send_buf.fill_(float(rank))
            recv_buf.zero_()

            # Register windows
            send_win = comm.register_window(send_buf, WindowFlag.CollSymmetric)
            recv_win = comm.register_window(recv_buf, WindowFlag.CollSymmetric)
            if send_win is None or recv_win is None:
                pytest.skip("ncclCommWindowRegister returned NULL (unsupported platform)")
            cleanup.callback(recv_win.close)
            cleanup.callback(send_win.close)

            # Create device comm with GIN resources
            dcomm = dev_comm_create(
                comm._comm,
                DevCommRequirements(
                    rail_gin_barrier_count=_CTA_COUNT,
                    gin_signal_count=1,
                    gin_context_count=1,
                    gin_force_enable=True,
                ),
            )
            cleanup.callback(dev_comm_destroy, comm._comm, dcomm)

            # Clear any stale CUDA error from ncclDevCommCreate
            handle_return(cudart.cudaGetLastError())

            # Copy devComm to device
            devcomm_buf = torch.zeros(
                DEVCOMM_NBYTES,
                dtype=torch.uint8,
                device="cuda:0",
            )
            handle_return(
                cudart.cudaMemcpy(
                    devcomm_buf.data_ptr(),
                    dcomm.address,
                    DEVCOMM_NBYTES,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )

            # Launch kernel
            stream = dev.create_stream()
            cleanup.callback(stream.close)

            kernel = ObjectCode.from_cubin(gin_cubin).get_kernel("gin_allreduce_kernel")
            launch(
                stream,
                LaunchConfig(grid=_CTA_COUNT, block=_THREADS_PER_CTA),
                kernel,
                send_win.handle,
                recv_win.handle,
                recv_buf.data_ptr(),
                output_buf.data_ptr(),
                _COUNT,
                devcomm_buf.data_ptr(),
            )
            stream.sync()
            gin_result = output_buf.cpu()

            # --- dist.all_reduce baseline ---
            baseline = torch.full((_COUNT,), float(rank), device="cuda:0")
            dist.all_reduce(baseline)
            baseline_cpu = baseline.cpu()

            # Compare
            print(f"[rank {rank}] GIN result[:4]   = {gin_result[:4].tolist()}")
            print(f"[rank {rank}] baseline[:4]     = {baseline_cpu[:4].tolist()}")
            assert gin_result.allclose(baseline_cpu), (
                f"rank {rank}: GIN result differs from dist.all_reduce baseline; "
                f"gin[:4]={gin_result[:4].tolist()}, "
                f"dist[:4]={baseline_cpu[:4].tolist()}"
            )
            print(f"[rank {rank}] PASSED")

    finally:
        _cleanup_process_group(dist)
