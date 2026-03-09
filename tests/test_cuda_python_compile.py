"""
Tests for NCCL device init kernel and torch.distributed NCCL on 2 local GPUs.

Both tests spawn 2 workers via torch.multiprocessing (spawn context):

  test_gin_devcomm_rank_info — ncclCommInitRank → ncclDevCommCreate → nvcc-compiled
      gin_info kernel → verify rank=i, nRanks=2 from ncclTeamWorld on each GPU.

  test_torch_dist_allreduce — torch.distributed (NCCL backend) allreduce:
      each rank contributes its rank+1, result must equal 1+2=3 on both ranks.

NOTE: nvcc is used (not NVRTC) because NCCL device headers use #include_next,
which NVRTC does not support.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import warnings

import pytest

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*_pack_.*")

from cuda.bindings import runtime as cudart  # noqa: E402
from cuda.core import Device, ObjectCode  # noqa: E402

from cuco.site_config import GPU_ARCH, NCCL_INCLUDE, NCCL_STATIC_LIB, NVCC  # noqa: E402

_NCCL_SO = os.path.join(os.path.dirname(NCCL_STATIC_LIB), "libnccl.so")

# ---------------------------------------------------------------------------
# ctypes types
# ---------------------------------------------------------------------------

ncclComm_t = ctypes.c_void_p

# ncclUniqueId: 128 opaque bytes.  Must be a Structure (not c_char*128) so
# ctypes passes it BY VALUE to ncclCommInitRank — arrays become pointers.
class _NCCLUniqueId(ctypes.Structure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = [("internal", ctypes.c_char * 128)]

# ncclDevCommRequirements_t: 56 bytes, layout verified with nvcc sizeof.
class _NCCLDevCommRequirements(ctypes.Structure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = [
        ("resourceRequirementsList", ctypes.c_void_p),
        ("teamRequirementsList",     ctypes.c_void_p),
        ("lsaMultimem",              ctypes.c_bool),
        ("_pad1",                    ctypes.c_uint8 * 3),
        ("barrierCount",             ctypes.c_int),
        ("lsaBarrierCount",          ctypes.c_int),
        ("railGinBarrierCount",      ctypes.c_int),
        ("lsaLLA2ABlockCount",       ctypes.c_int),
        ("lsaLLA2ASlotCount",        ctypes.c_int),
        ("ginForceEnable",           ctypes.c_bool),
        ("_pad2",                    ctypes.c_uint8 * 3),
        ("ginContextCount",          ctypes.c_int),
        ("ginSignalCount",           ctypes.c_int),
        ("ginCounterCount",          ctypes.c_int),
    ]

_NCCL_DEVCOMM_SIZE = 200  # ncclDevComm is opaque on host, 200 bytes
class _NCCLDevComm(ctypes.Structure):
    _pack_ = 1
    _layout_ = "ms"
    _fields_ = [("_opaque", ctypes.c_uint8 * _NCCL_DEVCOMM_SIZE)]

# ---------------------------------------------------------------------------
# GIN kernel — reads rank/nRanks via ncclTeamWorld
# ---------------------------------------------------------------------------

_GIN_KERNEL_SRC = r"""
#include <nccl_device.h>
extern "C" __global__ void gin_info(const ncclDevComm* devComm, int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ncclTeam world = ncclTeamWorld(*devComm);
        out[0] = world.rank;
        out[1] = world.nRanks;
    }
}
"""

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def gin_cubin(tmp_path_factory):
    work = tmp_path_factory.mktemp("gin")
    src = work / "gin_info.cu"
    out = work / "gin_info.cubin"
    src.write_text(_GIN_KERNEL_SRC)
    r = subprocess.run(
        [NVCC, "-I", NCCL_INCLUDE, f"-arch={GPU_ARCH}", "-cubin", str(src), "-o", str(out)],
        capture_output=True, text=True,
    )
    return r.returncode, (out.read_bytes() if out.exists() else None), r.stderr


def _require_2_gpus():
    import torch
    n = torch.cuda.device_count()
    if n < 2:
        pytest.skip(f"need >=2 GPUs, found {n}")


# ---------------------------------------------------------------------------
# Worker: GIN devcomm kernel (ncclCommInitRank + ncclDevCommCreate + kernel)
# ---------------------------------------------------------------------------

def _gin_worker(rank, world_size, nccl_so, cubin_bytes, uid_queue, result_queue):
    """Spawned worker: init 2-rank NCCL comm, create ncclDevComm, launch GIN kernel."""
    import torch
    os.environ["NCCL_GIN_ENABLE"] = "1"
    torch.cuda.set_device(rank)

    from cuda.bindings import runtime as cudart
    from cuda.core import Device, LaunchConfig, ObjectCode, launch

    dev = Device(rank)
    dev.set_current()
    lib = ctypes.CDLL(nccl_so)

    # Rank 0 generates unique ID (starts bootstrap server); rank 1 receives it.
    if rank == 0:
        uid = _NCCLUniqueId()
        if lib.ncclGetUniqueId(ctypes.byref(uid)) != 0:
            result_queue.put((rank, "ncclGetUniqueId failed"))
            return
        # Use string_at — bytes()/bytearray() truncate at first NUL in c_char arrays.
        uid_queue.put(ctypes.string_at(ctypes.addressof(uid), ctypes.sizeof(uid)))
    else:
        uid = _NCCLUniqueId.from_buffer_copy(uid_queue.get(timeout=60))

    comm = ncclComm_t()
    if lib.ncclCommInitRank(ctypes.byref(comm), world_size, uid, rank) != 0:
        result_queue.put((rank, "ncclCommInitRank failed"))
        return

    reqs = _NCCLDevCommRequirements()
    reqs.ginContextCount = reqs.ginSignalCount = reqs.railGinBarrierCount = 1
    dev_comm = _NCCLDevComm()
    if lib.ncclDevCommCreate(comm, ctypes.byref(reqs), ctypes.byref(dev_comm)) != 0:
        result_queue.put((rank, "ncclDevCommCreate failed"))
        lib.ncclCommDestroy(comm)
        return

    devcomm_buf = torch.zeros(_NCCL_DEVCOMM_SIZE, dtype=torch.uint8, device=f"cuda:{rank}")
    cudart.cudaMemcpy(devcomm_buf.data_ptr(), ctypes.addressof(dev_comm), _NCCL_DEVCOMM_SIZE,
                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

    out_buf = torch.zeros(2, dtype=torch.int32, device=f"cuda:{rank}")
    kernel = ObjectCode.from_cubin(cubin_bytes).get_kernel("gin_info")
    stream = dev.create_stream()
    launch(stream, LaunchConfig(grid=1, block=1), kernel,
           devcomm_buf.data_ptr(), out_buf.data_ptr())
    stream.sync()

    out = out_buf.cpu().tolist()
    result_queue.put((rank, out[0], out[1]))

    lib.ncclDevCommDestroy(comm, ctypes.byref(dev_comm))
    lib.ncclCommDestroy(comm)
    stream.close()


# ---------------------------------------------------------------------------
# Worker: torch.distributed allreduce
# ---------------------------------------------------------------------------

def _dist_worker(rank, world_size, result_queue):
    """Spawned worker: init torch.distributed (NCCL backend), verify allreduce."""
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        init_method="tcp://127.0.0.1:29500",
    )

    t = torch.tensor([float(rank + 1)], device=f"cuda:{rank}")
    dist.all_reduce(t)               # sum across ranks: 1+2 = 3
    result_queue.put((rank, t.item()))

    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNCCLIntraNode:
    """2-GPU intra-node NCCL tests using torch.multiprocessing."""

    def test_gin_devcomm_rank_info(self, gin_cubin):
        """GIN devcomm kernel: each rank must see rank=i, nRanks=2."""
        import torch.multiprocessing as mp

        rc, cubin_bytes, stderr = gin_cubin
        if rc != 0:
            pytest.skip(f"nvcc failed: {stderr}")
        _require_2_gpus()

        ctx = mp.get_context("spawn")
        uid_q, res_q = ctx.Queue(), ctx.Queue()
        procs = [ctx.Process(target=_gin_worker,
                             args=(r, 2, _NCCL_SO, cubin_bytes, uid_q, res_q))
                 for r in range(2)]
        for p in procs: p.start()
        for p in procs: p.join(timeout=120)

        results = {}
        while not res_q.empty():
            item = res_q.get_nowait()
            if len(item) == 3:
                results[item[0]] = (item[1], item[2])
            else:
                pytest.fail(f"rank {item[0]} error: {item[1]}")

        assert results == {0: (0, 2), 1: (1, 2)}, f"unexpected GIN output: {results}"

    def test_torch_dist_allreduce(self):
        """torch.distributed NCCL allreduce: rank r contributes r+1, sum must be 3."""
        import torch.multiprocessing as mp

        _require_2_gpus()

        ctx = mp.get_context("spawn")
        res_q = ctx.Queue()
        procs = [ctx.Process(target=_dist_worker, args=(r, 2, res_q)) for r in range(2)]
        for p in procs: p.start()
        for p in procs: p.join(timeout=120)

        results = {}
        while not res_q.empty():
            r, val = res_q.get_nowait()
            results[r] = val

        assert len(results) == 2, f"missing results: {results}"
        assert results[0] == 3.0 and results[1] == 3.0, f"allreduce wrong: {results}"
