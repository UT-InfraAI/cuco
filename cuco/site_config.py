"""
Cluster site configuration for CUCo.

Auto-detects CUDA, NCCL, MPI, and GPU environment, persists to ~/.cuco/site.yaml,
and exposes constants that evaluate.py / run_transform.py import.

Usage from workload scripts:
    from cuco.site_config import (
        NVCC, NCCL_INCLUDE, NCCL_STATIC_LIB, CUDA_LIB64,
        MPI_INCLUDE, MPI_INCLUDE_OPENMPI, MPI_LIB,
        GPU_ARCH, CUDA_VISIBLE_DEVICES,
        MULTI_NODE, HOSTFILE_PATH, NETWORK_CONFIG,
    )
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

_CONFIG_DIR = Path.home() / ".cuco"
_CONFIG_FILE = _CONFIG_DIR / "site.yaml"


# ── auto-detection helpers ────────────────────────────────────────────────

def _run(cmd: list[str], default: str = "") -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return r.stdout.strip() if r.returncode == 0 else default
    except Exception:
        return default


def detect_cuda_home() -> Optional[str]:
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    # Prefer versioned installs in /usr/local/ over the system nvcc
    for p in sorted(Path("/usr/local").glob("cuda-*"), reverse=True):
        if (p / "bin" / "nvcc").exists():
            return str(p)
    if Path("/usr/local/cuda/bin/nvcc").exists():
        return "/usr/local/cuda"
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    return None


def detect_nccl_home() -> Optional[str]:
    if os.environ.get("NCCL_HOME"):
        return os.environ["NCCL_HOME"]
    for p in sorted(Path("/usr/local").glob("nccl*"), reverse=True):
        if (p / "include").is_dir() and (p / "lib").is_dir():
            return str(p)
    if Path("/usr/include/nccl.h").exists():
        return "/usr"
    return None


def detect_mpi() -> Dict[str, Optional[str]]:
    """Return {include, include_openmpi, lib} paths for MPI."""
    mpicc = shutil.which("mpicc")
    if mpicc:
        incdirs = _run(["mpicc", "--showme:incdirs"])
        libdirs = _run(["mpicc", "--showme:libdirs"])
        if incdirs and libdirs:
            inc = incdirs.split()[0]
            lib = libdirs.split()[0]
            inc_ompi = str(Path(inc) / "openmpi") if (Path(inc) / "openmpi").is_dir() else inc
            return {"include": inc, "include_openmpi": inc_ompi, "lib": lib}

    for base in ["/usr/lib/x86_64-linux-gnu/openmpi",
                 "/usr/lib64/openmpi", "/usr/local/openmpi"]:
        inc = os.path.join(base, "include")
        lib = os.path.join(base, "lib")
        if os.path.isdir(inc) and os.path.isdir(lib):
            inc_ompi = os.path.join(inc, "openmpi") if os.path.isdir(os.path.join(inc, "openmpi")) else inc
            return {"include": inc, "include_openmpi": inc_ompi, "lib": lib}
    return {"include": None, "include_openmpi": None, "lib": None}


def detect_gpu_arch() -> Optional[str]:
    raw = _run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    if raw:
        first = raw.splitlines()[0].strip()
        major, _, minor = first.partition(".")
        if major.isdigit():
            return f"sm_{major}{minor or '0'}"
    return None


def detect_gpu_count() -> int:
    """Return the number of GPUs visible on this node."""
    raw = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if raw:
        return len([l for l in raw.splitlines() if l.strip()])
    return 1


def auto_detect() -> Dict[str, Any]:
    """Run full auto-detection and return a config dict."""
    cuda_home = detect_cuda_home()
    nccl_home = detect_nccl_home()
    mpi = detect_mpi()
    gpu_arch = detect_gpu_arch()
    gpu_count = detect_gpu_count()

    # Single-node default: all GPUs visible, MPI_NP = gpu count
    visible = ",".join(str(i) for i in range(gpu_count))

    return {
        "cuda_home": cuda_home,
        "nccl_home": nccl_home,
        "mpi_include": mpi["include"],
        "mpi_include_openmpi": mpi["include_openmpi"],
        "mpi_lib": mpi["lib"],
        "gpu_arch": gpu_arch or "sm_80",
        "gpu_count": gpu_count,
        "gpus_per_node": gpu_count,
        "default_np": gpu_count,
        "cuda_visible_devices": visible,
        "multi_node": False,
        "hostfile": None,
        "network": {
            "nccl_socket_ifname": None,
            "nccl_ib_hca": None,
            "nccl_ib_gid_index": None,
            "btl_tcp_if_include": None,
            "oob_tcp_if_include": None,
        },
    }


# ── config persistence ────────────────────────────────────────────────────

def save_config(cfg: Dict[str, Any], path: Path = _CONFIG_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(cfg, default_flow_style=False, sort_keys=False), encoding="utf-8")


def load_config(path: Path = _CONFIG_FILE) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_or_detect_config() -> Dict[str, Any]:
    """Load saved config, or auto-detect and save."""
    cfg = load_config()
    if cfg:
        return cfg
    cfg = auto_detect()
    save_config(cfg)
    return cfg


# ── interactive setup ─────────────────────────────────────────────────────

def interactive_setup() -> Dict[str, Any]:
    """Run auto-detection, show results, let user confirm/override, save."""
    cfg = auto_detect()

    print("\n  Detecting cluster environment...\n")
    _show_detected("CUDA home", cfg["cuda_home"])
    _show_detected("NCCL home", cfg["nccl_home"])
    _show_detected("MPI include", cfg["mpi_include"])
    _show_detected("MPI lib", cfg["mpi_lib"])
    _show_detected("GPU arch", cfg["gpu_arch"])
    _show_detected("GPUs on this node", str(cfg["gpu_count"]))

    print()
    multi = input("  Multi-node setup? [y/N]: ").strip().lower()
    cfg["multi_node"] = multi in ("y", "yes")

    if cfg["multi_node"]:
        # Multi-node: each node typically exposes 1 GPU per rank
        gpn = _prompt(f"    GPUs per node [1]") or "1"
        cfg["gpus_per_node"] = int(gpn)
        visible = ",".join(str(i) for i in range(cfg["gpus_per_node"]))
        cfg["cuda_visible_devices"] = visible

        nodes = _prompt("    Number of nodes [2]") or "2"
        cfg["num_nodes"] = int(nodes)
        cfg["default_np"] = cfg["num_nodes"] * cfg["gpus_per_node"]

        print("\n  Enter hostname for each node:\n")
        node_hostnames = []
        for i in range(1, cfg["num_nodes"] + 1):
            default = socket.gethostname() if i == 1 else ""
            hint = f" [{default}]" if default else ""
            name = _prompt(f"    Node {i}{hint}") or default
            node_hostnames.append(name)
        cfg["node_hostnames"] = node_hostnames

        print("\n  Enter network settings (leave blank to skip):\n")
        cfg["network"]["nccl_socket_ifname"] = _prompt("    NCCL_SOCKET_IFNAME") or None
        cfg["network"]["nccl_ib_hca"] = _prompt("    NCCL_IB_HCA") or None
        cfg["network"]["nccl_ib_gid_index"] = _prompt("    NCCL_IB_GID_INDEX") or None
        cfg["network"]["btl_tcp_if_include"] = _prompt("    btl_tcp_if_include") or None
        cfg["network"]["oob_tcp_if_include"] = _prompt("    oob_tcp_if_include") or None
        cfg["hostfile"] = _prompt("    Hostfile path") or None
    else:
        # Single-node: all GPUs visible, NP = GPU count
        gpn = _prompt(f"    GPUs to use [{cfg['gpu_count']}]") or str(cfg["gpu_count"])
        cfg["gpus_per_node"] = int(gpn)
        visible = ",".join(str(i) for i in range(cfg["gpus_per_node"]))
        cfg["cuda_visible_devices"] = visible
        cfg["default_np"] = cfg["gpus_per_node"]

    save_config(cfg)
    print(f"\n  Saved to {_CONFIG_FILE}\n")
    return cfg


def _show_detected(label: str, value: Any) -> None:
    mark = "[ok]" if value else "[!!]"
    val = value or "(not found)"
    print(f"  {mark} {label:20s} {val}")


def _prompt(label: str) -> str:
    return input(f"{label}: ").strip()


# ── derived constants ─────────────────────────────────────────────────────
# These are loaded at import time and used by evaluate.py / run_transform.py.

_cfg = get_or_detect_config()

_cuda_home = _cfg.get("cuda_home") or "/usr/local/cuda"
_nccl_home = _cfg.get("nccl_home") or "/usr/local/nccl"

NVCC = os.path.join(_cuda_home, "bin", "nvcc")
NCCL_INCLUDE = os.path.join(_nccl_home, "include")
NCCL_STATIC_LIB = os.path.join(_nccl_home, "lib", "libnccl_static.a")
CUDA_LIB64 = os.path.join(_cuda_home, "lib64")

MPI_INCLUDE = _cfg.get("mpi_include") or "/usr/lib/x86_64-linux-gnu/openmpi/include"
MPI_INCLUDE_OPENMPI = _cfg.get("mpi_include_openmpi") or os.path.join(MPI_INCLUDE, "openmpi")
MPI_LIB = _cfg.get("mpi_lib") or "/usr/lib/x86_64-linux-gnu/openmpi/lib"

GPU_ARCH = _cfg.get("gpu_arch") or "sm_80"
CUDA_VISIBLE_DEVICES = str(_cfg.get("cuda_visible_devices", "0"))
DEFAULT_NP = int(_cfg.get("default_np", 2))

MULTI_NODE = bool(_cfg.get("multi_node", False))
HOSTFILE_PATH = _cfg.get("hostfile")

_net = _cfg.get("network") or {}
NETWORK_CONFIG = {
    "nccl_socket_ifname": _net.get("nccl_socket_ifname"),
    "nccl_ib_hca": _net.get("nccl_ib_hca"),
    "nccl_ib_gid_index": _net.get("nccl_ib_gid_index"),
    "btl_tcp_if_include": _net.get("btl_tcp_if_include"),
    "oob_tcp_if_include": _net.get("oob_tcp_if_include"),
}


def build_mpirun_cmd(binary_path: str, np: int, extra_env: Dict[str, str] | None = None) -> list[str]:
    """Build the mpirun command list from site config.

    Single-node: simple mpirun without hostfile or IB flags.
    Multi-node: includes hostfile, NCCL network flags, and MCA TCP settings.
    """
    cmd: list[str] = ["mpirun"]

    if MULTI_NODE and HOSTFILE_PATH:
        cmd.extend(["--hostfile", HOSTFILE_PATH])

    cmd.extend(["-np", str(np), "--map-by", "node"])
    cmd.extend(["-x", "LD_LIBRARY_PATH", "-x", "CUDA_VISIBLE_DEVICES"])
    cmd.extend(["-x", "NCCL_GIN_ENABLE=1"])

    if MULTI_NODE:
        for key in ("nccl_socket_ifname", "nccl_ib_hca", "nccl_ib_gid_index"):
            val = NETWORK_CONFIG.get(key)
            if val:
                env_name = key.upper()
                cmd.extend(["-x", f"{env_name}={val}"])

        for key in ("btl_tcp_if_include", "oob_tcp_if_include"):
            val = NETWORK_CONFIG.get(key)
            if val:
                cmd.extend(["--mca", key, val])

    if extra_env:
        for k, v in extra_env.items():
            cmd.extend(["-x", f"{k}={v}"])

    cmd.append(str(binary_path))
    return cmd
