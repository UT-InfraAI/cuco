"""
Scaffolding logic for ``cuco_init``.

Creates a new workload directory from a seed CUDA file, renders templates,
copies support files, and prints the next-step instructions.
"""

from __future__ import annotations

import os
import re
import shutil
import socket
from pathlib import Path
from typing import Dict, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Reference workload whose run_evo / run_transform we clone and patch
_REFERENCE_WORKLOAD = _PROJECT_ROOT / "workloads" / "ds_v3_moe"


# ── seed-file validation ──────────────────────────────────────────────────

def validate_seed(seed_path: Path) -> Dict[str, bool]:
    """Scan the seed .cu file for required patterns."""
    text = seed_path.read_text(encoding="utf-8")
    return {
        "has_time_printf": bool(re.search(r'printf\s*\(\s*"Time:', text)),
        "has_verification": bool(re.search(r'printf\s*\(.*Verification.*PASS', text, re.DOTALL)),
        "has_nccl": bool(re.search(r'#include\s*[<"]nccl', text)),
    }


# ── template rendering ────────────────────────────────────────────────────

def _render_template(tmpl_path: Path, replacements: Dict[str, str]) -> str:
    """Read a .tmpl file and substitute {{KEY}} placeholders."""
    text = tmpl_path.read_text(encoding="utf-8")
    for key, val in replacements.items():
        text = text.replace("{{" + key + "}}", val)
    return text


def _patch_run_evo(text: str, workload_name: str, num_generations: int) -> str:
    """Patch a copy of the reference run_evo.py with workload-specific defaults."""
    text = re.sub(
        r'default="results_ds_v3_moe"',
        f'default="results_{workload_name}"',
        text,
    )
    text = re.sub(
        r'default="ds_v3_moe\.cu"',
        f'default="{workload_name}.cu"',
        text,
        count=0,  # replace all occurrences
    )
    text = re.sub(
        r'default=60\b',
        f'default={num_generations}',
        text,
        count=1,
    )
    return text


def _patch_run_transform(text: str, workload_name: str) -> str:
    """Patch a copy of the reference run_transform.py with workload-specific defaults."""
    text = re.sub(
        r'"ds_v3_moe\.cu"',
        f'"{workload_name}.cu"',
        text,
    )
    text = text.replace(
        '_transform_work"',
        '_transform_host_work"',
    )
    text = text.replace(
        '_transform_output"',
        '_transform_host_output"',
    )
    return text


# ── hostfile helpers ──────────────────────────────────────────────────────

def _generate_hostfile(
    num_nodes: int,
    gpus_per_node: int,
    node_hostnames: Optional[list] = None,
) -> str:
    """Generate a hostfile from configured hostnames, or placeholders as fallback."""
    if node_hostnames and len(node_hostnames) == num_nodes:
        lines = [f"{h} slots={gpus_per_node}" for h in node_hostnames]
    else:
        hostname = socket.gethostname()
        lines = [f"{hostname} slots={gpus_per_node}"]
        for i in range(2, num_nodes + 1):
            lines.append(f"<node{i}-hostname> slots={gpus_per_node}")
        if num_nodes > 1:
            lines.append(
                "# Replace <node*-hostname> placeholders with actual hostnames"
            )
    return "\n".join(lines) + "\n"


def _warn_hostfile_slots(hostfile: Path, expected_gpus: int) -> None:
    """Print a warning if any slots= value in the hostfile doesn't match gpus_per_node."""
    text = hostfile.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.search(r"slots=(\d+)", line)
        if match and int(match.group(1)) != expected_gpus:
            print(
                f"\n  \033[93mWARNING: Hostfile has slots={match.group(1)} "
                f"but site config says gpus_per_node={expected_gpus}.\033[0m"
                f"\n  Check {hostfile} and update if needed.\n"
            )
            return


# ── main scaffold ─────────────────────────────────────────────────────────

def scaffold(
    seed_path: Path,
    workload_name: str,
    output_dir: Path,
    mpi_np: int = 2,
    num_generations: int = 50,
    verification_str: str = "Verification: PASS",
    site_cfg: Optional[Dict] = None,
) -> Path:
    """Create a ready-to-run workload directory.

    Returns the Path to the created directory.
    """
    if output_dir.exists():
        raise FileExistsError(f"Directory already exists: {output_dir}")

    output_dir.mkdir(parents=True)

    # 1. Copy seed file
    shutil.copy2(seed_path, output_dir / seed_path.name)

    # 2. Render evaluate.py from template
    replacements = {
        "MPI_NP": str(mpi_np),
        "VERIFICATION_PASS_STR": verification_str,
        "WORKLOAD_NAME": workload_name,
    }
    eval_tmpl = _TEMPLATE_DIR / "evaluate.py.tmpl"
    (output_dir / "evaluate.py").write_text(
        _render_template(eval_tmpl, replacements), encoding="utf-8"
    )

    # 3. Copy + patch run_evo.py from reference
    ref_evo = _REFERENCE_WORKLOAD / "run_evo.py"
    if ref_evo.exists():
        evo_text = ref_evo.read_text(encoding="utf-8")
        evo_text = _patch_run_evo(evo_text, workload_name, num_generations)
        (output_dir / "run_evo.py").write_text(evo_text, encoding="utf-8")

    # 4. Copy + patch run_transform.py from reference
    ref_transform = _REFERENCE_WORKLOAD / "run_transform.py"
    if ref_transform.exists():
        tx_text = ref_transform.read_text(encoding="utf-8")
        tx_text = _patch_run_transform(tx_text, workload_name)
        (output_dir / "run_transform.py").write_text(tx_text, encoding="utf-8")

    # 5. Copy nccl_api_docs.py verbatim
    ref_docs = _REFERENCE_WORKLOAD / "nccl_api_docs.py"
    if ref_docs.exists():
        shutil.copy2(ref_docs, output_dir / "nccl_api_docs.py")

    # 6. Render .gitignore
    gi_tmpl = _TEMPLATE_DIR / "gitignore.tmpl"
    (output_dir / ".gitignore").write_text(
        _render_template(gi_tmpl, replacements), encoding="utf-8"
    )

    # 7. Create build/hostfile for multi-node setups
    if site_cfg and site_cfg.get("multi_node"):
        build_dir = output_dir / "build"
        build_dir.mkdir(exist_ok=True)
        hostfile_src = site_cfg.get("hostfile")
        hostfile_dst = build_dir / "hostfile"
        gpus_per_node = site_cfg.get("gpus_per_node", 1)
        if hostfile_src and Path(hostfile_src).exists():
            shutil.copy2(hostfile_src, hostfile_dst)
            _warn_hostfile_slots(hostfile_dst, gpus_per_node)
        else:
            num_nodes = site_cfg.get("num_nodes", 2)
            node_hostnames = site_cfg.get("node_hostnames")
            hostfile_dst.write_text(
                _generate_hostfile(num_nodes, gpus_per_node, node_hostnames),
                encoding="utf-8",
            )

    return output_dir


# ── pretty-print results ──────────────────────────────────────────────────

_GREEN = "\033[92m"
_YELLOW = "\033[93m"
_CYAN = "\033[96m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def print_summary(
    output_dir: Path,
    workload_name: str,
    seed_name: str,
    mpi_np: int,
    num_generations: int,
    checks: Dict[str, bool],
    site_cfg: Dict,
) -> None:
    """Print a user-friendly summary after scaffolding."""
    rel = output_dir.relative_to(_PROJECT_ROOT) if output_dir.is_relative_to(_PROJECT_ROOT) else output_dir

    print()
    _ok = f"  {_GREEN}[ok]{_RESET}"
    _warn = f"  {_YELLOW}[!!]{_RESET}"

    # --- Site config summary ---
    cuda_home = site_cfg.get("cuda_home") or "(not found)"
    nccl_home = site_cfg.get("nccl_home") or "(not found)"
    arch = site_cfg.get("gpu_arch") or "sm_80"
    mode = "multi-node" if site_cfg.get("multi_node") else "single-node"

    print(f"  {_BOLD}Using cluster config from ~/.cuco/site.yaml{_RESET}")
    print(f"    CUDA: {cuda_home}  |  NCCL: {nccl_home}")
    print(f"    Arch: {arch}  |  Mode: {mode}")
    print()

    # --- Created files ---
    is_multi = site_cfg.get("multi_node", False)
    print(f"  {_GREEN}Created {rel}/{_RESET}")
    files = [
        (f"{seed_name}", "seed kernel"),
        ("evaluate.py", "build + run + score"),
        ("run_evo.py", "evolution launcher"),
        ("run_transform.py", "host-to-device transform"),
        ("nccl_api_docs.py", "NCCL API reference"),
        (".gitignore", ""),
    ]
    if is_multi:
        files.append(("build/hostfile", "MPI hostfile for multi-node"))
    for fname, desc in files:
        desc_str = f"  ({desc})" if desc else ""
        print(f"    {fname:24s}{desc_str}")
    print()

    # --- Seed checks ---
    print(f"  {_BOLD}Seed file checks:{_RESET}")
    mark = _ok if checks["has_time_printf"] else _warn
    print(f'{mark} Timing output: printf("Time: ...")')
    mark = _ok if checks["has_verification"] else _warn
    print(f'{mark} Verification: printf("Verification: PASS")')
    mark = _ok if checks["has_nccl"] else _warn
    print(f"{mark} NCCL includes")
    if not checks["has_time_printf"]:
        print(f'\n  {_YELLOW}WARNING: Seed file must print "Time: X.XXXX ms" for the evaluator to parse timing.{_RESET}')
    if not checks["has_verification"]:
        print(f'\n  {_YELLOW}WARNING: Seed file must print "Verification: PASS" for correctness checking.{_RESET}')
    print()

    # --- Review checklist ---
    print(f"  {_BOLD}Review evaluate.py if needed:{_RESET}")
    print(f"    - VERIFICATION_PASS_STR: \"{checks.get('verification_str', 'Verification: PASS')}\"")
    print(f"      Change if your kernel uses a different success string.")
    print(f"    - MPI_NP: {mpi_np}")
    print(f"      Change if your kernel needs a different rank count.")
    print()

    # --- Multi-node reminder ---
    if is_multi:
        hostfile_src = site_cfg.get("hostfile")
        if hostfile_src and Path(hostfile_src).exists():
            print(f"  {_BOLD}Multi-node:{_RESET} Hostfile copied from {hostfile_src}")
        else:
            print(f"  {_YELLOW}Multi-node: Please edit build/hostfile with your node hostnames.{_RESET}")
        print(f"    Network: IB_HCA={site_cfg.get('network', {}).get('nccl_ib_hca') or '(not set)'}, "
              f"IFNAME={site_cfg.get('network', {}).get('nccl_socket_ifname') or '(not set)'}")
        print()

    # --- Run command ---
    print(f"  {_BOLD}To start evolution:{_RESET}")
    print(f"    {_CYAN}cd {rel}{_RESET}")
    print(f"    {_CYAN}python run_evo.py --init_program {seed_name} \\{_RESET}")
    print(f"    {_CYAN}                  --results_dir results_{workload_name} \\{_RESET}")
    print(f"    {_CYAN}                  --num_generations {num_generations}{_RESET}")
    print()
