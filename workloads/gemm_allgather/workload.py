#!/usr/bin/env python3
"""Launch evolution for GEMM + LSA Allgather using JSON workload spec."""

from __future__ import annotations

from pathlib import Path

from cuco.workload import run_evolution

HERE = Path(__file__).parent

if __name__ == "__main__":
    run_evolution(
        HERE / "spec.json",
        extra_evo_kwargs={
            "init_program_paths_per_island": {
                1: str(HERE / "gemm_allgather_fused.py"),
            },
        },
    )
