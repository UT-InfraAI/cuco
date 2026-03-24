"""Shared evolution-runner helpers for JSON workload specs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from cuco.core import EvolutionConfig, EvolutionRunner
from cuco.database import DatabaseConfig
from cuco.launch import LocalJobConfig
from cuco.workload.spec import load_workload_spec


def default_db_config(db_path: str = "evolution_db.sqlite") -> DatabaseConfig:
    return DatabaseConfig(
        db_path=db_path,
        num_islands=2,
        archive_size=60,
        elite_selection_ratio=0.3,
        num_archive_inspirations=3,
        num_top_k_inspirations=3,
        migration_interval=8,
        migration_rate=0.15,
        island_elitism=True,
        enforce_island_separation=True,
        parent_selection_strategy="weighted",
        parent_selection_lambda=8.0,
    )


def default_job_config(eval_program: str = "evaluate.py", extra_eval_args: Optional[Dict[str, Any]] = None) -> LocalJobConfig:
    return LocalJobConfig(eval_program_path=eval_program, extra_cmd_args=extra_eval_args or {})


_PHASE_CONFIGS = {
    "explore": {
        "patch_type_probs": [0.15, 0.70, 0.15],
        "temperatures": [0.2, 0.5, 0.8],
    },
    "exploit": {
        "patch_type_probs": [0.25, 0.60, 0.15],
        "temperatures": [0.0, 0.2, 0.5],
    },
}


def two_phase_main(
    *,
    task_sys_msg: str,
    language: str,
    init_program: str,
    results_dir: str,
    num_generations: int,
    explore_fraction: float = 0.4,
    llm_models: Optional[List[str]] = None,
    llm_kwargs: Optional[Dict[str, Any]] = None,
    meta_llm_models: Optional[List[str]] = None,
    meta_llm_kwargs: Optional[Dict[str, Any]] = None,
    max_parallel_jobs: int = 1,
    extra_evo_kwargs: Optional[Dict[str, Any]] = None,
    eval_program: str = "evaluate.py",
    extra_eval_args: Optional[Dict[str, Any]] = None,
) -> None:
    if llm_models is None:
        llm_models = ["bedrock/us.anthropic.claude-opus-4-6-v1"]
    if llm_kwargs is None:
        llm_kwargs = {"max_tokens": 32768}
    if meta_llm_models is None:
        meta_llm_models = ["bedrock/us.anthropic.claude-opus-4-6-v1"]
    if meta_llm_kwargs is None:
        meta_llm_kwargs = {"temperatures": [0.0], "max_tokens": 8192}
    if extra_evo_kwargs is None:
        extra_evo_kwargs = {}

    Path(results_dir).mkdir(parents=True, exist_ok=True)
    explore_gens = max(1, int(num_generations * explore_fraction))

    for phase, phase_gens in [("explore", explore_gens), ("exploit", num_generations)]:
        phase_cfg = _PHASE_CONFIGS[phase]
        phase_llm_kwargs = dict(llm_kwargs)
        if "temperatures" not in phase_llm_kwargs:
            phase_llm_kwargs["temperatures"] = phase_cfg["temperatures"]

        evo_config = EvolutionConfig(
            task_sys_msg=task_sys_msg,
            language=language,
            patch_types=["diff", "full", "cross"],
            patch_type_probs=phase_cfg["patch_type_probs"],
            num_generations=phase_gens,
            max_parallel_jobs=max_parallel_jobs,
            max_patch_resamples=3,
            max_patch_attempts=4,
            job_type="local",
            llm_models=llm_models,
            llm_kwargs=phase_llm_kwargs,
            meta_rec_interval=8,
            meta_llm_models=meta_llm_models,
            meta_llm_kwargs=meta_llm_kwargs,
            meta_max_recommendations=5,
            init_program_path=init_program,
            results_dir=results_dir,
            max_novelty_attempts=5,
            code_embed_sim_threshold=0.995,
            use_text_feedback=True,
            embedding_model="bedrock-amazon.titan-embed-text-v1",
            **extra_evo_kwargs,
        )

        EvolutionRunner(
            evo_config=evo_config,
            job_config=default_job_config(eval_program, extra_eval_args),
            db_config=default_db_config(),
            verbose=True,
        ).run()


def run_evolution(spec_path: str | Path, extra_evo_kwargs: Optional[Dict[str, Any]] = None) -> None:
    spec_path = Path(spec_path).resolve()
    spec = load_workload_spec(spec_path)
    eval_runner = Path(__file__).parent / "eval_runner.py"

    p = argparse.ArgumentParser(description=f"Evolve {spec.name}")
    p.add_argument("--num_generations", type=int, default=spec.num_generations)
    p.add_argument("--explore_fraction", type=float, default=spec.explore_fraction)
    p.add_argument("--results_dir", type=str, default=spec.results_dir or f"results_{spec.name}")
    p.add_argument("--init_program", type=str, default=str(spec.seed_file))
    args = p.parse_args()

    two_phase_main(
        task_sys_msg=spec.task_sys_msg(),
        language=spec.language,
        init_program=args.init_program,
        results_dir=args.results_dir,
        num_generations=args.num_generations,
        explore_fraction=args.explore_fraction,
        eval_program=str(eval_runner),
        extra_eval_args={"spec_path": str(spec_path)},
        extra_evo_kwargs=extra_evo_kwargs,
    )
