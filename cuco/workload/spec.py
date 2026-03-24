"""JSON-backed workload specification and evaluator factory."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cuco.workload.eval_utils import TIME_PATTERN, VERIFICATION_PASS_STR, score_from_time_ms


@dataclass
class EvalResult:
    correct: bool
    score: float
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    text_feedback: str = ""


@dataclass
class WorkloadSpec:
    schema_version: int
    name: str
    seed_file: Path
    evaluator_type: str
    evaluator_config: Dict[str, Any]
    task_description: str
    api_docs: str = ""
    language: str = "python"
    num_generations: int = 60
    explore_fraction: float = 0.4
    results_dir: Optional[str] = None
    island_seeds: Dict[int, Path] = field(default_factory=dict)
    workload_dir: Path = field(default_factory=Path, repr=False)

    def task_sys_msg(self) -> str:
        msg = self.task_description
        if self.api_docs:
            msg += f"\n\n---\n\n## API Reference\n\n{self.api_docs}"
        return msg


def _read_optional_text(payload: Dict[str, Any], workload_dir: Path, *, key: str) -> str:
    inline_key = key
    file_key = f"{key}_file"
    has_inline = inline_key in payload
    has_file = file_key in payload

    if has_inline and has_file:
        raise ValueError(f"Use only one of '{inline_key}' or '{file_key}'")

    if has_inline:
        value = payload[inline_key]
        if not isinstance(value, str):
            raise ValueError(f"'{inline_key}' must be a string")
        return value

    if has_file:
        file_value = payload[file_key]
        if not isinstance(file_value, str):
            raise ValueError(f"'{file_key}' must be a string")
        path = (workload_dir / file_value).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing {file_key}: {path}")
        return path.read_text(encoding="utf-8")

    return ""


def load_workload_spec(spec_path: str | Path) -> WorkloadSpec:
    p = Path(spec_path).resolve()
    workload_dir = p.parent
    payload = json.loads(p.read_text(encoding="utf-8"))

    schema_version = int(payload.get("schema_version", 0))
    if schema_version != 1:
        raise ValueError(f"Unsupported workload spec schema_version={schema_version}; expected 1")

    for field_name in ("name", "seed_file", "evaluator"):
        if field_name not in payload:
            raise ValueError(f"Missing required field: '{field_name}'")

    name = payload["name"]
    seed_file = payload["seed_file"]
    if not isinstance(name, str) or not name:
        raise ValueError("'name' must be a non-empty string")
    if not isinstance(seed_file, str) or not seed_file:
        raise ValueError("'seed_file' must be a non-empty string")

    evaluator = payload["evaluator"]
    if not isinstance(evaluator, dict):
        raise ValueError("'evaluator' must be an object")
    evaluator_type = evaluator.get("type")
    evaluator_config = evaluator.get("config", {})
    if not isinstance(evaluator_type, str) or not evaluator_type:
        raise ValueError("'evaluator.type' must be a non-empty string")
    if not isinstance(evaluator_config, dict):
        raise ValueError("'evaluator.config' must be an object")

    task_description = _read_optional_text(payload, workload_dir, key="task_description")
    if not task_description:
        raise ValueError("Specify 'task_description' or 'task_description_file'")
    api_docs = _read_optional_text(payload, workload_dir, key="api_docs")

    resolved_seed = (workload_dir / seed_file).resolve()
    if not resolved_seed.exists():
        raise FileNotFoundError(f"Missing seed_file: {resolved_seed}")

    language = payload.get("language", "python")
    num_generations = int(payload.get("num_generations", 60))
    explore_fraction = float(payload.get("explore_fraction", 0.4))
    results_dir = payload.get("results_dir")

    if not isinstance(language, str) or not language:
        raise ValueError("'language' must be a non-empty string")
    if not 0.0 < explore_fraction <= 1.0:
        raise ValueError("'explore_fraction' must be in (0, 1]")
    if results_dir is not None and not isinstance(results_dir, str):
        raise ValueError("'results_dir' must be a string when provided")

    return WorkloadSpec(
        schema_version=schema_version,
        name=name,
        seed_file=resolved_seed,
        evaluator_type=evaluator_type,
        evaluator_config=evaluator_config,
        task_description=task_description,
        api_docs=api_docs,
        language=language,
        num_generations=num_generations,
        explore_fraction=explore_fraction,
        results_dir=results_dir,
        workload_dir=workload_dir,
    )


class SubprocessEvaluator:
    def __init__(
        self,
        *,
        cmd_template: List[str],
        workload_dir: Path,
        verify_contains: str = VERIFICATION_PASS_STR,
        time_pattern: str = TIME_PATTERN.pattern,
        timeout_sec: int = 300,
        num_runs: int = 2,
        score_mode: str = "inverse_time_ms",
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
    ):
        if not cmd_template or not all(isinstance(x, str) for x in cmd_template):
            raise ValueError("SubprocessEvaluator 'cmd_template' must be a non-empty list[str]")
        self._cmd_template = cmd_template
        self._workload_dir = workload_dir
        self._verify_contains = verify_contains
        self._time_regex = re.compile(time_pattern)
        self._timeout_sec = timeout_sec
        self._num_runs = num_runs
        self._score_mode = score_mode
        self._env = env or {}
        self._cwd = (workload_dir / cwd).resolve() if cwd else workload_dir

    def _format_cmd(self, program_path: Path) -> List[str]:
        mapping = {
            "program_path": str(program_path),
            "python_executable": sys.executable,
            "workload_dir": str(self._workload_dir),
        }
        return [arg.format(**mapping) for arg in self._cmd_template]

    def _parse_time_ms(self, stdout: str) -> Optional[float]:
        match = self._time_regex.search(stdout)
        return float(match.group(1)) if match else None

    def _score(self, time_ms: float) -> float:
        if self._score_mode == "inverse_time_ms":
            return score_from_time_ms(time_ms)
        if self._score_mode == "negative_time_ms":
            return -time_ms
        raise ValueError(f"Unsupported score_mode: {self._score_mode}")

    def __call__(self, code: str, work_dir: Path) -> EvalResult:
        prog = work_dir / "program.py"
        prog.write_text(code, encoding="utf-8")

        all_times: List[float] = []
        log_parts: List[str] = []
        run_env = os.environ.copy()
        run_env.update(self._env)

        try:
            for i in range(self._num_runs):
                cmd = self._format_cmd(prog)
                t0 = time.perf_counter()
                r = subprocess.run(
                    cmd,
                    cwd=str(self._cwd),
                    capture_output=True,
                    text=True,
                    timeout=self._timeout_sec,
                    env=run_env,
                )
                elapsed = time.perf_counter() - t0
                log_parts.append(
                    f"=== Run {i + 1} ({elapsed:.1f}s) ===\n"
                    f"$ {' '.join(cmd)}\n"
                    f"{r.stdout}\n{r.stderr}\n"
                )

                if r.returncode != 0:
                    (work_dir / "run.log").write_text("".join(log_parts), encoding="utf-8")
                    return EvalResult(False, 0.0, error=f"exit {r.returncode}\\n{(r.stderr or r.stdout)[:2000]}")

                if self._verify_contains and self._verify_contains not in r.stdout:
                    (work_dir / "run.log").write_text("".join(log_parts), encoding="utf-8")
                    return EvalResult(False, 0.0, error=f"Verification failed\\n{r.stdout[-1000:]}")

                t = self._parse_time_ms(r.stdout)
                if t is not None:
                    all_times.append(t)

        except subprocess.TimeoutExpired:
            return EvalResult(False, 0.0, error=f"Timeout ({self._timeout_sec}s)")

        (work_dir / "run.log").write_text("".join(log_parts), encoding="utf-8")

        if not all_times:
            return EvalResult(False, 0.0, error="No timing found in output")

        best = min(all_times)
        return EvalResult(True, self._score(best), metrics={"time_ms": best, "all_run_times_ms": all_times})


def build_evaluator(spec: WorkloadSpec) -> Callable[[str, Path], EvalResult]:
    if spec.evaluator_type == "subprocess_stdout_v1":
        cfg = spec.evaluator_config
        return SubprocessEvaluator(
            cmd_template=cfg.get("cmd", []),
            workload_dir=spec.workload_dir,
            verify_contains=cfg.get("verify_contains", VERIFICATION_PASS_STR),
            time_pattern=cfg.get("time_pattern", TIME_PATTERN.pattern),
            timeout_sec=int(cfg.get("timeout_sec", 300)),
            num_runs=int(cfg.get("num_runs", 2)),
            score_mode=cfg.get("score_mode", "inverse_time_ms"),
            env=cfg.get("env"),
            cwd=cfg.get("cwd"),
        )

    raise ValueError(f"Unsupported evaluator type: {spec.evaluator_type}")
