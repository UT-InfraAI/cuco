#!/usr/bin/env python3
"""Generic evaluation runner for JSON WorkloadSpec-based workloads."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from cuco.workload.spec import build_evaluator, load_workload_spec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec_path", required=True)
    ap.add_argument("--program_path", required=True)
    ap.add_argument("--results_dir", required=True)
    args = ap.parse_args()

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)

    def _fail(error: str, feedback: str = "") -> None:
        (results / "metrics.json").write_text(json.dumps({
            "combined_score": 0.0,
            "public": {},
            "private": {"error": error},
            "text_feedback": feedback,
        }))
        (results / "correct.json").write_text(json.dumps({"correct": False, "error": error}))

    try:
        spec = load_workload_spec(args.spec_path)
        evaluator = build_evaluator(spec)
    except Exception as exc:
        _fail(f"Spec load failed: {exc}\n{traceback.format_exc()}")
        return

    program_path = Path(args.program_path)
    if not program_path.exists():
        _fail(f"Program file not found: {args.program_path}")
        return

    code = program_path.read_text(encoding="utf-8")

    try:
        result = evaluator(code, results)
    except Exception as exc:
        _fail(f"Evaluate raised: {exc}\n{traceback.format_exc()}")
        return

    (results / "metrics.json").write_text(json.dumps({
        "combined_score": result.score,
        "public": result.metrics,
        "private": {"error": result.error} if result.error else {},
        "text_feedback": result.text_feedback,
    }))
    (results / "correct.json").write_text(json.dumps({
        "correct": result.correct,
        "error": result.error or None,
    }))


if __name__ == "__main__":
    main()
