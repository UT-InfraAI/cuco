"""Minimal shared evaluation helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Optional

VERIFICATION_PASS_STR = "Verification: PASS"
TIME_PATTERN = re.compile(r"Time:\\s*([0-9.]+)\\s*ms")


def parse_time_ms(stdout: str) -> Optional[float]:
    m = TIME_PATTERN.search(stdout)
    return float(m.group(1)) if m else None


def score_from_time_ms(time_ms: float) -> float:
    if time_ms <= 0:
        return 0.0
    return 1000.0 / time_ms


def write_json(path: Path, payload: Dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def best_score_so_far(evo_root: Path) -> float:
    best = 0.0
    for f in evo_root.glob("gen_*/results/metrics.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            best = max(best, float(d.get("combined_score", 0.0)))
        except Exception:
            continue
    return best


