"""Stores and loads experiment results from timestamped directories."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class ExperimentResult:
    name: str
    timestamp: str                     # ISO 8601
    model_name: str
    seed: int
    pre_score: float                   # mean robustness score before fine-tuning
    post_score: Optional[float]        # mean robustness score after (None if no fine-tuning)
    total_prompts: int
    category_scores: dict              # per-category mean scores
    config: dict                       # full config snapshot

    @property
    def improvement(self) -> Optional[float]:
        if self.post_score is None:
            return None
        return self.post_score - self.pre_score

    def save(self, base_dir: str = "experiments/runs") -> Path:
        """Save to experiments/runs/<timestamp>_<name>/result.json"""
        run_dir = Path(base_dir) / f"{self.timestamp}_{self.name}"
        run_dir.mkdir(parents=True, exist_ok=True)
        out = run_dir / "result.json"
        out.write_text(json.dumps(asdict(self), indent=2))
        return out

    @classmethod
    def load(cls, path) -> "ExperimentResult":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    @classmethod
    def make_timestamp(cls) -> str:
        return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")


def list_experiments(base_dir: str = "experiments/runs") -> list:
    """Return sorted list of result.json paths in base_dir."""
    p = Path(base_dir)
    if not p.exists():
        return []
    return sorted(p.glob("*/result.json"))
