"""Toki — adversarial fine-tuning lab for small LLMs."""
from __future__ import annotations

__version__ = "0.8.0"

from toki.generate import AdversarialGenerator
from toki.evaluate import RobustnessEvaluator
from toki.dataset import AdversarialDataset
from toki.experiment import TokiExperiment, ExperimentConfig
from toki.results import ExperimentResult
from toki.benchmark import BenchmarkReport, BenchmarkStats, generate_report
from toki.report import to_json, to_html
from toki.hub import DatasetMetadata, HubUploader, build_dataset_card, write_card
from toki.pipeline import HardeningPipeline, PipelineConfig, PipelineResult, RoundResult
from toki.compare import (
    BASELINES,
    ComparisonConfig,
    ComparisonResult,
    ModelScores,
    ModelSpec,
    compare_models,
)
from toki.leaderboard import (
    Leaderboard,
    LeaderboardConfig,
    LeaderboardEntry,
    LeaderboardResult,
)
from toki.mutator import (
    PromptMutator,
    MutationConfig,
    MutationResult,
    Individual,
)

__all__ = [
    "AdversarialGenerator",
    "RobustnessEvaluator",
    "AdversarialDataset",
    "TokiExperiment",
    "ExperimentConfig",
    "ExperimentResult",
    "BenchmarkReport",
    "BenchmarkStats",
    "generate_report",
    "to_json",
    "to_html",
    "DatasetMetadata",
    "HubUploader",
    "build_dataset_card",
    "write_card",
    "HardeningPipeline",
    "PipelineConfig",
    "PipelineResult",
    "RoundResult",
    "BASELINES",
    "ComparisonConfig",
    "ComparisonResult",
    "ModelScores",
    "ModelSpec",
    "compare_models",
    "Leaderboard",
    "LeaderboardConfig",
    "LeaderboardEntry",
    "LeaderboardResult",
    "PromptMutator",
    "MutationConfig",
    "MutationResult",
    "Individual",
]
