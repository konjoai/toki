"""Toki — adversarial fine-tuning lab for small LLMs."""
from __future__ import annotations

__version__ = "1.1.0"

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
from toki.ranking import (
    Ranking,
    RankingConfig,
    RankingEntry,
    RankingResult,
)
from toki.leaderboard import (
    Leaderboard,
    LeaderboardEntry,
)
from toki.mutator import (
    PromptMutator,
    MutationConfig,
    MutationResult,
    Individual,
)
from toki.distill import (
    CorpusDistiller,
    DistillConfig,
    DistillResult,
    DistillStats,
)
from toki.integration import (
    EvaluatedRobustnessTest,
    EvaluatedReport,
    EvaluatedItem,
    QualityRubric,
    QualityScores,
    LatencyMetrics,
    has_kairu,
)
from toki.judge import (
    JudgeCriteria,
    CriterionScore,
    JudgeVerdict,
    JudgeConfig,
    JudgeBase,
    MockJudge,
    JudgePipeline,
)
from toki.campaign import (
    CampaignConfig,
    CampaignResult,
    RedTeamCampaign,
    run_campaign,
)
from toki.coverage import (
    CATEGORY_AXIS,
    ENCODING_AXIS,
    LANGUAGE_AXIS,
    SEVERITY_AXIS,
    CoverageMap,
    compute_coverage,
    label_positions,
)
from toki.regression import (
    Baseline,
    CategoryDelta,
    RegressionReport,
    compare as compare_regression,
)
from toki.consistency import (
    JUDGE_NAMES,
    ConsistencyEntry,
    ConsistencyEvaluator,
    ConsistencyReport,
    fleiss_kappa,
)
from toki.multilingual import (
    MultilingualGenerator,
    MultilingualPrompt,
    encode_base64,
    encode_rot13,
    encode_zwsp,
    generate_battery,
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
    "Ranking",
    "RankingConfig",
    "RankingEntry",
    "RankingResult",
    "Leaderboard",
    "LeaderboardEntry",
    "PromptMutator",
    "MutationConfig",
    "MutationResult",
    "Individual",
    "CorpusDistiller",
    "DistillConfig",
    "DistillResult",
    "DistillStats",
    "EvaluatedRobustnessTest",
    "EvaluatedReport",
    "EvaluatedItem",
    "QualityRubric",
    "QualityScores",
    "LatencyMetrics",
    "has_kairu",
    "JudgeCriteria",
    "CriterionScore",
    "JudgeVerdict",
    "JudgeConfig",
    "JudgeBase",
    "MockJudge",
    "JudgePipeline",
    "CampaignConfig",
    "CampaignResult",
    "RedTeamCampaign",
    "run_campaign",
    # Phase 11 — P1 roadmap
    "CATEGORY_AXIS",
    "ENCODING_AXIS",
    "LANGUAGE_AXIS",
    "SEVERITY_AXIS",
    "CoverageMap",
    "compute_coverage",
    "label_positions",
    "Baseline",
    "CategoryDelta",
    "RegressionReport",
    "compare_regression",
    "JUDGE_NAMES",
    "ConsistencyEntry",
    "ConsistencyEvaluator",
    "ConsistencyReport",
    "fleiss_kappa",
    "MultilingualGenerator",
    "MultilingualPrompt",
    "encode_base64",
    "encode_rot13",
    "encode_zwsp",
    "generate_battery",
]
