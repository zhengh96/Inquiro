"""Inquiro Evolution — domain-agnostic self-evolution infrastructure 🧬.

Provides the generic infrastructure for experience-based learning:

- **types** — Data models: Experience, ExperienceQuery, TrajectorySnapshot
- **store** — ExperienceStore: PostgreSQL CRUD with namespace isolation
- **collector** — TrajectoryCollector: Extract execution data from trajectories
- **extractor** — ExperienceExtractor: LLM-powered insight extraction engine
- **enricher** — PromptEnricher: Inject relevant experiences into agent prompts
- **fitness** — FitnessEvaluator: Multi-dimensional fitness scoring
- **ranker** — ExperienceRanker: Pruning, decay, and conflict resolution

All domain-specific configuration (categories, prompts, fitness dimensions,
thresholds) is injected via ``EvolutionProfile`` from the upper layer.
Inquiro treats all string fields as opaque — it does NOT interpret
category names, context tags, or insight content semantically.
"""

from inquiro.evolution.discovery_collector import DiscoveryTrajectoryCollector
from inquiro.evolution.enricher import PromptEnricher
from inquiro.evolution.extractor import ExperienceExtractor
from inquiro.evolution.fitness import FitnessEvaluator
from inquiro.evolution.ranker import ExperienceRanker
from inquiro.evolution.types import (
    Experience,
    ExperienceQuery,
    FitnessUpdate,
    PruneConfig,
    TrajectorySnapshot,
    ToolCallRecord,
    ResultMetrics,
    EnrichmentResult,
)

__all__ = [
    "DiscoveryTrajectoryCollector",
    "Experience",
    "ExperienceQuery",
    "FitnessUpdate",
    "PruneConfig",
    "TrajectorySnapshot",
    "ToolCallRecord",
    "ResultMetrics",
    "EnrichmentResult",
    "ExperienceExtractor",
    "PromptEnricher",
    "FitnessEvaluator",
    "ExperienceRanker",
]
