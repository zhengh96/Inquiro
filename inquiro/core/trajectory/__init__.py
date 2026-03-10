"""Inquiro Discovery Trajectory recording system 📊.

Provides structured recording of the Discovery pipeline execution
for debugging (L1), quality comparison (L2), and strategy
optimization (L3).

Uses composition over inheritance with EvoMaster Trajectory.
"""

from inquiro.core.trajectory.index import (  # noqa: F401
    CostBreakdown,
    RoundRecord,
    TrajectoryIndex,
    TrajectoryRecord,
    TrendPoint,
)
from inquiro.core.trajectory.query_analyzer import (  # noqa: F401
    QueryTemplateAnalyzer,
    TemplateEffectivenessRecord,
)
from inquiro.core.trajectory.models import (  # noqa: F401
    AnalysisPhaseRecord,
    CleaningPhaseRecord,
    ConsensusRecord,
    DiscoveryRoundRecord,
    DiscoverySummary,
    DiscoveryTrajectory,
    FocusPromptRecord,
    GapPhaseRecord,
    ModelAnalysisRecord,
    QueryRecord,
    SearchPhaseRecord,
    ServerStats,
    SynthesisRecord,
    TrajectoryEvent,
    TrajectoryEventType,
)
from inquiro.core.trajectory.gap_hints import (  # noqa: F401
    GapSearchHint,
    GapSearchHintAccumulator,
)
from inquiro.core.trajectory.feedback import (  # noqa: F401
    FeedbackResult,
    TrajectoryFeedbackProvider,
)
from inquiro.core.trajectory.writer import TrajectoryWriter  # noqa: F401
