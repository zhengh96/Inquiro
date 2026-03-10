"""Inquiro Experiment lifecycle managers 📋.

Provides experiment classes that manage the complete lifecycle
of research and synthesis tasks, including quality gates and retries.

Experiments:
    InquiroBaseExp: Shared base class with common lifecycle logic 📋
    SynthesisExp: Single synthesis task lifecycle 📊
    AnalysisExp: Three-model parallel analysis for discovery mode 🔬
    SearchExp: Search round lifecycle with adaptive reasoning 🔍
    DiscoverySynthesisExp: Multi-LLM synthesis for discovery mode 🧬
    PerspectiveCommentaryExp: Multi-perspective expert commentary 🎭
"""

from inquiro.exps.base_exp import InquiroBaseExp
from inquiro.exps.analysis_exp import AnalysisExp
from inquiro.exps.discovery_synthesis_exp import DiscoverySynthesisExp
from inquiro.exps.perspective_commentary_exp import (
    CommentaryResult,
    PerspectiveCommentaryExp,
)
from inquiro.exps.search_exp import SearchExp
from inquiro.exps.synthesis_exp import SynthesisExp

__all__ = [
    "AnalysisExp",
    "CommentaryResult",
    "DiscoverySynthesisExp",
    "InquiroBaseExp",
    "PerspectiveCommentaryExp",
    "SearchExp",
    "SynthesisExp",
]
