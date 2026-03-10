"""Inquiro Agent implementations 🎯.

Provides domain-agnostic research and synthesis agents
built on top of EvoMaster's BaseAgent.

Base Classes:
    InquiroAgentBase: Shared infrastructure for all Inquiro agents 🏗️

Agents:
    SearchAgent: Search agent with adaptive reasoning (SRDR loop) 🔍
    SynthesisAgent: Multi-report synthesis (Read-Reason-Synthesize) 📊
"""

from inquiro.agents.base import InquiroAgentBase
from inquiro.agents.search_agent import SearchAgent
from inquiro.agents.synthesis_agent import SynthesisAgent

__all__ = [
    "InquiroAgentBase",
    "SearchAgent",
    "SynthesisAgent",
]
