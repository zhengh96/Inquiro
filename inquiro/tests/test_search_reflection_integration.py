"""Integration tests for intensity-based search reflection injection 🧪.

Verifies:
- STANDARD mode: prompt contains diversity constraints, no ThinkTool.
- DISCOVERY mode: prompt contains reflection protocol, ThinkTool registered.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from inquiro.core.types import DiscoveryConfig
from inquiro.exps.search_exp import SearchExp
from inquiro.tests.mock_helpers import (
    MockLLM,
    build_sample_evaluation_task,
)


# ============================================================================
# 🏭 Helpers
# ============================================================================


def _build_discovery_config(**overrides: Any) -> DiscoveryConfig:
    """Build a DiscoveryConfig with optional overrides 🔧."""
    defaults: dict[str, Any] = {
        "max_rounds": 3,
        "timeout_per_round": 300,
    }
    defaults.update(overrides)
    return DiscoveryConfig(**defaults)


def _build_search_exp(
    adaptive_search: bool = False,
) -> SearchExp:
    """Create a SearchExp with sensible test defaults 🔍."""
    from inquiro.infrastructure.cancellation import CancellationToken
    from inquiro.infrastructure.cost_tracker import CostTracker

    tools = MagicMock()
    tools.__len__ = MagicMock(return_value=0)
    tools.get_mcp_server_names = MagicMock(return_value=[])

    return SearchExp(
        llm=MockLLM(),
        tools=tools,
        event_emitter=MagicMock(),
        cost_tracker=CostTracker(max_per_task=10.0, max_total=100.0),
        cancellation_token=CancellationToken(),
        adaptive_search=adaptive_search,
    )


# ============================================================================
# 🧪 Tests
# ============================================================================


class TestIntensityBasedReflection:
    """Tests for intensity-driven reflection injection 🎯."""

    def test_standard_mode_no_think_tool(self) -> None:
        """STANDARD intensity should NOT register ThinkTool ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        system_prompt = exp._render_system_prompt(task, intensity="standard")
        user_prompt = "test"

        # Track tool registration calls
        registered_tools: list[Any] = []
        original_register = exp.tools.register

        def tracking_register(tool: Any) -> None:
            registered_tools.append(tool)
            return original_register(tool)

        exp.tools.register = tracking_register

        exp._create_search_agent(
            task=task,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            intensity="standard",
        )

        # No ThinkTool should be registered
        think_tools = [
            t for t in registered_tools
            if type(t).__name__ == "ThinkTool"
        ]
        assert len(think_tools) == 0, (
            "ThinkTool should NOT be registered in STANDARD mode"
        )

    def test_discovery_mode_registers_think_tool(self) -> None:
        """DISCOVERY intensity should register ThinkTool ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()
        system_prompt = exp._render_system_prompt(
            task, intensity="discovery",
        )
        user_prompt = "test"

        # Track tool registration calls
        registered_tools: list[Any] = []
        original_register = exp.tools.register

        def tracking_register(tool: Any) -> None:
            registered_tools.append(tool)
            return original_register(tool)

        exp.tools.register = tracking_register

        exp._create_search_agent(
            task=task,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            intensity="discovery",
        )

        # ThinkTool should be registered
        think_tools = [
            t for t in registered_tools
            if type(t).__name__ == "ThinkTool"
        ]
        assert len(think_tools) == 1, (
            "ThinkTool should be registered in DISCOVERY mode"
        )

    def test_standard_prompt_has_diversity_constraints(self) -> None:
        """STANDARD prompt should contain pre-finish diversity rules ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task, intensity="standard")

        assert "at least 3 different" in prompt, (
            "STANDARD prompt must include tool diversity constraint"
        )
        assert "fewer than 5 evidence-producing" in prompt, (
            "STANDARD prompt must include minimum evidence threshold"
        )
        assert "60%" in prompt, (
            "STANDARD prompt must include tool concentration limit"
        )
        # Should NOT contain reflection protocol
        assert "REFLECTION PROTOCOL" not in prompt, (
            "STANDARD prompt should not contain discovery reflection"
        )

    def test_discovery_prompt_has_reflection_protocol(self) -> None:
        """DISCOVERY prompt should contain reflection protocol ✅."""
        exp = _build_search_exp()
        task = build_sample_evaluation_task()

        prompt = exp._render_system_prompt(task, intensity="discovery")

        # Should have both the base diversity constraints AND reflection
        assert "at least 3 different" in prompt, (
            "DISCOVERY prompt must include tool diversity constraint"
        )
        assert "REFLECTION PROTOCOL" in prompt, (
            "DISCOVERY prompt must include reflection protocol"
        )
        assert "think" in prompt.lower(), (
            "DISCOVERY prompt must reference the think tool"
        )
        assert "under 150 words" in prompt, (
            "DISCOVERY prompt must include brevity constraint"
        )
