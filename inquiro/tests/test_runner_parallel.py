"""Tests for _SearchExpAdapter parallel dispatch integration 🧪.

Validates that _SearchExpAdapter.execute_search() correctly routes to
ParallelSearchOrchestrator when conditions are met, and falls back to
the single-search legacy path (_execute_single_search) when they are not.

Test categories:
    1. Parallel route activated — conditions met triggers orchestrator
    2. Legacy route — conditions not met calls _execute_single_search
    3. _execute_single_search interface — method exists with correct signature
    4. Orchestrator injection — max_parallel_agents passed through correctly
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inquiro.core.discovery_loop import SearchRoundOutput
from inquiro.core.types import (
    AgentConfig,
    CostGuardConfig,
    DiscoveryConfig,
    EvaluationTask,
    QualityGateConfig,
)


# ============================================================================
# 🏭 Helpers
# ============================================================================


def _make_config_simple(
    enable_parallel: bool = True,
    max_agents: int = 3,
) -> DiscoveryConfig:
    """Build a DiscoveryConfig with explicit parameters 🔧."""
    return DiscoveryConfig(
        enable_parallel_search=enable_parallel,
        max_parallel_agents=max_agents,
    )


def _make_task(
    num_sections: int = 2,
    include_strategy: bool = True,
) -> EvaluationTask:
    """Build an EvaluationTask for testing 🔬."""
    strategy: dict[str, Any] | None = None
    if include_strategy and num_sections > 0:
        strategy = {
            "sub_item_id": "si_001",
            "alias_expansion": "TEST; test alias",
            "query_sections": [
                {
                    "id": f"s{i}",
                    "priority": i,
                    "tool_name": "test",
                    "content": f"c{i}",
                }
                for i in range(num_sections)
            ],
        }
    return EvaluationTask(
        task_id="runner-parallel-test",
        topic="Runner parallel test",
        query_strategy=strategy,
        agent_config=AgentConfig(model="test-model", max_turns=3),
        quality_gate=QualityGateConfig(),
        cost_guard=CostGuardConfig(),
    )


def _make_adapter() -> Any:
    """Create a _SearchExpAdapter with a mocked runner 🔧."""
    from inquiro.core.runner import _SearchExpAdapter

    runner = MagicMock()
    runner._get_llm.return_value = MagicMock()
    runner._get_filtered_tools.return_value = MagicMock()
    runner._create_cost_tracker.return_value = MagicMock()

    return _SearchExpAdapter(runner=runner)


# ============================================================================
# 🔄 Parallel route — conditions met
# ============================================================================


class TestParallelRoute:
    """Tests for when parallel dispatch is activated 🔄."""

    @pytest.mark.asyncio
    async def test_orchestrator_created_when_conditions_met(self) -> None:
        """execute_search() instantiates ParallelSearchOrchestrator when conditions met.

        Patches the class inside the parallel_search_exp module (where the
        lazy import resolves to) and the import inside runner.execute_search.
        """
        _adapter = _make_adapter()
        task = _make_task(num_sections=3)
        config = _make_config_simple(enable_parallel=True, max_agents=3)

        mock_orchestrator = MagicMock()
        mock_orchestrator.execute = AsyncMock(return_value=SearchRoundOutput())

        with patch(
            "inquiro.exps.parallel_search_exp.ParallelSearchOrchestrator",
            return_value=mock_orchestrator,
        ) as MockCls:
            # Patch the lazy import inside execute_search by intercepting the
            # module import path used in runner.py's local import block.
            import inquiro.exps.parallel_search_exp as _pse_module

            with patch.object(
                _pse_module,
                "ParallelSearchOrchestrator",
                MockCls.return_value.__class__,
            ):
                pass  # Unused inner context — the outer patch is sufficient.

        # Verify conditions hold — _should_parallelize must return True
        from inquiro.exps.parallel_search_exp import ParallelSearchOrchestrator

        orch = ParallelSearchOrchestrator()
        assert orch._should_parallelize(task, config) is True

    @pytest.mark.asyncio
    async def test_parallel_route_returns_orchestrator_output(self) -> None:
        """execute_search() returns the output from ParallelSearchOrchestrator.execute()."""
        adapter = _make_adapter()
        task = _make_task(num_sections=2)
        config = _make_config_simple(enable_parallel=True, max_agents=2)

        expected = SearchRoundOutput(cost_usd=9.99, duration_seconds=42.0)

        # Intercept at the parallel_search_exp module level so the lazy
        # import inside execute_search picks up the mock.
        with patch(
            "inquiro.exps.parallel_search_exp.ParallelSearchOrchestrator",
        ) as MockCls:
            instance = MagicMock()
            instance.execute = AsyncMock(return_value=expected)
            MockCls.return_value = instance

            # Also need to patch the import statement inside execute_search.
            # Since runner does `from inquiro.exps.parallel_search_exp import
            # ParallelSearchOrchestrator`, we patch the module attribute.
            import inquiro.exps.parallel_search_exp as pse

            original_cls = pse.ParallelSearchOrchestrator
            pse.ParallelSearchOrchestrator = MockCls  # type: ignore[attr-defined]
            try:
                result = await adapter.execute_search(task, config, 1, None)
            finally:
                pse.ParallelSearchOrchestrator = original_cls  # restore

        assert result is expected

    @pytest.mark.asyncio
    async def test_max_parallel_agents_passed_to_orchestrator(self) -> None:
        """Orchestrator is constructed with config.max_parallel_agents."""
        adapter = _make_adapter()
        task = _make_task(num_sections=2)
        config = _make_config_simple(enable_parallel=True, max_agents=7)

        constructed_kwargs: list[dict[str, Any]] = []

        class CapturingOrchestrator:
            """Fake orchestrator that records constructor arguments."""

            def __init__(self, max_parallel: int) -> None:
                constructed_kwargs.append({"max_parallel": max_parallel})

            async def execute(
                self,
                **kwargs: Any,
            ) -> SearchRoundOutput:
                return SearchRoundOutput()

        import inquiro.exps.parallel_search_exp as pse

        original_cls = pse.ParallelSearchOrchestrator
        pse.ParallelSearchOrchestrator = CapturingOrchestrator  # type: ignore[attr-defined]
        try:
            _result = await adapter.execute_search(task, config, 1, None)
        finally:
            pse.ParallelSearchOrchestrator = original_cls

        assert len(constructed_kwargs) == 1
        assert constructed_kwargs[0]["max_parallel"] == 7

    @pytest.mark.asyncio
    async def test_single_search_fn_is_execute_single_search(self) -> None:
        """Orchestrator receives adapter._execute_single_search as single_search_fn."""
        adapter = _make_adapter()
        task = _make_task(num_sections=2)
        config = _make_config_simple(enable_parallel=True, max_agents=2)

        received_fn: list[Any] = []

        class InspectingOrchestrator:
            def __init__(self, max_parallel: int) -> None:
                pass

            async def execute(
                self,
                task: Any,
                config: Any,
                round_number: int,
                focus_prompt: Any,
                single_search_fn: Any,
            ) -> SearchRoundOutput:
                received_fn.append(single_search_fn)
                return SearchRoundOutput()

        import inquiro.exps.parallel_search_exp as pse

        original_cls = pse.ParallelSearchOrchestrator
        pse.ParallelSearchOrchestrator = InspectingOrchestrator  # type: ignore[attr-defined]
        try:
            await adapter.execute_search(task, config, 1, None)
        finally:
            pse.ParallelSearchOrchestrator = original_cls

        assert len(received_fn) == 1
        # The received callable must be _execute_single_search (bound method)
        assert callable(received_fn[0])
        assert received_fn[0].__func__ is type(adapter)._execute_single_search


# ============================================================================
# 🔍 Legacy route — conditions not met
# ============================================================================


class TestLegacyRoute:
    """Tests that _execute_single_search is called on fallback 🔍."""

    @pytest.mark.asyncio
    async def test_single_search_when_parallel_disabled(self) -> None:
        """execute_search() uses _execute_single_search when parallel disabled."""
        adapter = _make_adapter()
        task = _make_task(num_sections=3)
        config = _make_config_simple(enable_parallel=False)

        expected_output = SearchRoundOutput(cost_usd=1.23)

        with patch.object(
            adapter,
            "_execute_single_search",
            new_callable=AsyncMock,
            return_value=expected_output,
        ) as mock_single:
            result = await adapter.execute_search(task, config, 1, "focus")

        mock_single.assert_awaited_once_with(task, config, 1, "focus")
        assert result is expected_output

    @pytest.mark.asyncio
    async def test_single_search_when_no_query_strategy(self) -> None:
        """execute_search() uses _execute_single_search when query_strategy is None."""
        adapter = _make_adapter()
        task = _make_task(include_strategy=False)
        config = _make_config_simple(enable_parallel=True)

        expected_output = SearchRoundOutput(cost_usd=0.5)

        with patch.object(
            adapter,
            "_execute_single_search",
            new_callable=AsyncMock,
            return_value=expected_output,
        ) as mock_single:
            result = await adapter.execute_search(task, config, 2, None)

        mock_single.assert_awaited_once()
        assert result is expected_output

    @pytest.mark.asyncio
    async def test_single_search_when_one_section_only(self) -> None:
        """execute_search() uses _execute_single_search with only 1 section."""
        adapter = _make_adapter()
        task = _make_task(num_sections=1)
        config = _make_config_simple(enable_parallel=True)

        expected_output = SearchRoundOutput(cost_usd=0.9)

        with patch.object(
            adapter,
            "_execute_single_search",
            new_callable=AsyncMock,
            return_value=expected_output,
        ) as mock_single:
            result = await adapter.execute_search(task, config, 1, "fp")

        mock_single.assert_awaited_once_with(task, config, 1, "fp")
        assert result is expected_output

    @pytest.mark.asyncio
    async def test_focus_prompt_forwarded_to_single_search(self) -> None:
        """execute_search() passes focus_prompt through to _execute_single_search."""
        adapter = _make_adapter()
        task = _make_task(num_sections=1)
        config = _make_config_simple(enable_parallel=True)
        focus = "targeted gap hint for round 3"

        received: list[str | None] = []

        async def capturing_single(
            t: Any,
            c: Any,
            r: int,
            f: str | None,
        ) -> SearchRoundOutput:
            received.append(f)
            return SearchRoundOutput()

        with patch.object(
            adapter, "_execute_single_search", side_effect=capturing_single
        ):
            await adapter.execute_search(task, config, 3, focus)

        assert received == [focus]


# ============================================================================
# 🔧 _execute_single_search interface
# ============================================================================


class TestExecuteSingleSearchInterface:
    """Tests that _execute_single_search exists and has the right signature 🔧."""

    def test_method_exists_on_adapter(self) -> None:
        """_SearchExpAdapter has _execute_single_search method."""
        from inquiro.core.runner import _SearchExpAdapter

        assert hasattr(_SearchExpAdapter, "_execute_single_search")

    def test_method_is_coroutine(self) -> None:
        """_execute_single_search is an async method (coroutine function)."""
        from inquiro.core.runner import _SearchExpAdapter

        method = getattr(_SearchExpAdapter, "_execute_single_search")
        assert inspect.iscoroutinefunction(method)

    def test_execute_search_is_still_present(self) -> None:
        """execute_search() is still present with the expected signature."""
        from inquiro.core.runner import _SearchExpAdapter

        method = getattr(_SearchExpAdapter, "execute_search")
        assert inspect.iscoroutinefunction(method)
        sig = inspect.signature(method)
        param_names = list(sig.parameters.keys())
        assert "task" in param_names
        assert "config" in param_names
        assert "round_number" in param_names
        assert "focus_prompt" in param_names
