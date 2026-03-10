"""Inquiro EvalTaskRunner — Service-level task orchestrator 🎯.

Manages shared resources (MCP connection pool, LLM provider pool) and
dispatches research/synthesis tasks to their respective Exp instances.

**Does NOT inherit BasePlayground** — lifecycle is managed directly by
the FastAPI application layer. This is a deliberate design decision:
Playground is for interactive/batch scenarios; EvalTaskRunner is for
long-lived service deployments with shared resource pools.

Architecture position:
    FastAPI Service Layer
        -> EvalTaskRunner (this class)
            -> DiscoveryLoop (multi-round discovery pipeline)
                -> SearchExp → SearchAgent (MCP search + adaptive reasoning)
                -> EvidencePipeline (zero-LLM cleaning)
                -> AnalysisExp (3-LLM parallel analysis)
                -> GapAnalysis (coverage + convergence)
                -> DiscoverySynthesisExp (optional final synthesis)
            -> SynthesisExp (per-task synthesis lifecycle)
                -> SynthesisAgent (LLM reasoning loop)

Key responsibilities:
    - Manage MCPConnectionPool and LLMProviderPool 🔧
    - Route /research requests through DiscoveryLoop 🔬
    - Route /synthesize requests to SynthesisExp 📊
    - Async task execution with cancellation support 🛑
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any

from inquiro.infrastructure.cancellation import CancellationToken
from inquiro.infrastructure.cost_tracker import CostTracker
from inquiro.infrastructure.degradation import DegradationManager
from inquiro.infrastructure.errors import classify_error
from inquiro.infrastructure.evidence_memory import EvidenceMemory
from inquiro.infrastructure.event_emitter import EventEmitter
from inquiro.infrastructure.logging_context import (
    clear_logging_context,
    set_logging_context,
)
from inquiro.infrastructure.metrics import MetricsCollector
from inquiro.infrastructure.result_cache import CompletedResultsCache
from inquiro.infrastructure.tool_effectiveness import ToolEffectivenessTracker
from inquiro.infrastructure.tool_routing import ToolRoutingStrategy
from inquiro.infrastructure.tracing import set_trace_context

if TYPE_CHECKING:
    from inquiro.core.discovery_loop import FeedbackProvider
    from inquiro.core.evidence_pool import SharedEvidencePool
    from inquiro.core.types import (
        DiscoveryConfig,
        DiscoveryResult,
        EvaluationResult,
        EvaluationTask,
        SynthesisResult,
        SynthesisTask,
    )
    from inquiro.infrastructure.llm_pool import LLMProviderPool
    from inquiro.infrastructure.mcp_pool import MCPConnectionPool
    from inquiro.infrastructure.skill_service import SkillService

logger = logging.getLogger(__name__)

# ⏰ Default wall-clock timeout (seconds) when not configured on the task
_DEFAULT_TIMEOUT_SECONDS: float = 1200.0

# 📊 Fallback decision confidence cap (prevents overconfidence w/o synthesis)
_CONFIDENCE_CAP: float = 0.95

# 📊 Cautious threshold ratio relative to coverage_threshold
_CAUTIOUS_THRESHOLD_RATIO: float = 0.625

# 📝 Prompt template for LLM group summarisation
_GROUP_SUMMARIZER_SYSTEM = (
    "You are a research evidence summarizer. Your task is to synthesize "
    "a group of evidence items that were excluded from primary analysis "
    "due to token limits. Produce a concise yet informative summary that "
    "captures the key findings, patterns, and any contradictions.\n\n"
    "Requirements:\n"
    "- Write 100-300 words.\n"
    "- Highlight the 3-5 most important findings or claims.\n"
    "- Note any contradictions or disagreements across sources.\n"
    "- Mention notable source URLs when particularly authoritative.\n"
    "- End with a brief statement of what this group covers overall.\n"
    "- Do NOT invent information beyond what is in the evidence.\n"
    "- Write in English, academic register."
)


class _LLMGroupSummarizer:
    """LLM-backed group summarizer using a fast model (e.g. Haiku) 📝.

    Implements the GroupSummarizer protocol.  Each call constructs a
    one-shot prompt with the evidence group and runs it through
    ``BaseLLM.query()`` in a thread pool.

    Attributes:
        _llm_pool: LLM provider pool for model access.
        _model: Model name to use (e.g. 'haiku').
    """

    def __init__(self, llm_pool: LLMProviderPool, model: str = "haiku") -> None:
        """Initialize LLMGroupSummarizer 🔧.

        Args:
            llm_pool: LLM provider pool for model access.
            model: Model name to request from the pool.
        """
        self._llm_pool = llm_pool
        self._model = model

    async def summarize(
        self,
        tag: str,
        items: list[Any],
        included_count: int,
    ) -> str:
        """Summarize a group of excluded evidence items via LLM 📝.

        Args:
            tag: Evidence tag group name.
            items: Excluded evidence items in this group.
            included_count: Items from this group already in primary selection.

        Returns:
            LLM-generated summary text.
        """
        from evomaster.utils.llm import Dialog, UserMessage

        prompt = self._build_prompt(tag, items, included_count)
        dialog = Dialog(
            system=_GROUP_SUMMARIZER_SYSTEM,
            messages=[UserMessage(content=prompt)],
        )

        llm = self._llm_pool.get_llm(self._model)
        response = await asyncio.to_thread(llm.query, dialog)
        content = getattr(response, "content", "") or ""
        return content.strip()

    def _build_prompt(
        self,
        tag: str,
        items: list[Any],
        included_count: int,
    ) -> str:
        """Build the user prompt for a single tag group 🔧.

        Args:
            tag: Evidence tag group name.
            items: Excluded evidence items.
            included_count: Items already in primary selection.

        Returns:
            Formatted prompt string with evidence items.
        """
        # 📏 Cap items to avoid excessive token usage
        max_items = 80
        capped = items[:max_items]
        overflow = len(items) - max_items if len(items) > max_items else 0

        lines = [
            f"Evidence group: {tag.upper()}",
            f"Items in primary selection: {included_count}",
            f"Items to summarize: {len(items)}"
            + (f" (showing first {max_items})" if overflow else ""),
            "",
            "--- Evidence Items ---",
        ]
        for i, ev in enumerate(capped, 1):
            url = getattr(ev, "url", "") or "no-url"
            summary = getattr(ev, "summary", "") or "no-summary"
            quality = getattr(ev, "quality_label", "") or ""
            source = getattr(ev, "source", "") or ""
            # 📏 Truncate long summaries to manage token usage
            if len(summary) > 400:
                summary = summary[:397] + "..."
            parts = [f"[{i}] {url}"]
            if source:
                parts.append(f"  Source: {source}")
            if quality:
                parts.append(f"  Quality: {quality}")
            parts.append(f"  Summary: {summary}")
            lines.append("\n".join(parts))

        if overflow:
            lines.append(f"\n... and {overflow} more items not shown.")

        lines.append(
            "\n--- Task ---\n"
            "Synthesize the above evidence items into a concise summary. "
            "Capture the most important findings and note any contradictions."
        )
        return "\n".join(lines)


class EvalTaskRunner:
    """Service-level task orchestrator 🎯.

    Manages shared resources (MCP pool, LLM pool) and dispatches
    research/synthesis tasks to their respective Exp instances.

    Does NOT inherit BasePlayground — lifecycle managed directly
    by the FastAPI service layer.

    Attributes:
        mcp_pool: Service-level MCP connection pool (shared across tasks).
        llm_pool: Service-level LLM provider pool (shared across tasks).
        _active_tasks: Mapping of task_id -> CancellationToken for
            currently running tasks. Protected by ``_tasks_lock``.
        _tasks_lock: Threading lock protecting ``_active_tasks`` against
            concurrent access from the event loop and worker threads.
        _circuit_breakers: Per-MCP-server circuit breakers.
    """

    def __init__(
        self,
        mcp_pool: MCPConnectionPool,
        llm_pool: LLMProviderPool,
        skill_service: SkillService | None = None,
        ensemble_defaults: dict[str, Any] | None = None,
    ) -> None:
        """Initialize with shared service-level resources 🔧.

        Args:
            mcp_pool: Service-level MCP connection pool. Initialized at
                application startup and shared across all concurrent tasks.
            llm_pool: Service-level LLM provider pool. Manages LLM client
                instances with connection reuse.
            skill_service: Optional SkillService instance. When provided,
                its registry is used for agent skills. Call
                ``skill_service.setup()`` before constructing the runner.
            ensemble_defaults: Deprecated. Kept for backward compatibility
                but ensemble execution has been removed.
        """
        self.mcp_pool = mcp_pool
        self.llm_pool = llm_pool
        self.skill_service = skill_service
        self._ensemble_defaults = ensemble_defaults or {}
        self._active_tasks: dict[str, CancellationToken] = {}
        # 🔒 threading.Lock (not asyncio.Lock) because _active_tasks is
        # accessed from both the event loop thread and worker threads
        # spawned by asyncio.to_thread().
        self._tasks_lock = threading.Lock()
        self._circuit_breakers: dict[str, Any] = {}
        self._completed_results = CompletedResultsCache(max_size=1000, ttl_seconds=3600)
        # 📝 Use module-level logger (see top of file)

        # 📊 Infrastructure: Metrics collection
        self._metrics = MetricsCollector()

        # 🛡️ Infrastructure: Graceful degradation
        self._degradation = DegradationManager()

        # 🧠 Infrastructure: Cross-task evidence memory
        self._evidence_memory = EvidenceMemory()

        # 🎯 Infrastructure: Tool effectiveness tracking
        self._tool_effectiveness = ToolEffectivenessTracker()

        # 🔧 Infrastructure: Tool routing strategy
        # Maps task domains to relevant MCP servers. When EvaluationTask
        # specifies a domain, only tools from matched servers are exposed
        # to the Agent prompt — reducing token usage. Falls back to all
        # tools when domain is None or unknown (backward compatible).
        self._tool_routing = ToolRoutingStrategy(domain_config={
            "clinical": [
                "bohrium", "biomcp", "pubmed", "clinicaltrials",
                "paper-search",
            ],
            "academic": ["bohrium", "biomcp", "pubmed", "paper-search"],
            "target": ["opentargets", "uniprot", "chembl"],
            "chemical": ["chembl", "pubchem"],
            "patent": ["patent_search"],
            "general": ["perplexity", "brave", "fetch"],
        })

        # 🔄 Infrastructure: Shared evidence pools for cross-task reuse
        self._shared_evidence_pools: dict[str, SharedEvidencePool] = {}
        self._pools_lock = threading.Lock()

    def get_or_create_evidence_pool(
        self,
        pool_id: str,
    ) -> SharedEvidencePool:
        """Get or create a SharedEvidencePool by identifier 🔄.

        Thread-safe lookup/creation of named evidence pools.
        Pools with the same ``pool_id`` return the same instance,
        enabling evidence sharing across tasks.

        Args:
            pool_id: Unique pool identifier (e.g., dimension_id).

        Returns:
            SharedEvidencePool instance for the given pool_id.
        """
        from inquiro.core.evidence_pool import SharedEvidencePool

        with self._pools_lock:
            if pool_id not in self._shared_evidence_pools:
                self._shared_evidence_pools[pool_id] = SharedEvidencePool()
                logger.info(
                    "🔄 Created new SharedEvidencePool: pool_id=%s",
                    pool_id,
                )
            return self._shared_evidence_pools[pool_id]

    @property
    def skill_registry(self) -> Any:
        """Return the SkillRegistry from SkillService (or None) 🎯.

        Backward-compatible property so that downstream code
        (SynthesisExp, DiscoveryLoop) can still access
        ``runner.skill_registry`` without changes.

        Returns:
            SkillRegistry instance, or None if no SkillService
            is configured or setup was not called.
        """
        if self.skill_service is not None:
            return self.skill_service.get_registry()
        return None

    @property
    def degradation_manager(self) -> DegradationManager:
        """Return the degradation manager for downstream use 🛡️.

        Exposes degradation state so that Exp layers and downstream
        consumers can inspect or react to LLM/MCP fallback status.

        Returns:
            DegradationManager instance.
        """
        return self._degradation

    async def submit_research(
        self,
        task: EvaluationTask,
        event_emitter: EventEmitter | None = None,
    ) -> EvaluationResult:
        """Submit and run an atomic research task 🔬.

        All tasks route through the DISCOVERY pipeline (unified entry).
        The pipeline intensity (quick/standard/thorough/discovery) controls
        the actual depth of research performed.

        Args:
            task: The evaluation task definition containing topic, rules,
                checklist, output_schema, agent_config, and cost_guard.
            event_emitter: Optional SSE event emitter for progress updates.
                If None, a no-op emitter is used.

        Returns:
            EvaluationResult with decision, evidence, and reasoning.
        """
        return await self._run_discovery(task, event_emitter)

    # ====================================================================
    # 🔄 DISCOVERY pipeline routing
    # ====================================================================

    async def _run_discovery(
        self,
        task: EvaluationTask,
        event_emitter: EventEmitter | None = None,
    ) -> EvaluationResult:
        """Run the DISCOVERY pipeline via DiscoveryLoop 🔄.

        Creates protocol-compatible adapters that wrap SearchExp and
        AnalysisExp, constructs a DiscoveryLoop, runs all rounds,
        optionally runs synthesis, and converts the result to
        EvaluationResult for API compatibility.

        Args:
            task: Evaluation task for the discovery pipeline.
            event_emitter: Optional SSE event emitter for progress updates.

        Returns:
            EvaluationResult populated with discovery pipeline outputs.
        """
        task_id = task.task_id

        # 🔍 Set trace context for log correlation
        set_trace_context(trace_id=task_id)
        set_logging_context(task_id=task_id)

        logger.info(
            "🔄 Submitting DISCOVERY pipeline task: %s",
            task_id,
        )

        # 🛑 Create cancellation token
        token = CancellationToken()
        self._register_active_task(task_id, token)

        # 📡 Create or reuse event emitter
        emitter = event_emitter or self._create_event_emitter()

        # 📊 Subscribe metrics collector to task events
        self._metrics.subscribe_to_emitter(emitter)

        try:
            # 🔧 Parse DiscoveryConfig from task
            config = self._parse_discovery_config(task)

            # 🔄 Initialize trajectory feedback if available
            feedback_provider = self._init_feedback_provider(task)

            # 🧬 Initialize evolution provider if profile available
            evolution_provider = await self._init_evolution_provider(task)

            # 🏗️ Create loop + adapters + wiring
            loop, evidence_pool = self._create_discovery_loop(
                task=task,
                config=config,
                emitter=emitter,
                token=token,
                feedback_provider=feedback_provider,
                evolution_provider=evolution_provider,
            )

            # ⏰ Resolve total timeout from config
            total_timeout = float(config.timeout_total)

            # 📦 Resolve KB seeded evidence (pass-through from upper layer)
            raw_seeded = task.seeded_evidence
            seeded_gap_hints = task.seeded_gap_hints
            seeded_evidence: list | None = None
            if raw_seeded:
                from inquiro.core.types import Evidence
                seeded_evidence = [
                    e if isinstance(e, Evidence)
                    else Evidence.model_validate(e)
                    for e in raw_seeded
                ]
                logger.info(
                    "📦 KB seeded evidence: %d items for task %s",
                    len(seeded_evidence),
                    task_id,
                )

            # 🚀 Run the discovery loop with timeout
            async with asyncio.timeout(total_timeout):
                discovery_result = await loop.run(
                    task,
                    config,
                    shared_evidence_pool=evidence_pool,
                    seeded_evidence=seeded_evidence,
                    seeded_gap_hints=seeded_gap_hints,
                )

            # 🧬 Post-discovery: synthesis + evolution hooks + trajectory
            synthesis_result = await self._run_post_discovery_synthesis(
                task=task,
                config=config,
                discovery_result=discovery_result,
                loop=loop,
                emitter=emitter,
                token=token,
                evolution_provider=evolution_provider,
            )

            # 🔄 Convert DiscoveryResult → EvaluationResult
            result = self._discovery_to_evaluation_result(
                discovery_result=discovery_result,
                task=task,
                synthesis_result=synthesis_result,
                config=config,
            )

            # 🧠 Store evidence in cross-task memory
            if result.evidence_index:
                try:
                    self._evidence_memory.store(
                        task_id=task_id,
                        evidence_list=[
                            {
                                "id": ev.id,
                                "source": ev.source,
                                "url": getattr(ev, "url", None) or "",
                                "summary": ev.summary,
                                "quality_label": (
                                    getattr(ev, "quality_label", None)
                                    or "medium"
                                ),
                            }
                            for ev in result.evidence_index
                        ],
                    )
                except Exception as exc:
                    logger.warning(
                        "⚠️ Failed to store evidence in memory: %s", exc,
                    )

            # 📝 Store completed result
            self._completed_results.put(
                task_id,
                {
                    "status": "completed",
                    "result": result.model_dump(),
                },
            )

            logger.info(
                "✅ DISCOVERY pipeline completed for task %s: "
                "rounds=%d coverage=%.2f cost=$%.2f",
                task_id,
                discovery_result.total_rounds,
                discovery_result.final_coverage,
                discovery_result.total_cost_usd,
            )

            return result

        except TimeoutError:
            # ⏰ Total timeout exceeded
            token.cancel(
                reason="Discovery pipeline total timeout exceeded",
            )
            logger.warning(
                "⏰ DISCOVERY task %s timed out",
                task_id,
            )
            partial = self._build_discovery_timeout_result(task_id)
            self._completed_results.put(
                task_id,
                {
                    "status": "timeout",
                    "result": partial.model_dump(),
                },
            )
            return partial

        except Exception as exc:
            # ❌ Pipeline failure
            error_type = classify_error(exc)
            if error_type == "transient":
                self._degradation.suggest_llm_fallback(
                    reason=(f"Transient error in discovery task {task_id}: {exc}"),
                )

            logger.error(
                "❌ DISCOVERY task %s failed: %s",
                task_id,
                exc,
                exc_info=True,
            )
            self._completed_results.put(
                task_id,
                {"status": "failed", "error": str(exc)},
            )
            raise

        finally:
            self._unregister_active_task(task_id)
            clear_logging_context()

    def _init_feedback_provider(
        self,
        task: EvaluationTask,
    ) -> FeedbackProvider | None:
        """Initialize trajectory feedback provider if available 🔄.

        Args:
            task: Evaluation task with optional trajectory_dir.

        Returns:
            TrajectoryFeedbackProvider or None on failure/absence.
        """
        if not task.trajectory_dir:
            return None
        try:
            from inquiro.core.trajectory.feedback import (
                TrajectoryFeedbackProvider,
            )

            return TrajectoryFeedbackProvider(task.trajectory_dir)
        except Exception as exc:
            logger.warning(
                "⚠️ Trajectory feedback init failed: %s",
                exc,
            )
            return None

    async def _init_evolution_provider(
        self,
        task: EvaluationTask,
    ) -> Any | None:
        """Initialize evolution provider from task profile 🧬.

        Creates a UnifiedEvolutionProvider via the factory function,
        with mechanisms selected by the task's intensity level.

        Args:
            task: Evaluation task with optional evolution_profile dict.

        Returns:
            Initialized UnifiedEvolutionProvider, or None if
            profile is absent or initialization fails.
        """
        if not task.evolution_profile:
            return None
        try:
            from targetmaster.evolution.discovery_provider import (
                create_evolution_provider,
            )
            from targetmaster.evolution.profile import (
                EvolutionProfile,
            )
            from inquiro.evolution.store_factory import get_store

            profile = EvolutionProfile.model_validate(
                task.evolution_profile,
            )
            store = await get_store()

            async def _llm_fn(prompt: str) -> str:
                """Lightweight LLM call for extraction 🧬."""
                extraction_llm = self._get_llm(
                    getattr(profile, "extraction_model", None) or None,
                )
                messages = [
                    {
                        "role": "system",
                        "content": "You are a concise extraction assistant.",
                    },
                    {"role": "user", "content": prompt},
                ]
                resp = await asyncio.to_thread(
                    extraction_llm._call, messages,
                )
                return str(resp.content)

            # 🎛️ Resolve intensity level from task's discovery_config
            intensity_level = "STANDARD"
            if hasattr(task, "discovery_config") and task.discovery_config:
                dc = task.discovery_config
                if isinstance(dc, dict):
                    intensity_level = dc.get(
                        "intensity_level", "STANDARD",
                    )
                elif hasattr(dc, "intensity_level"):
                    intensity_level = str(
                        getattr(dc, "intensity_level", "STANDARD"),
                    )

            context_tags = getattr(task, "context_tags", []) or []
            provider = create_evolution_provider(
                store=store,
                profile=profile,
                llm_fn=_llm_fn,
                context_tags=context_tags,
                sub_item_id=task.task_id,
                task=task,
                intensity_level=intensity_level,
            )
            await provider.prepare_enrichment()
            logger.info(
                "🧬 Evolution provider created for task %s "
                "(intensity=%s)",
                task.task_id,
                intensity_level,
            )
            return provider
        except Exception as exc:
            logger.warning(
                "⚠️ Evolution provider init failed: %s",
                exc,
            )
            return None

    def _create_discovery_loop(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        emitter: EventEmitter,
        token: CancellationToken,
        feedback_provider: FeedbackProvider | None,
        evolution_provider: Any | None,
    ) -> tuple[Any, Any]:
        """Create DiscoveryLoop with adapters and wiring 🏗️.

        Constructs protocol-compatible adapters, the DiscoveryLoop,
        wires circular references, and resolves the shared evidence
        pool.

        Args:
            task: Evaluation task.
            config: Parsed DiscoveryConfig.
            emitter: Event emitter for progress updates.
            token: Cancellation token for the pipeline.
            feedback_provider: Optional trajectory feedback provider.
            evolution_provider: Optional evolution enrichment provider.

        Returns:
            Tuple of (DiscoveryLoop, shared_evidence_pool_or_None).
        """
        from inquiro.core.discovery_loop import (
            DiscoveryLoop,
            MockFocusPromptGenerator,
        )
        from inquiro.core.evidence_pipeline import EvidencePipeline
        from inquiro.core.gap_analysis import (
            CoverageJudge,
            GapAnalysis,
            MockCoverageJudge,
        )
        from inquiro.core.metadata_enricher import MetadataEnricher

        # 🏗️ Create protocol adapters
        search_adapter = _SearchExpAdapter(
            runner=self,
            event_emitter=emitter,
            cancellation_token=token,
            feedback_provider=feedback_provider,
        )
        analysis_adapter = _AnalysisExpAdapter(
            runner=self,
            event_emitter=emitter,
            cancellation_token=token,
        )

        # 📝 Create group summarizer if configured
        group_summarizer = None
        if config.condenser_summarizer_model:
            try:
                group_summarizer = _LLMGroupSummarizer(
                    llm_pool=self.llm_pool,
                    model=config.condenser_summarizer_model,
                )
                logger.info(
                    "📝 LLM group summarizer enabled (model=%s)",
                    config.condenser_summarizer_model,
                )
            except Exception as exc:
                logger.warning(
                    "⚠️ Failed to create LLM group summarizer: %s — "
                    "falling back to template summaries.",
                    exc,
                )

        # 🤖 Create coverage judge — LLM when model configured, else mock
        from inquiro.core.llm_coverage_judge import LLMCoverageJudge

        coverage_judge: CoverageJudge
        if config.coverage_judge_model:
            try:
                coverage_judge = LLMCoverageJudge(
                    llm_pool=self.llm_pool,
                    model=config.coverage_judge_model,
                )
                logger.info(
                    "🤖 LLM coverage judge enabled (model=%s, mode=%s)",
                    config.coverage_judge_model,
                    config.coverage_judge_mode,
                )
            except Exception as exc:
                logger.warning(
                    "⚠️ Failed to create LLM coverage judge: %s — "
                    "falling back to MockCoverageJudge.",
                    exc,
                )
                coverage_judge = MockCoverageJudge()
        else:
            coverage_judge = MockCoverageJudge()

        # 🏷️ Create metadata enricher if LLM pool available
        metadata_enricher = None
        if self.llm_pool:
            try:
                metadata_enricher = MetadataEnricher(
                    llm_pool=self.llm_pool,
                )
                logger.info("🏷️ MetadataEnricher enabled")
            except Exception:
                logger.warning(
                    "⚠️ Failed to create MetadataEnricher — "
                    "metadata extraction disabled"
                )

        # 🏗️ Create DiscoveryLoop with injected components
        loop = DiscoveryLoop(
            search_executor=search_adapter,
            analysis_executor=analysis_adapter,
            gap_analysis=GapAnalysis(
                coverage_judge=coverage_judge,
                coverage_judge_mode=config.coverage_judge_mode,
            ),
            evidence_pipeline=EvidencePipeline(),
            focus_generator=MockFocusPromptGenerator(),
            trajectory_dir=task.trajectory_dir,
            feedback_provider=feedback_provider,
            evolution_provider=evolution_provider,
            group_summarizer=group_summarizer,
            metadata_enricher=metadata_enricher,
        )

        # 🧬 Wire adapters to loop for enrichment access
        search_adapter._discovery_loop = loop
        analysis_adapter._discovery_loop = loop

        # 🔄 Resolve shared evidence pool (if evidence_pool_id set)
        evidence_pool = None
        if task.evidence_pool_id:
            evidence_pool = self.get_or_create_evidence_pool(
                task.evidence_pool_id,
            )
            logger.info(
                "🔄 Using shared evidence pool '%s' for task %s (pool_size=%d)",
                task.evidence_pool_id,
                task.task_id,
                evidence_pool.size,
            )

        return loop, evidence_pool

    async def _run_post_discovery_synthesis(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        discovery_result: DiscoveryResult,
        loop: Any,
        emitter: EventEmitter,
        token: CancellationToken,
        evolution_provider: Any | None,
    ) -> Any:
        """Run optional synthesis after discovery and handle hooks 🧬.

        Performs synthesis if analysis_models configured, invokes the
        evolution on_synthesis_complete hook, and writes synthesis
        cost to the trajectory JSONL.

        Args:
            task: Original evaluation task.
            config: Discovery configuration.
            discovery_result: Result from DiscoveryLoop.run().
            loop: The DiscoveryLoop instance (for trajectory writer).
            emitter: Event emitter for progress updates.
            token: Cancellation token.
            evolution_provider: Optional evolution provider for hooks.

        Returns:
            SynthesisResult or None.
        """
        task_id = task.task_id
        synthesis_result = None

        if config.enable_synthesis:
            # 🧬 Get synthesis enrichment from evolution provider
            synthesis_enrichment = None
            if evolution_provider:
                try:
                    synthesis_enrichment = evolution_provider.get_synthesis_enrichment()
                except Exception:
                    logger.warning(
                        "⚠️ Evolution synthesis enrichment failed",
                        exc_info=True,
                    )

            synthesis_result = await self._run_discovery_synthesis(
                task=task,
                config=config,
                discovery_result=discovery_result,
                event_emitter=emitter,
                cancellation_token=token,
                evolution_enrichment=synthesis_enrichment,
            )

            # 🧬 Call on_synthesis_complete hook
            if evolution_provider and synthesis_result:
                await self._invoke_evolution_synthesis_hook(
                    evolution_provider,
                    synthesis_result,
                    discovery_result,
                    task_id,
                )

        # 📊 Write synthesis cost to trajectory JSONL
        if synthesis_result is not None and loop.trajectory_writer is not None:
            from inquiro.core.trajectory.models import SynthesisRecord

            synth_cost = float(
                getattr(synthesis_result, "cost_usd", 0.0) or 0.0,
            )
            loop.trajectory_writer.write_synthesis(
                SynthesisRecord(cost_usd=synth_cost),
            )
            logger.debug(
                "📊 Wrote synthesis cost $%.4f to trajectory for task %s",
                synth_cost,
                task_id,
            )

        return synthesis_result

    async def _invoke_evolution_synthesis_hook(
        self,
        evolution_provider: Any,
        synthesis_result: Any,
        discovery_result: DiscoveryResult,
        task_id: str,
    ) -> None:
        """Invoke the evolution on_synthesis_complete hook 🧬.

        Args:
            evolution_provider: Initialized evolution provider.
            synthesis_result: Synthesis output.
            discovery_result: Discovery loop output.
            task_id: Task identifier for logging.
        """
        try:
            from inquiro.core.types import RoundMetrics

            final_metrics = RoundMetrics(
                evidence_count=len(discovery_result.evidence),
                new_evidence_count=0,
                coverage=discovery_result.final_coverage,
                cost_usd=discovery_result.total_cost_usd,
                round_index=discovery_result.total_rounds,
            )
            synth_record = getattr(
                synthesis_result,
                "synthesis_record",
                synthesis_result,
            )
            await evolution_provider.on_synthesis_complete(
                synth_record,
                final_metrics,
            )
        except Exception:
            logger.warning(
                "⚠️ Evolution on_synthesis_complete failed for task %s",
                task_id,
                exc_info=True,
            )

    def _parse_discovery_config(self, task: EvaluationTask) -> DiscoveryConfig:
        """Parse DiscoveryConfig from task, using INTENSITY_PRESETS as base 🔧.

        Always returns a valid DiscoveryConfig. When task.discovery_config
        is provided, uses it as override on top of the intensity preset.
        When absent, uses "standard" preset defaults.

        Injects ``analysis_models`` from the LLM pool when the list is
        empty, picking the first N models where N = ``analysis_model_count``.

        Args:
            task: Evaluation task with optional discovery_config.

        Returns:
            Fully resolved DiscoveryConfig.
        """
        from inquiro.core.types import INTENSITY_PRESETS, DiscoveryConfig

        if task.discovery_config:
            # Task has explicit config — resolve intensity preset as base
            raw = dict(task.discovery_config)
            intensity = raw.get("intensity", "standard")
            base = dict(INTENSITY_PRESETS.get(intensity, INTENSITY_PRESETS["standard"]))
            base.update(raw)  # Task overrides take precedence
            config = DiscoveryConfig.model_validate(base)
        else:
            # No explicit config — use standard defaults
            config = DiscoveryConfig.model_validate(INTENSITY_PRESETS["standard"])

        # 🔧 Inject analysis_models from LLM pool when empty
        llm_pool = getattr(self, "llm_pool", None)
        if not config.analysis_models and llm_pool:
            available = llm_pool.get_available_models()
            count = config.analysis_model_count
            config.analysis_models = available[:count]
            logger.info(
                "🔧 Injected %d analysis models from LLM pool: %s",
                len(config.analysis_models),
                config.analysis_models,
            )

        return config

    async def _run_discovery_synthesis(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        discovery_result: DiscoveryResult,
        event_emitter: EventEmitter | None = None,
        cancellation_token: CancellationToken | None = None,
        evolution_enrichment: str | None = None,
    ) -> Any:
        """Run optional synthesis after discovery loop converges 🧬.

        Creates a DiscoverySynthesisExp and runs it with all
        accumulated evidence and claims from the discovery result.

        Args:
            task: Original evaluation task.
            config: Discovery configuration with analysis_models.
            discovery_result: Result from the DiscoveryLoop.
            event_emitter: SSE event emitter.
            cancellation_token: Cancellation signal.
            evolution_enrichment: Optional enrichment text to inject
                into synthesis system prompt.

        Returns:
            SynthesisResult from DiscoverySynthesisExp, or None on
            failure.
        """
        from inquiro.exps.discovery_synthesis_exp import (
            DiscoverySynthesisExp,
        )

        task_id = task.task_id
        logger.info(
            "🧬 Running discovery synthesis for task %s",
            task_id,
        )

        try:
            # 🔧 Get default LLM
            llm = self._get_llm(task.agent_config.model)

            # 💰 Create cost tracker
            cost_tracker = self._create_cost_tracker(
                max_per_task=task.cost_guard.max_cost_per_task,
            )

            # 🏗️ Create synthesis exp
            synthesis_exp = DiscoverySynthesisExp(
                task=task,
                llm=llm,
                llm_pool=self.llm_pool,
                cost_tracker=cost_tracker,
                event_emitter=event_emitter,
                cancellation_token=cancellation_token,
                evolution_enrichment=evolution_enrichment,
            )

            # 🔄 Convert CleanedEvidence → Evidence for synthesis
            all_evidence = EvalTaskRunner._map_evidence_list(
                discovery_result.evidence,
            )

            # 🗜️ Guard synthesis against token overflow
            # Estimate token budget: each evidence item contributes
            # ~summary_chars/3.5 + ~80 metadata tokens to the prompt.
            # Synthesis models typically have 128K-200K context; reserve
            # ~100K tokens for evidence, ~28K for system/checklist/claims.
            _SYNTH_TOKEN_BUDGET = 100_000
            _TOKENS_PER_META = 80  # URL, query, quality, round fields

            estimated_tokens = sum(
                len(ev.summary or "") / 3.5 + _TOKENS_PER_META
                for ev in all_evidence
            )

            if estimated_tokens > _SYNTH_TOKEN_BUDGET or len(all_evidence) > 80:
                from inquiro.core.evidence_condenser import (
                    CondenserConfig,
                    EvidenceCondenser,
                )

                # Target item count based on budget: aim for ~100K tokens
                avg_tokens_per_item = (
                    estimated_tokens / len(all_evidence)
                    if all_evidence
                    else 500
                )
                target_items = max(
                    30,
                    min(80, int(_SYNTH_TOKEN_BUDGET / avg_tokens_per_item)),
                )

                synth_condenser = EvidenceCondenser(
                    CondenserConfig(
                        tier1_threshold=target_items,
                        tier1_target=target_items,
                    )
                )
                checklist_strs: list[str] = []
                if task.checklist and hasattr(task.checklist, "required"):
                    checklist_strs = [
                        item.description
                        for item in task.checklist.required
                        if hasattr(item, "description")
                    ]
                condensed = synth_condenser.condense(
                    all_evidence, checklist_strs
                )
                logger.info(
                    "🗜️ Synthesis condenser: %d → %d items "
                    "(est. tokens: %.0f → %.0f, target=%d)",
                    condensed.meta.original_count,
                    condensed.meta.condensed_count,
                    estimated_tokens,
                    sum(
                        len(ev.summary or "") / 3.5 + _TOKENS_PER_META
                        for ev in condensed.evidence
                    ),
                    target_items,
                )
                all_evidence = condensed.evidence

            # 🔄 Convert claim dicts → ReasoningClaim for synthesis
            all_claims = EvalTaskRunner._map_claim_list(
                discovery_result.claims,
            )

            # 🔑 Deduplicate claims by content to avoid repeating rounds
            seen_claim_keys: set[str] = set()
            deduped_claims: list[Any] = []
            for claim in all_claims:
                claim_key = (
                    claim.claim
                    if hasattr(claim, "claim")
                    else str(claim)
                )
                if claim_key not in seen_claim_keys:
                    seen_claim_keys.add(claim_key)
                    deduped_claims.append(claim)
            if len(deduped_claims) < len(all_claims):
                logger.info(
                    "🔑 Claim dedup: %d → %d (removed %d duplicates)",
                    len(all_claims),
                    len(deduped_claims),
                    len(all_claims) - len(deduped_claims),
                )
            all_claims = deduped_claims

            # 📊 Build round summaries for synthesis context
            round_summaries = [
                {
                    "round_number": rs.round_number,
                    "evidence_count": rs.cleaned_evidence_count,
                    "coverage_ratio": rs.coverage_ratio,
                }
                for rs in discovery_result.round_summaries
            ]

            # 🚀 Run synthesis
            result = await synthesis_exp.run_synthesis(
                task=task,
                config=config,
                all_evidence=all_evidence,
                all_claims=all_claims,
                coverage_ratio=discovery_result.final_coverage,
                round_summaries=round_summaries,
            )

            logger.info(
                "✅ Discovery synthesis completed for task %s: "
                "decision=%s confidence=%.2f",
                task_id,
                result.consensus_decision,
                result.evaluation_result.confidence
                if result.evaluation_result
                else 0.0,
            )
            return result

        except Exception as exc:
            logger.warning(
                "⚠️ Discovery synthesis failed for task %s: %s "
                "(proceeding without synthesis)",
                task_id,
                exc,
                exc_info=True,
            )
            return None

    @staticmethod
    def _map_evidence_list(
        cleaned_evidence: list[Any],
    ) -> list[Any]:
        """Map CleanedEvidence items to Evidence objects 🔄.

        Args:
            cleaned_evidence: CleanedEvidence objects from DiscoveryResult.

        Returns:
            List of Evidence objects for API compatibility.
        """
        from inquiro.core.types import Evidence

        return [
            Evidence(
                id=ce.id,
                source=ce.mcp_server,
                url=ce.url,
                query=ce.source_query,
                summary=ce.summary,
                evidence_tag=ce.tag.value if ce.tag else None,
                doi=getattr(ce, "doi", None),
                clinical_trial_id=getattr(ce, "clinical_trial_id", None),
            )
            for ce in cleaned_evidence
        ]

    @staticmethod
    def _map_claim_list(
        claim_dicts: list[dict[str, Any]],
    ) -> list[Any]:
        """Map claim dictionaries to ReasoningClaim objects 🔄.

        Args:
            claim_dicts: Raw claim dictionaries from DiscoveryResult.

        Returns:
            List of ReasoningClaim objects for API compatibility.
        """
        from inquiro.core.types import EvidenceStrength, ReasoningClaim

        claims: list[ReasoningClaim] = []
        for claim_dict in claim_dicts:
            strength_raw = claim_dict.get("strength", "moderate")
            try:
                strength = EvidenceStrength(strength_raw)
            except ValueError:
                strength = EvidenceStrength.MODERATE
            claims.append(
                ReasoningClaim(
                    claim=claim_dict.get("claim", ""),
                    evidence_ids=claim_dict.get("evidence_ids", []),
                    strength=strength,
                )
            )
        return claims

    def _discovery_to_evaluation_result(
        self,
        discovery_result: DiscoveryResult,
        task: EvaluationTask,
        synthesis_result: Any = None,
        config: DiscoveryConfig | None = None,
    ) -> EvaluationResult:
        """Convert DiscoveryResult to EvaluationResult for API compat 🔄.

        Maps CleanedEvidence → Evidence, claim dicts → ReasoningClaim,
        and populates discovery-specific fields on EvaluationResult.

        When synthesis_result is provided, uses its decision and
        confidence instead of deriving them from the raw discovery.

        Args:
            discovery_result: Output from DiscoveryLoop.run().
            task: Original evaluation task.
            synthesis_result: Optional synthesis output for enrichment.
            config: Optional DiscoveryConfig for threshold lookup.

        Returns:
            EvaluationResult populated with discovery pipeline data.
        """
        from inquiro.core.types import (
            ChecklistCoverage,
            EvaluationResult,
            ResearchContext,
        )

        # 🔄 Map CleanedEvidence → Evidence
        evidence_index = EvalTaskRunner._map_evidence_list(
            discovery_result.evidence,
        )

        # 🔄 Map claim dicts → ReasoningClaim
        reasoning = EvalTaskRunner._map_claim_list(
            discovery_result.claims,
        )

        # 📊 Determine decision, confidence, and reasoning
        decision, confidence, reasoning = self._determine_decision(
            discovery_result=discovery_result,
            synthesis_result=synthesis_result,
            config=config,
            fallback_reasoning=reasoning,
        )

        # 📋 Build checklist coverage from last gap report
        checklist_coverage = ChecklistCoverage()
        gaps_remaining: list[str] = []
        if discovery_result.gap_reports:
            last_gap = discovery_result.gap_reports[-1]
            checklist_coverage = ChecklistCoverage(
                required_covered=last_gap.covered_items,
                required_missing=last_gap.uncovered_items,
            )
            gaps_remaining = list(last_gap.uncovered_items)

        # 📋 Build ResearchContext from gap reports
        coverage_map: dict[str, bool] = {}
        if discovery_result.gap_reports:
            last_gap = discovery_result.gap_reports[-1]
            for item_id in getattr(last_gap, 'covered_items', []):
                coverage_map[item_id] = True
            for item_id in getattr(last_gap, 'uncovered_items', []):
                coverage_map[item_id] = False

        research_context = ResearchContext(
            coverage_map=coverage_map,
            information_gaps=gaps_remaining,
            conflicting_evidence=[],
            search_strategies_used=[],
            tool_effectiveness={},
        )

        # 📊 Build metadata and compute total cost
        metadata = self._build_discovery_metadata(
            discovery_result,
            synthesis_result,
        )
        total_cost = discovery_result.total_cost_usd
        if synthesis_result is not None and hasattr(synthesis_result, "cost_usd"):
            total_cost += synthesis_result.cost_usd

        return EvaluationResult(
            task_id=task.task_id,
            decision=decision,
            confidence=confidence,
            reasoning=reasoning,
            evidence_index=evidence_index,
            search_rounds=discovery_result.total_rounds,
            round_logs=[],
            checklist_coverage=checklist_coverage,
            gaps_remaining=gaps_remaining,
            doubts_remaining=[],
            cost=total_cost,
            pipeline_mode="discovery",
            confidence_source=(
                "consensus_confidence" if synthesis_result
                else "coverage_ratio"
            ),
            discovery_rounds=discovery_result.total_rounds,
            discovery_coverage=discovery_result.final_coverage,
            research_context=research_context,
            metadata=metadata,
        )

    def _determine_decision(
        self,
        discovery_result: DiscoveryResult,
        synthesis_result: Any,
        config: DiscoveryConfig | None,
        fallback_reasoning: list[Any],
    ) -> tuple[Any, float, list[Any]]:
        """Determine final decision from synthesis or coverage 📊.

        If synthesis is available, uses its decision and confidence.
        Otherwise falls back to coverage-threshold-based derivation.

        Args:
            discovery_result: Output from DiscoveryLoop.
            synthesis_result: Optional synthesis output.
            config: Optional DiscoveryConfig for threshold lookup.
            fallback_reasoning: Default reasoning claims from
                discovery claims mapping.

        Returns:
            Tuple of (decision, confidence, reasoning).
        """
        from inquiro.core.types import Decision

        if (
            synthesis_result is not None
            and hasattr(synthesis_result, "evaluation_result")
            and synthesis_result.evaluation_result is not None
        ):
            synth_eval = synthesis_result.evaluation_result
            reasoning = (
                synth_eval.reasoning if synth_eval.reasoning else fallback_reasoning
            )
            return synth_eval.decision, synth_eval.confidence, reasoning

        # 🔄 Fallback: derive from coverage using config thresholds
        coverage_threshold = config.coverage_threshold if config else 0.80
        cautious_threshold = coverage_threshold * _CAUTIOUS_THRESHOLD_RATIO
        cov = discovery_result.final_coverage

        if cov >= coverage_threshold:
            return (
                Decision.POSITIVE,
                min(cov, _CONFIDENCE_CAP),
                fallback_reasoning,
            )
        if cov >= cautious_threshold:
            return Decision.CAUTIOUS, cov, fallback_reasoning
        return Decision.NEGATIVE, cov, fallback_reasoning

    @staticmethod
    def _build_discovery_metadata(
        discovery_result: DiscoveryResult,
        synthesis_result: Any,
    ) -> dict[str, Any]:
        """Assemble discovery metadata dict for EvaluationResult 📊.

        Args:
            discovery_result: Output from DiscoveryLoop.
            synthesis_result: Optional synthesis output.

        Returns:
            Metadata dict with round summaries, timeline, and
            optional synthesis info.
        """
        metadata: dict[str, Any] = {
            "discovery": True,
            "total_rounds": discovery_result.total_rounds,
            "termination_reason": (discovery_result.termination_reason),
            "trajectory_id": discovery_result.trajectory_id,
            "round_summaries": [
                rs.model_dump() for rs in discovery_result.round_summaries
            ],
        }
        # 🧬 Per-round coverage timeline for evolution R1CR
        if discovery_result.gap_reports:
            metadata["coverage_timeline"] = [
                {
                    "round_num": gr.round_number,
                    "covered_items": list(gr.covered_items),
                    "uncovered_items": list(
                        gr.uncovered_items,
                    ),
                    "coverage_ratio": gr.coverage_ratio,
                }
                for gr in discovery_result.gap_reports
            ]
        if synthesis_result is not None:
            metadata["synthesis_decision"] = getattr(
                synthesis_result,
                "consensus_decision",
                None,
            )
            metadata["synthesis_ratio"] = getattr(
                synthesis_result,
                "consensus_ratio",
                None,
            )
        return metadata

    @staticmethod
    def _build_discovery_timeout_result(
        task_id: str,
    ) -> EvaluationResult:
        """Build a partial result for a timed-out DISCOVERY task ⏰.

        Args:
            task_id: The unique task identifier.

        Returns:
            EvaluationResult stub marked as timed-out with discovery
            pipeline metadata.
        """
        from inquiro.core.types import (
            ChecklistCoverage,
            Decision,
            EvaluationResult,
            EvidenceStrength,
            ReasoningClaim,
        )

        return EvaluationResult(
            task_id=task_id,
            decision=Decision.NEGATIVE,
            confidence=0.0,
            reasoning=[
                ReasoningClaim(
                    claim=(
                        "Discovery pipeline timed out before completing all rounds."
                    ),
                    evidence_ids=[],
                    strength=EvidenceStrength.WEAK,
                ),
            ],
            evidence_index=[],
            search_rounds=0,
            round_logs=[],
            checklist_coverage=ChecklistCoverage(),
            gaps_remaining=["Discovery pipeline timeout"],
            doubts_remaining=[],
            cost=0.0,
            pipeline_mode="discovery",
            metadata={"discovery": True, "timeout": True},
        )

    async def submit_synthesis(
        self,
        task: SynthesisTask,
        event_emitter: EventEmitter | None = None,
    ) -> SynthesisResult:
        """Submit and run a synthesis task 📊.

        Creates a SynthesisExp with a back-reference to this runner
        (for internal research calls), runs the synthesis agent in a
        thread pool, and returns the result.

        Args:
            task: The synthesis task definition containing topic,
                input_reports, synthesis_rules, output_schema, and config.
            event_emitter: Optional SSE event emitter for progress updates.
                If None, a no-op emitter is used.

        Returns:
            SynthesisResult with synthesized decision and cross-references.
        """
        from inquiro.exps.synthesis_exp import SynthesisExp

        task_id = task.task_id

        # 🔍 Set trace context for log correlation
        set_trace_context(trace_id=task_id)

        # 📝 Set logging context for task-level log fields
        set_logging_context(task_id=task_id)

        logger.info("📊 Submitting synthesis task: %s", task_id)

        # 🛑 Create cancellation token
        token = CancellationToken()
        self._register_active_task(task_id, token)

        # 📡 Create or reuse event emitter
        emitter = event_emitter or self._create_event_emitter()

        # 📊 Subscribe metrics collector to task events
        self._metrics.subscribe_to_emitter(emitter)

        # 💰 Create cost tracker from task config
        cost_tracker = self._create_cost_tracker(
            max_per_task=task.cost_guard.max_cost_per_task,
        )

        # 🔧 Get LLM instance
        llm = self._get_llm(task.agent_config.model)

        # 🏗️ Create SynthesisExp with back-reference to self
        exp = SynthesisExp(
            task=task,
            llm=llm,
            task_runner=self,
            quality_gate_config=task.quality_gate,
            cost_tracker=cost_tracker,
            event_emitter=emitter,
            cancellation_token=token,
            skill_registry=self.skill_registry,
        )

        # ⏰ Resolve wall-clock timeout
        timeout_secs = self._resolve_timeout(task.cost_guard)

        try:
            # 🔄 Run synchronous Exp in thread pool with timeout
            async with asyncio.timeout(timeout_secs):
                result = await asyncio.to_thread(exp.run_sync)

            # 📝 Store completed result
            self._completed_results.put(
                task_id,
                {"status": "completed", "result": result.model_dump()},
            )

            return result

        except TimeoutError:
            # ⏰ Wall-clock timeout — suggest degradation
            self._degradation.suggest_llm_fallback(
                reason=(f"Synthesis task {task_id} timed out after {timeout_secs}s"),
            )

            # ⏰ Cancel cooperatively
            token.cancel(
                reason=(f"Wall-clock timeout after {timeout_secs}s"),
            )
            logger.warning(
                "⏰ Synthesis task %s timed out after %.0fs",
                task_id,
                timeout_secs,
            )
            partial = self._build_synthesis_timeout_result(
                task_id,
                timeout_secs,
            )
            self._completed_results.put(
                task_id,
                {
                    "status": "timeout",
                    "result": partial.model_dump(),
                },
            )
            return partial

        except Exception as e:
            # 🛡️ Record failure for degradation tracking
            error_type = classify_error(e)
            if error_type == "transient":
                self._degradation.suggest_llm_fallback(
                    reason=(f"Transient error in synthesis task {task_id}: {e}"),
                )

            logger.error(
                "❌ Synthesis task %s failed: %s",
                task_id,
                e,
            )
            self._completed_results.put(task_id, {"status": "failed", "error": str(e)})
            raise

        finally:
            self._unregister_active_task(task_id)
            clear_logging_context()

    def run_research_sync(
        self,
        task: EvaluationTask,
    ) -> EvaluationResult:
        """Run a lightweight synchronous research task 🔬.

        Used by SynthesisExp deep-dive. Runs SearchAgent directly
        with adaptive_search=True so the Agent performs in-context
        search+reasoning. Avoids event loop nesting by using
        ThreadPoolExecutor.

        Args:
            task: Evaluation task to research.

        Returns:
            EvaluationResult from adaptive search.
        """
        import concurrent.futures

        from inquiro.core.types import INTENSITY_PRESETS, DiscoveryConfig
        from inquiro.exps.search_exp import SearchExp
        from inquiro.infrastructure.cancellation import CancellationToken
        from inquiro.infrastructure.cost_tracker import CostTracker

        # 🔧 Get resources
        llm = self._get_llm(
            getattr(task.agent_config, "model", None),
        )
        tools = self._get_filtered_tools(
            task.tools_config if hasattr(task, "tools_config") else None,
        )

        # 🎯 Inject tool effectiveness tracker
        for tool in tools.get_all_tools():
            if hasattr(tool, "_effectiveness_tracker"):
                tool._effectiveness_tracker = self._tool_effectiveness

        cost_tracker = CostTracker(
            max_per_task=task.cost_guard.max_cost_per_task
            if hasattr(task, "cost_guard") and task.cost_guard
            else 10.0,
            max_total=100.0,
        )
        mcp_server_configs = self._get_mcp_server_configs(
            task.tools_config if hasattr(task, "tools_config") else None,
        )

        # 🔍 Create SearchExp with adaptive mode
        search_exp = SearchExp(
            llm=llm,
            tools=tools,
            cost_tracker=cost_tracker,
            cancellation_token=CancellationToken(),
            skill_registry=self.skill_registry,
            adaptive_search=True,
            mcp_server_configs=mcp_server_configs,
        )

        config = DiscoveryConfig(**INTENSITY_PRESETS["standard"])

        # 🏃 Run in separate thread to avoid event loop nesting
        async def _run() -> Any:
            return await search_exp.run_search(
                task=task,
                config=config,
                round_number=1,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _run())
            search_result = future.result(timeout=300)

        # 🔄 Convert SearchRoundResult → EvaluationResult
        return self._search_result_to_evaluation_result(search_result, task)

    def _search_result_to_evaluation_result(
        self,
        search_result: Any,
        task: EvaluationTask,
    ) -> EvaluationResult:
        """Convert SearchRoundResult to EvaluationResult for sync path 🔄.

        Args:
            search_result: Result from SearchExp.run_search().
            task: Original evaluation task.

        Returns:
            EvaluationResult with agent self-assessment confidence.
        """
        from inquiro.core.types import Decision, Evidence, EvaluationResult

        # Extract evidence from search result
        evidence_list: list[Evidence] = []
        if hasattr(search_result, 'evidence') and search_result.evidence:
            evidence_list = self._map_evidence_list(search_result.evidence)
        elif (
            hasattr(search_result, 'raw_evidence')
            and search_result.raw_evidence
        ):
            evidence_list = self._map_evidence_list(
                search_result.raw_evidence,
            )

        # Extract decision/confidence from adaptive finish tool output
        raw_output = getattr(search_result, 'raw_output', {}) or {}
        decision_str = raw_output.get("decision", "cautious")
        confidence = raw_output.get("confidence", 0.5)

        try:
            decision = Decision(decision_str.lower())
        except (ValueError, AttributeError):
            decision = Decision.CAUTIOUS

        return EvaluationResult(
            task_id=task.task_id,
            decision=decision,
            confidence=min(max(float(confidence), 0.0), 1.0),
            evidence_index=evidence_list,
            search_rounds=1,
            cost=getattr(search_result, 'cost', 0.0),
            pipeline_mode="discovery",
            confidence_source="agent_self_assessment",
        )

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        """Get the status of a running or completed task 📋.

        Thread-safe: acquires ``_tasks_lock`` for consistent reads.

        Args:
            task_id: The unique task identifier.

        Returns:
            Status dictionary with:
                - task_id: The task identifier.
                - status: "running" | "completed" | "failed" | "cancelled".
                - is_active: Whether the task is currently running.
        """
        with self._tasks_lock:
            is_active = task_id in self._active_tasks
            token = self._active_tasks.get(task_id)

        # 🔍 Check completed results
        completed = self._completed_results.get(task_id)
        if completed:
            return {
                "task_id": task_id,
                "status": completed["status"],
                "is_active": is_active,
                "result": completed.get("result"),
                "error": completed.get("error"),
            }

        # 🔄 Currently active
        if is_active and token is not None:
            if token.is_cancelled:
                return {
                    "task_id": task_id,
                    "status": "cancelled",
                    "is_active": True,
                }
            return {
                "task_id": task_id,
                "status": "running",
                "is_active": True,
            }

        # ❓ Unknown task
        return {
            "task_id": task_id,
            "status": "unknown",
            "is_active": False,
        }

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task 🛑.

        Sets the cancellation token for the specified task, which will
        cause the agent to stop at the next step boundary.

        Thread-safe: acquires ``_tasks_lock`` for the lookup.

        Args:
            task_id: The unique task identifier to cancel.

        Returns:
            True if task was found and cancellation was signaled.
            False if task was not found (already completed or unknown).
        """
        with self._tasks_lock:
            token = self._active_tasks.get(task_id)
        if token is not None:
            token.cancel(reason=f"User requested cancellation of {task_id}")
            logger.info("🛑 Cancellation requested for task %s", task_id)
            return True
        logger.warning("⚠️ Cannot cancel task %s: not found in active tasks", task_id)
        return False

    async def cleanup(self) -> None:
        """Clean up all managed resources 🧹.

        Cancels all active tasks, closes MCP connections, and releases
        LLM provider resources. Called during application shutdown.

        Thread-safe: acquires ``_tasks_lock`` for task operations.
        """
        logger.info("🧹 Cleaning up EvalTaskRunner resources...")

        # 🛑 Cancel all active tasks (snapshot under lock)
        with self._tasks_lock:
            snapshot = dict(self._active_tasks)
        for task_id, token in snapshot.items():
            if not token.is_cancelled:
                token.cancel(reason="Application shutdown")
                logger.info("🛑 Cancelled task %s for shutdown", task_id)

        # 🔌 Close MCP pool (if it has a cleanup method)
        if hasattr(self.mcp_pool, "close"):
            try:
                close_result = self.mcp_pool.close()
                if asyncio.iscoroutine(close_result):
                    await close_result
                logger.info("🔌 MCP connection pool closed")
            except Exception as e:
                logger.warning("⚠️ Error closing MCP pool: %s", e)

        # 🔌 Close LLM pool (if it has a cleanup method)
        if hasattr(self.llm_pool, "close"):
            try:
                close_result = self.llm_pool.close()
                if asyncio.iscoroutine(close_result):
                    await close_result
                logger.info("🔌 LLM provider pool closed")
            except Exception as e:
                logger.warning("⚠️ Error closing LLM pool: %s", e)

        with self._tasks_lock:
            self._active_tasks.clear()
        logger.info("✅ EvalTaskRunner cleanup complete")

    def _register_active_task(
        self,
        task_id: str,
        token: CancellationToken,
    ) -> None:
        """Register a task as active with its cancellation token 📝.

        Thread-safe: acquires ``_tasks_lock``.

        Args:
            task_id: The unique task identifier.
            token: The cancellation token for cooperative cancellation.
        """
        with self._tasks_lock:
            self._active_tasks[task_id] = token
        logger.debug("📝 Registered active task: %s", task_id)

    def _unregister_active_task(self, task_id: str) -> None:
        """Unregister a completed/cancelled task 📝.

        Thread-safe: acquires ``_tasks_lock``.

        Args:
            task_id: The unique task identifier to remove.
        """
        with self._tasks_lock:
            self._active_tasks.pop(task_id, None)
        logger.debug("📝 Unregistered task: %s", task_id)

    def _create_cost_tracker(
        self,
        max_per_task: float,
        max_total: float | None = None,
    ) -> CostTracker:
        """Create a CostTracker instance for a task 💰.

        Args:
            max_per_task: Maximum cost per individual task (USD).
            max_total: Maximum total cost across all sub-tasks (USD).
                Defaults to 3x max_per_task if not specified.

        Returns:
            Configured CostTracker instance.
        """
        effective_max_total = max_total or (max_per_task * 3.0)
        return CostTracker(
            max_per_task=max_per_task,
            max_total=effective_max_total,
        )

    def _create_event_emitter(self) -> EventEmitter:
        """Create a fresh EventEmitter for tasks 📡.

        Returns:
            A new EventEmitter instance.
        """
        return EventEmitter()

    def _get_llm(self, model: str | None = None) -> Any:
        """Get an LLM instance from the provider pool 🤖.

        Delegates to the LLM pool's ``get_llm`` method. Falls back to
        returning the pool itself if no such method exists (for simple
        single-LLM setups).

        Args:
            model: LLM model identifier (e.g., "claude-bedrock").
                None uses the pool's default model.

        Returns:
            LLM instance suitable for passing to agent constructors.
        """
        if hasattr(self.llm_pool, "get_llm"):
            return self.llm_pool.get_llm(model)
        # 🔄 Fallback: pool itself is the LLM instance
        return self.llm_pool

    def _get_filtered_tools(self, tools_config: Any) -> Any:
        """Get a filtered ToolRegistry from the MCP pool 🔧.

        Delegates to the MCP pool's ``get_tools`` method. When a
        ``domain`` attribute is present on ``tools_config``, applies
        domain-based tool routing via ``ToolRoutingStrategy``.

        Falls back to returning an empty ToolRegistry if no such
        method exists on the MCP pool.

        Args:
            tools_config: ToolsConfig specifying which MCP servers
                to use and optional domain for routing.

        Returns:
            ToolRegistry with filtered tools for the task.
        """
        if hasattr(self.mcp_pool, "get_tools"):
            # ⚠️ None -> all servers; non-empty list -> only those servers
            mcp_servers: list[str] | None = None
            if hasattr(tools_config, "mcp_servers") and tools_config.mcp_servers:
                mcp_servers = tools_config.mcp_servers
            registry = self.mcp_pool.get_tools(mcp_servers)

            # 🎯 Apply tool routing strategy for domain-based filtering
            domain = getattr(tools_config, "domain", None)
            if domain and self._tool_routing is not None:
                filtered_tools = self._tool_routing.filter_tools(
                    registry=registry,
                    domain=domain,
                )
                # 📊 Rebuild registry from filtered tool list
                from evomaster.agent.tools.base import ToolRegistry

                filtered_registry = ToolRegistry()
                for tool in filtered_tools:
                    filtered_registry.register(tool)
                logger.info(
                    "🎯 Tool routing: %d/%d tools selected for domain '%s'",
                    len(filtered_tools),
                    len(registry.get_all_tools()),
                    domain,
                )
                return filtered_registry

            return registry

        # ⚠️ Fallback: return empty registry — agent will have no
        # search tools
        from evomaster.agent.tools.base import ToolRegistry

        logger.warning(
            "⚠️ MCP pool does not support get_tools() "
            "(tools_config=%s), returning empty tool registry "
            "— agent will have no search tools",
            tools_config,
        )
        return ToolRegistry()

    def _get_mcp_server_configs(self, tools_config: Any) -> dict[str, dict[str, Any]]:
        """Get MCP server configs scoped to current task 🔧.

        Uses task-selected `mcp_servers` when available; otherwise
        falls back to all configured MCP servers.
        """
        all_configs = getattr(self.mcp_pool, "servers", {})
        if not isinstance(all_configs, dict):
            return {}

        selected = getattr(tools_config, "mcp_servers", None)
        if not selected:
            return dict(all_configs)

        scoped: dict[str, dict[str, Any]] = {}
        for name in selected:
            cfg = all_configs.get(name)
            if isinstance(cfg, dict):
                scoped[name] = cfg
        return scoped

    def get_active_task_count(self) -> int:
        """Return the number of currently active tasks 📊.

        Thread-safe: acquires ``_tasks_lock``.

        Returns:
            Count of tasks in _active_tasks.
        """
        with self._tasks_lock:
            return len(self._active_tasks)

    def get_active_task_ids(self) -> list[str]:
        """Return IDs of all currently active tasks 📋.

        Thread-safe: acquires ``_tasks_lock``.

        Returns:
            List of active task ID strings.
        """
        with self._tasks_lock:
            return list(self._active_tasks.keys())

    # ====================================================================
    # ⏰ Timeout helpers
    # ====================================================================

    @staticmethod
    def _resolve_timeout(
        cost_guard: Any,
    ) -> float:
        """Resolve the wall-clock timeout from task cost guard ⏰.

        Falls back to ``_DEFAULT_TIMEOUT_SECONDS`` when the cost_guard
        does not define ``timeout_seconds``.

        Args:
            cost_guard: CostGuardConfig (or compatible) instance.

        Returns:
            Timeout duration in seconds.
        """
        return (
            getattr(cost_guard, "timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
            or _DEFAULT_TIMEOUT_SECONDS
        )

    @staticmethod
    def _build_timeout_result(
        task_id: str,
        timeout_secs: float,
    ) -> EvaluationResult:
        """Build a partial EvaluationResult for a timed-out task ⏰.

        Returns a negative-decision result with zero confidence so
        downstream systems know the evaluation was incomplete.

        Args:
            task_id: The unique task identifier.
            timeout_secs: The timeout value that was exceeded.

        Returns:
            EvaluationResult stub marked as timed-out.
        """
        from inquiro.core.types import (
            ChecklistCoverage,
            Decision,
            EvaluationResult,
            EvidenceStrength,
            ReasoningClaim,
        )

        return EvaluationResult(
            task_id=task_id,
            decision=Decision.NEGATIVE,
            confidence=0.0,
            reasoning=[
                ReasoningClaim(
                    claim=(
                        f"Task timed out after {timeout_secs:.0f}s "
                        "before completing research."
                    ),
                    evidence_ids=[],
                    strength=EvidenceStrength.WEAK,
                ),
            ],
            evidence_index=[],
            search_rounds=0,
            round_logs=[],
            checklist_coverage=ChecklistCoverage(),
            gaps_remaining=[
                f"Wall-clock timeout ({timeout_secs:.0f}s exceeded)",
            ],
            doubts_remaining=[],
            cost=0.0,
        )

    @staticmethod
    def _build_synthesis_timeout_result(
        task_id: str,
        timeout_secs: float,
    ) -> SynthesisResult:
        """Build a partial SynthesisResult for a timed-out task ⏰.

        Returns a negative-decision result with zero confidence so
        downstream systems know the synthesis was incomplete.

        Args:
            task_id: The unique task identifier.
            timeout_secs: The timeout value that was exceeded.

        Returns:
            SynthesisResult stub marked as timed-out.
        """
        from inquiro.core.types import (
            Decision,
            EvidenceStrength,
            ReasoningClaim,
            SynthesisResult,
        )

        return SynthesisResult(
            task_id=task_id,
            decision=Decision.NEGATIVE,
            confidence=0.0,
            reasoning=[
                ReasoningClaim(
                    claim=(
                        f"Task timed out after {timeout_secs:.0f}s "
                        "before completing synthesis."
                    ),
                    evidence_ids=[],
                    strength=EvidenceStrength.WEAK,
                ),
            ],
            evidence_index=[],
            source_reports=[],
            cross_references=[],
            gaps_remaining=[
                f"Wall-clock timeout ({timeout_secs:.0f}s exceeded)",
            ],
            deep_dives_triggered=[],
            cost=0.0,
        )


# ============================================================================
# 🔧 Protocol adapters for DiscoveryLoop integration
# ============================================================================


class _SearchExpAdapter:
    """Adapts SearchExp to SearchExecutor protocol for DiscoveryLoop 🔧.

    Thin wrapper that creates a SearchExp per round, runs it, and
    converts the SearchRoundResult into a SearchRoundOutput expected
    by the DiscoveryLoop protocol.

    The adapter does NOT hold LLM — it obtains LLM and tools from the
    runner's shared resource pools for each round.

    Attributes:
        _runner: Reference to EvalTaskRunner for resource access.
        _event_emitter: SSE event emitter for progress updates.
        _cancellation_token: Cooperative cancellation signal.
        _discovery_loop: Optional reference to DiscoveryLoop for
            reading evolution enrichment state.
    """

    def __init__(
        self,
        runner: EvalTaskRunner,
        event_emitter: EventEmitter | None = None,
        cancellation_token: CancellationToken | None = None,
        feedback_provider: FeedbackProvider | None = None,
    ) -> None:
        """Initialize adapter with runner reference 🔧.

        Args:
            runner: EvalTaskRunner instance for shared resources.
            event_emitter: SSE event emitter.
            cancellation_token: Cancellation signal.
            feedback_provider: Optional FeedbackProvider for injecting
                historical search hints into SearchExp.
        """
        self._runner = runner
        self._event_emitter = event_emitter
        self._cancellation_token = cancellation_token
        self._feedback_provider = feedback_provider
        self._discovery_loop: Any = None  # Set after DiscoveryLoop init
        self._logger = logging.getLogger(
            f"{__name__}._SearchExpAdapter",
        )

    async def execute_search(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None = None,
    ) -> Any:
        """Execute one search round via SearchExp 🔍.

        Delegates to ``ParallelSearchOrchestrator`` when parallel search
        is enabled and the task carries a multi-section query strategy.
        Falls back to ``_execute_single_search`` otherwise.

        Args:
            task: Evaluation task with rules and checklist.
            config: Discovery pipeline configuration.
            round_number: Current round number (1-based).
            focus_prompt: Optional focus guidance for targeted search.

        Returns:
            SearchRoundOutput with raw evidence and metadata.
        """
        # 🔄 Parallel dispatch when conditions are met
        if (
            config.enable_parallel_search
            and task.query_strategy is not None
            and len(task.query_strategy.get("query_sections", [])) > 1
        ):
            from inquiro.exps.parallel_search_exp import (
                ParallelSearchOrchestrator,
            )

            self._logger.info(
                "🔄 SearchExpAdapter: parallel search path "
                "(round=%d, task=%s, sections=%d)",
                round_number,
                task.task_id,
                len(task.query_strategy.get("query_sections", [])),
            )
            orchestrator = ParallelSearchOrchestrator(
                max_parallel=config.max_parallel_agents,
            )
            return await orchestrator.execute(
                task=task,
                config=config,
                round_number=round_number,
                focus_prompt=focus_prompt,
                single_search_fn=self._execute_single_search,
            )

        # 🔍 Single-search path (legacy default)
        return await self._execute_single_search(
            task,
            config,
            round_number,
            focus_prompt,
        )

    async def _execute_single_search(
        self,
        task: EvaluationTask,
        config: DiscoveryConfig,
        round_number: int,
        focus_prompt: str | None = None,
    ) -> Any:
        """Execute a single search round (core logic) 🔍.

        Creates a fresh SearchExp instance, runs it, and converts
        the result to SearchRoundOutput for DiscoveryLoop.

        Args:
            task: Evaluation task with rules and checklist.
            config: Discovery pipeline configuration.
            round_number: Current round number (1-based).
            focus_prompt: Optional focus guidance for targeted search.

        Returns:
            SearchRoundOutput with raw evidence and metadata.
        """
        from inquiro.core.discovery_loop import SearchRoundOutput
        from inquiro.exps.search_exp import SearchExp

        self._logger.info(
            "🔍 SearchExpAdapter: starting round %d for task %s",
            round_number,
            task.task_id,
        )

        # 🔧 Get LLM and tools from runner pools
        llm = self._runner._get_llm(task.agent_config.model)
        tools = self._runner._get_filtered_tools(task.tools_config)

        # 🎯 Inject tool effectiveness tracker into MCP wrappers
        for tool in tools.get_all_tools():
            if hasattr(tool, "_effectiveness_tracker"):
                tool._effectiveness_tracker = self._runner._tool_effectiveness

        # 💰 Create cost tracker for this search round
        cost_tracker = self._runner._create_cost_tracker(
            max_per_task=task.cost_guard.max_cost_per_task,
        )

        # 🧬 Read evolution enrichment from DiscoveryLoop
        evolution_enrichment: str | None = None
        if self._discovery_loop:
            evolution_enrichment = getattr(
                self._discovery_loop,
                "_current_search_enrichment",
                None,
            )

        # 🏗️ Create SearchExp
        search_exp = SearchExp(
            llm=llm,
            tools=tools,
            event_emitter=self._event_emitter,
            cost_tracker=cost_tracker,
            cancellation_token=self._cancellation_token,
            feedback_provider=self._feedback_provider,
            evolution_enrichment=evolution_enrichment,
            skill_registry=self._runner.skill_registry,
            adaptive_search=False,
            mcp_server_configs=self._runner._get_mcp_server_configs(
                task.tools_config if hasattr(task, 'tools_config') else None,
            ),
        )

        # 🚀 Run search (pass trajectory_dir for agent JSONL recording)
        result = await search_exp.run_search(
            task=task,
            config=config,
            round_number=round_number,
            focus_prompt=focus_prompt,
            agent_trajectory_dir=task.trajectory_dir or None,
        )

        # 🔄 Convert SearchRoundResult → SearchRoundOutput
        # Map CleanedEvidence back to Evidence for the loop
        from inquiro.core.types import Evidence

        evidence_items: list[Evidence] = []
        for ce in result.cleaned_evidence:
            evidence_items.append(
                Evidence(
                    id=ce.id,
                    source=ce.mcp_server,
                    url=ce.url,
                    query=ce.source_query,
                    summary=ce.summary,
                )
            )

        return SearchRoundOutput(
            evidence=evidence_items,
            queries_executed=result.queries_executed,
            mcp_tools_used=result.mcp_tools_used,
            cost_usd=result.cost_usd,
            duration_seconds=result.duration_seconds,
            agent_trajectory_ref=result.agent_trajectory_ref,
        )


class _AnalysisExpAdapter:
    """Adapts AnalysisExp to AnalysisExecutor protocol for DiscoveryLoop 🔧.

    Thin wrapper that creates an AnalysisExp per round, runs it, and
    converts the AggregatedResult into an AnalysisRoundOutput expected
    by the DiscoveryLoop protocol.

    The adapter does NOT hold LLM — it obtains LLM from the runner's
    shared resource pools for each round.

    Attributes:
        _runner: Reference to EvalTaskRunner for resource access.
        _event_emitter: SSE event emitter for progress updates.
        _cancellation_token: Cooperative cancellation signal.
        _discovery_loop: Optional reference to DiscoveryLoop for
            reading evolution enrichment state.
    """

    def __init__(
        self,
        runner: EvalTaskRunner,
        event_emitter: EventEmitter | None = None,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Initialize adapter with runner reference 🔧.

        Args:
            runner: EvalTaskRunner instance for shared resources.
            event_emitter: SSE event emitter.
            cancellation_token: Cancellation signal.
        """
        self._runner = runner
        self._event_emitter = event_emitter
        self._cancellation_token = cancellation_token
        self._discovery_loop: Any = None  # Set after DiscoveryLoop init
        self._logger = logging.getLogger(
            f"{__name__}._AnalysisExpAdapter",
        )

    async def execute_analysis(
        self,
        task: EvaluationTask,
        evidence: list[Any],
        config: DiscoveryConfig,
        round_number: int,
        supplementary_context: str | None = None,
    ) -> Any:
        """Analyze evidence with multi-model consensus via AnalysisExp 🔬.

        Creates a fresh AnalysisExp instance, runs parallel analysis
        across configured models, and converts to AnalysisRoundOutput.

        Args:
            task: Evaluation task with rules and checklist.
            evidence: Cleaned evidence to analyze (Evidence objects).
            config: Discovery pipeline configuration.
            round_number: Current round number (1-based).
            supplementary_context: Optional text summaries of evidence
                excluded by Tier-2 condensation, forwarded to AnalysisExp.

        Returns:
            AnalysisRoundOutput with claims and consensus info.
        """
        from inquiro.core.discovery_loop import AnalysisRoundOutput
        from inquiro.core.types import CleanedEvidence, EvidenceTag
        from inquiro.exps.analysis_exp import AnalysisExp

        self._logger.info(
            "🔬 AnalysisExpAdapter: starting round %d for task %s "
            "with %d evidence items",
            round_number,
            task.task_id,
            len(evidence),
        )

        # 🔧 Get default LLM from runner
        llm = self._runner._get_llm(task.agent_config.model)

        # 💰 Create cost tracker for this analysis round
        cost_tracker = self._runner._create_cost_tracker(
            max_per_task=task.cost_guard.max_cost_per_task,
        )

        # 🧬 Read evolution enrichment from DiscoveryLoop
        evolution_enrichment: str | None = None
        if self._discovery_loop:
            evolution_enrichment = getattr(
                self._discovery_loop,
                "_current_analysis_enrichment",
                None,
            )

        # 🏗️ Create AnalysisExp
        analysis_exp = AnalysisExp(
            task=task,
            llm=llm,
            llm_pool=self._runner.llm_pool,
            cost_tracker=cost_tracker,
            event_emitter=self._event_emitter,
            cancellation_token=self._cancellation_token,
            evolution_enrichment=evolution_enrichment,
        )

        # 🔄 Convert Evidence → CleanedEvidence for AnalysisExp
        cleaned_evidence: list[CleanedEvidence] = []
        for ev in evidence:
            cleaned_evidence.append(
                CleanedEvidence(
                    id=ev.id,
                    summary=ev.summary,
                    url=ev.url,
                    tag=EvidenceTag.OTHER,
                    source_query=ev.query,
                    mcp_server=ev.source,
                )
            )

        # 🚀 Run analysis
        aggregated, analysis_cost = await analysis_exp.run_analysis(
            task=task,
            cleaned_evidence=cleaned_evidence,
            analysis_models=config.analysis_models,
            round_number=round_number,
            supplementary_context=supplementary_context,
        )

        # 🔄 Convert AggregatedResult → AnalysisRoundOutput
        # Extract claims from aggregated reasoning
        claims: list[dict[str, Any]] = []
        if hasattr(aggregated, "structured_reasoning"):
            claims = list(aggregated.structured_reasoning or [])
        elif hasattr(aggregated, "reasoning_summary"):
            if aggregated.reasoning_summary:
                claims = [
                    {
                        "claim": aggregated.reasoning_summary,
                        "evidence_ids": [],
                        "strength": "moderate",
                    }
                ]

        # 📊 Build model_decisions from aggregated result
        model_decisions: list[dict[str, Any]] = []
        if hasattr(aggregated, "model_decisions"):
            for model_name, dec in (aggregated.model_decisions or {}).items():
                model_decisions.append(
                    {
                        "model": model_name,
                        "decision": dec,
                    }
                )

        return AnalysisRoundOutput(
            claims=claims,
            model_decisions=model_decisions,
            consensus_decision=(
                aggregated.decision.value
                if hasattr(aggregated.decision, "value")
                else str(aggregated.decision)
            ),
            consensus_ratio=getattr(
                aggregated,
                "consensus_ratio",
                0.0,
            ),
            cost_usd=analysis_cost,
            duration_seconds=0.0,
            checklist_coverage=getattr(
                aggregated, "checklist_coverage", None
            ),
            coverage_conflicts=getattr(
                aggregated, "coverage_conflicts", []
            ),
            gaps_remaining=getattr(
                aggregated, "gaps_remaining", []
            ),
            doubts_remaining=getattr(
                aggregated, "doubts_remaining", []
            ),
        )
