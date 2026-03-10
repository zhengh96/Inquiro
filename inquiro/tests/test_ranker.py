"""Tests for ExperienceRanker — pruning, decay, conflicts, merging 🧪.

Tests the ExperienceRanker class for:
- Pruning low-fitness experiences
- Applying fitness decay over time
- Detecting contradictory insights
- Resolving conflicts by keeping higher-fitness experiences
- Merging similar experiences to reduce redundancy

Uses Google Python Style Guide. English comments with emojis.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from inquiro.evolution.ranker import ExperienceRanker
from inquiro.evolution.types import (
    Experience,
    PruneConfig,
)


# ============================================================================
# 🏗️ Fixtures
# ============================================================================


@pytest.fixture
def mock_store() -> AsyncMock:
    """Create a mock ExperienceStore for testing 🏗️.

    Returns:
        AsyncMock configured with typical ExperienceStore methods.
    """
    store = AsyncMock()
    store.prune = AsyncMock()
    store.apply_decay = AsyncMock()
    store.list_by_namespace = AsyncMock()
    store.get_by_id = AsyncMock()
    store.update_fitness = AsyncMock()
    store.update = AsyncMock()
    store.delete = AsyncMock()
    return store


@pytest.fixture
def ranker(mock_store: AsyncMock) -> ExperienceRanker:
    """Create an ExperienceRanker with mocked store 🏗️.

    Args:
        mock_store: Mocked ExperienceStore.

    Returns:
        ExperienceRanker instance.
    """
    return ExperienceRanker(store=mock_store)


# ============================================================================
# 🧪 ExperienceRanker Tests
# ============================================================================


class TestExperienceRanker:
    """Tests for ExperienceRanker class 🧪."""

    def test_initialization(self, mock_store: AsyncMock) -> None:
        """ExperienceRanker should initialize with store reference 🏗️."""
        ranker = ExperienceRanker(store=mock_store)
        assert ranker._store is mock_store

    @pytest.mark.asyncio
    async def test_prune_delegates_to_store(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Prune should delegate to store.prune() 🗑️."""
        mock_store.prune.return_value = 5

        config = PruneConfig(
            min_fitness=0.3,
            min_uses=5,
            decay_factor=0.95,
            decay_interval_days=7,
        )

        pruned_count = await ranker.prune("targetmaster", config)

        assert pruned_count == 5
        mock_store.prune.assert_called_once_with("targetmaster", config)

    @pytest.mark.asyncio
    async def test_apply_decay_delegates_to_store(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Apply_decay should delegate to store.apply_decay() 📉."""
        mock_store.apply_decay.return_value = 10

        decayed_count = await ranker.apply_decay("targetmaster", 0.95)

        assert decayed_count == 10
        mock_store.apply_decay.assert_called_once_with("targetmaster", 0.95)

    @pytest.mark.asyncio
    async def test_apply_decay_invalid_factor(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """Apply_decay should reject invalid decay_factor ❌."""
        with pytest.raises(ValueError, match="decay_factor must be in"):
            await ranker.apply_decay("targetmaster", 0.0)

        with pytest.raises(ValueError, match="decay_factor must be in"):
            await ranker.apply_decay("targetmaster", 1.5)

    @pytest.mark.asyncio
    async def test_detect_conflicts_empty_namespace(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Detect_conflicts should return empty list for empty namespace 🔍."""
        mock_store.list_by_namespace.return_value = []

        conflicts = await ranker.detect_conflicts("targetmaster")

        assert conflicts == []

    @pytest.mark.asyncio
    async def test_detect_conflicts_single_category(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Detect_conflicts should find contradictory insights 🔍."""
        # 🏗️ Create experiences with conflicting insights
        exp1 = Experience(
            id="exp1",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for comprehensive literature search",
            source="trajectory",
        )
        exp2 = Experience(
            id="exp2",
            namespace="targetmaster",
            category="search_strategy",
            insight="Avoid PubMed due to low signal-to-noise ratio",
            source="trajectory",
        )
        mock_store.list_by_namespace.return_value = [exp1, exp2]

        conflicts = await ranker.detect_conflicts("targetmaster")

        # Should detect conflict between exp1 (use PubMed) and exp2 (avoid PubMed)
        assert len(conflicts) == 1
        assert conflicts[0] == ("exp1", "exp2")

    @pytest.mark.asyncio
    async def test_detect_conflicts_no_conflicts(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Detect_conflicts should return empty for non-conflicting insights 🔍."""
        exp1 = Experience(
            id="exp1",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for comprehensive literature search",
            source="trajectory",
        )
        exp2 = Experience(
            id="exp2",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use Google Scholar for recent preprints",
            source="trajectory",
        )
        mock_store.list_by_namespace.return_value = [exp1, exp2]

        conflicts = await ranker.detect_conflicts("targetmaster")

        # No conflict — both suggest using different tools
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_detect_conflicts_different_categories(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Detect_conflicts should only compare within same category 🔍."""
        exp1 = Experience(
            id="exp1",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use source A",
            source="trajectory",
        )
        exp2 = Experience(
            id="exp2",
            namespace="targetmaster",
            category="checklist_insight",
            insight="Avoid source A",
            source="trajectory",
        )
        mock_store.list_by_namespace.return_value = [exp1, exp2]

        conflicts = await ranker.detect_conflicts("targetmaster")

        # Different categories — should not conflict
        assert conflicts == []

    @pytest.mark.asyncio
    async def test_resolve_conflicts_empty_list(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Resolve_conflicts should handle empty list gracefully 🔧."""
        resolved_count = await ranker.resolve_conflicts([])

        assert resolved_count == 0
        mock_store.get_by_id.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_conflicts_keeps_higher_fitness(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Resolve_conflicts should keep higher-fitness experience 🔧."""
        exp1 = Experience(
            id="exp1",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed",
            source="trajectory",
            fitness_score=0.7,
        )
        exp2 = Experience(
            id="exp2",
            namespace="targetmaster",
            category="search_strategy",
            insight="Avoid PubMed",
            source="trajectory",
            fitness_score=0.4,
        )

        # Mock store to return experiences
        async def mock_get_by_id(exp_id: str):
            if exp_id == "exp1":
                return exp1
            elif exp_id == "exp2":
                return exp2
            return None

        mock_store.get_by_id.side_effect = mock_get_by_id

        conflicts = [("exp1", "exp2")]
        resolved_count = await ranker.resolve_conflicts(conflicts)

        assert resolved_count == 1
        # Should mark exp2 (lower fitness) for pruning via FitnessUpdate
        mock_store.update_fitness.assert_called_once()
        call_args = mock_store.update_fitness.call_args
        # Positional args: (exp_id, FitnessUpdate)
        assert call_args[0][0] == "exp2"
        delta = call_args[0][1]
        assert delta.experience_id == "exp2"
        assert delta.signal == 0.0
        assert delta.was_helpful is False

    @pytest.mark.asyncio
    async def test_resolve_conflicts_missing_experience(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Resolve_conflicts should skip missing experiences gracefully ⚠️."""
        mock_store.get_by_id.return_value = None

        conflicts = [("exp1", "exp2")]
        resolved_count = await ranker.resolve_conflicts(conflicts)

        assert resolved_count == 0
        mock_store.update_fitness.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_similar_empty_namespace(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Merge_similar should return 0 for empty namespace 🔗."""
        mock_store.list_by_namespace.return_value = []

        merged_count = await ranker.merge_similar("targetmaster", 0.8)

        assert merged_count == 0

    @pytest.mark.asyncio
    async def test_merge_similar_invalid_threshold(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """Merge_similar should reject invalid similarity_threshold ❌."""
        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            await ranker.merge_similar("targetmaster", -0.1)

        with pytest.raises(ValueError, match="similarity_threshold must be in"):
            await ranker.merge_similar("targetmaster", 1.5)

    @pytest.mark.asyncio
    async def test_merge_similar_high_similarity(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Merge_similar should merge experiences with high text overlap 🔗."""
        exp1 = Experience(
            id="exp1",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for comprehensive literature search on targets",
            source="trajectory",
            fitness_score=0.7,
            context_tags=["modality:SmallMolecule"],
        )
        exp2 = Experience(
            id="exp2",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for comprehensive literature search",
            source="trajectory",
            fitness_score=0.5,
            context_tags=["modality:Antibody"],
        )
        mock_store.list_by_namespace.return_value = [exp1, exp2]

        # Mock get_by_id for keeper
        async def mock_get_by_id(exp_id: str):
            if exp_id == "exp1":
                return exp1
            elif exp_id == "exp2":
                return exp2
            return None

        mock_store.get_by_id.side_effect = mock_get_by_id

        merged_count = await ranker.merge_similar("targetmaster", 0.7)

        # Should merge exp2 into exp1 (higher fitness)
        assert merged_count == 1
        # Should update exp1 with merged tags
        mock_store.update.assert_called_once()
        # Should delete exp2
        mock_store.delete.assert_called_once_with("exp2")

    @pytest.mark.asyncio
    async def test_merge_similar_low_similarity(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Merge_similar should not merge experiences with low similarity 🔗."""
        exp1 = Experience(
            id="exp1",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for literature search",
            source="trajectory",
            fitness_score=0.7,
        )
        exp2 = Experience(
            id="exp2",
            namespace="targetmaster",
            category="search_strategy",
            insight="Focus on clinical trial databases",
            source="trajectory",
            fitness_score=0.5,
        )
        mock_store.list_by_namespace.return_value = [exp1, exp2]

        merged_count = await ranker.merge_similar("targetmaster", 0.8)

        # Should not merge — low similarity
        assert merged_count == 0
        mock_store.update.assert_not_called()
        mock_store.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_merge_similar_different_categories(
        self,
        ranker: ExperienceRanker,
        mock_store: AsyncMock,
    ) -> None:
        """Merge_similar should only merge within same category 🔗."""
        exp1 = Experience(
            id="exp1",
            namespace="targetmaster",
            category="search_strategy",
            insight="Use PubMed for comprehensive search",
            source="trajectory",
            fitness_score=0.7,
        )
        exp2 = Experience(
            id="exp2",
            namespace="targetmaster",
            category="checklist_insight",
            insight="Use PubMed for comprehensive search",
            source="trajectory",
            fitness_score=0.5,
        )
        mock_store.list_by_namespace.return_value = [exp1, exp2]

        merged_count = await ranker.merge_similar("targetmaster", 0.8)

        # Should not merge — different categories
        assert merged_count == 0

    def test_compute_similarity_identical_texts(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_compute_similarity should return 1.0 for identical texts 🔍."""
        text = "Use PubMed for literature search"
        similarity = ranker._compute_similarity(text, text)
        assert similarity == 1.0

    def test_compute_similarity_completely_different(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_compute_similarity should return 0.0 for completely different texts 🔍."""
        text1 = "Use PubMed for search"
        text2 = "Focus clinical trial databases"
        similarity = ranker._compute_similarity(text1, text2)
        # Should be low but not necessarily 0.0 (may have common stopwords)
        assert similarity < 0.3

    def test_compute_similarity_partial_overlap(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_compute_similarity should compute Jaccard for partial overlap 🔍."""
        text1 = "Use PubMed for comprehensive search"
        text2 = "Use PubMed for quick search"
        similarity = ranker._compute_similarity(text1, text2)
        # Should have moderate similarity (shared words: use, pubmed, for, search)
        assert 0.5 < similarity < 1.0

    def test_compute_similarity_case_insensitive(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_compute_similarity should be case-insensitive 🔍."""
        text1 = "Use PubMed"
        text2 = "use pubmed"
        similarity = ranker._compute_similarity(text1, text2)
        assert similarity == 1.0

    def test_are_conflicting_use_vs_avoid(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_are_conflicting should detect use vs avoid patterns 🔍."""
        text1 = "Use PubMed for searches"
        text2 = "Avoid PubMed for searches"
        assert ranker._are_conflicting(text1, text2) is True

    def test_are_conflicting_prefer_vs_dont_use(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_are_conflicting should detect prefer vs don't use patterns 🔍."""
        text1 = "Prefer PubMed for searches"
        text2 = "Don't use PubMed for searches"
        assert ranker._are_conflicting(text1, text2) is True

    def test_are_conflicting_both_positive(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_are_conflicting should not flag both positive statements 🔍."""
        text1 = "Use PubMed"
        text2 = "Use Google Scholar"
        assert ranker._are_conflicting(text1, text2) is False

    def test_are_conflicting_both_negative(
        self,
        ranker: ExperienceRanker,
    ) -> None:
        """_are_conflicting should not flag both negative statements 🔍."""
        text1 = "Avoid PubMed"
        text2 = "Avoid Wikipedia"
        assert ranker._are_conflicting(text1, text2) is False
