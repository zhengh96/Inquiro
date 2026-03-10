"""Tests for Inquiro API endpoints 🧪.

Tests all REST API endpoints using httpx AsyncClient:
- POST /api/v1/research — Submit research task
- POST /api/v1/synthesize — Submit synthesis task
- GET /api/v1/task/{task_id} — Query task status
- GET /api/v1/task/{task_id}/stream — SSE progress stream
- DELETE /api/v1/task/{task_id} — Cancel task
- GET /api/v1/health — Health check

Uses pytest-asyncio + httpx + dependency overrides for isolated testing.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from inquiro.api.schemas import (
    ResearchRequest,
    SynthesizeRequest,
    TaskStatus,
    TaskType,
)
from inquiro.api.router import _resolve_event_emitter_instance
from inquiro.infrastructure.event_emitter import EventEmitter


# ============================================================
# 🔬 POST /api/v1/research Tests
# ============================================================


class TestSubmitResearch:
    """Tests for POST /api/v1/research endpoint 🔬."""

    @pytest.mark.asyncio
    async def test_submit_research_returns_202(
        self,
        async_client,
        sample_research_request: ResearchRequest,
    ) -> None:
        """Submitting valid research request should return 202 Accepted."""
        # Arrange
        payload = sample_research_request.model_dump()
        # Act
        response = await async_client.post("/api/v1/research", json=payload)
        # Assert
        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == sample_research_request.task_id
        assert data["status"] == "accepted"
        assert "stream_url" in data
        assert "poll_url" in data

    @pytest.mark.asyncio
    async def test_submit_research_stream_url_format(
        self,
        async_client,
        sample_research_request: ResearchRequest,
    ) -> None:
        """Stream URL should point to /api/v1/task/{id}/stream."""
        # Arrange
        payload = sample_research_request.model_dump()
        # Act
        response = await async_client.post("/api/v1/research", json=payload)
        # Assert
        data = response.json()
        expected_stream = f"/api/v1/task/{sample_research_request.task_id}/stream"
        assert data["stream_url"] == expected_stream

    @pytest.mark.asyncio
    async def test_submit_research_invalid_body_returns_422(
        self,
        async_client,
    ) -> None:
        """Invalid request body should return 422 Unprocessable Entity."""
        # Arrange: Missing required fields
        # Act
        response = await async_client.post("/api/v1/research", json={"invalid": "body"})
        # Assert
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_submit_research_empty_body_returns_422(
        self,
        async_client,
    ) -> None:
        """Empty request body should return 422."""
        # Arrange / Act
        response = await async_client.post("/api/v1/research", json={})
        # Assert
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_submit_research_creates_task_state(
        self,
        async_client,
        sample_research_request: ResearchRequest,
        test_task_store: dict[str, Any],
    ) -> None:
        """Submitting research should create queryable task state."""
        # Arrange
        payload = sample_research_request.model_dump()
        # Act
        await async_client.post("/api/v1/research", json=payload)
        # Assert: ✅ Task exists in store with correct type
        assert sample_research_request.task_id in test_task_store
        task_state = test_task_store[sample_research_request.task_id]
        assert task_state.task_type == TaskType.RESEARCH


class TestEventEmitterResolve:
    """Tests for defensive event emitter normalization 🔧."""

    def test_resolve_event_emitter_instance_passthrough(self) -> None:
        """Existing instance should be returned unchanged ✅."""
        emitter = EventEmitter()
        resolved = _resolve_event_emitter_instance(emitter)
        assert resolved is emitter

    def test_resolve_event_emitter_class_instantiates(self) -> None:
        """Class object should be instantiated automatically ✅."""
        resolved = _resolve_event_emitter_instance(EventEmitter)
        assert isinstance(resolved, EventEmitter)


# ============================================================
# 📊 POST /api/v1/synthesize Tests
# ============================================================


class TestSubmitSynthesis:
    """Tests for POST /api/v1/synthesize endpoint 📊."""

    @pytest.mark.asyncio
    async def test_submit_synthesis_returns_202(
        self,
        async_client,
        sample_synthesize_request: SynthesizeRequest,
    ) -> None:
        """Submitting valid synthesis request should return 202 Accepted."""
        # Arrange
        payload = sample_synthesize_request.model_dump()
        # Act
        response = await async_client.post("/api/v1/synthesize", json=payload)
        # Assert
        assert response.status_code == 202
        data = response.json()
        assert data["task_id"] == sample_synthesize_request.task_id
        assert data["status"] == "accepted"

    @pytest.mark.asyncio
    async def test_submit_synthesis_invalid_body_returns_422(
        self,
        async_client,
    ) -> None:
        """Invalid synthesis request should return 422."""
        # Arrange / Act
        response = await async_client.post("/api/v1/synthesize", json={"task_id": "x"})
        # Assert
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_submit_synthesis_empty_reports_returns_422(
        self,
        async_client,
    ) -> None:
        """Synthesis with empty input_reports should return 422."""
        # Arrange: input_reports has min_length=1
        payload = {
            "task_id": "test-synthesis-empty",
            "task": {
                "objective": "Test synthesis with no reports",
                "input_reports": [],
                "output_schema": {"type": "object"},
            },
        }
        # Act
        response = await async_client.post("/api/v1/synthesize", json=payload)
        # Assert
        assert response.status_code == 422


# ============================================================
# 📊 GET /api/v1/task/{task_id} Tests
# ============================================================


class TestGetTask:
    """Tests for GET /api/v1/task/{task_id} endpoint 📊."""

    @pytest.mark.asyncio
    async def test_get_existing_task(
        self,
        async_client,
        sample_research_request: ResearchRequest,
    ) -> None:
        """Getting an existing task should return 200 with status."""
        # Arrange: Submit a task first
        payload = sample_research_request.model_dump()
        await async_client.post("/api/v1/research", json=payload)
        # Act
        response = await async_client.get(
            f"/api/v1/task/{sample_research_request.task_id}"
        )
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == sample_research_request.task_id
        assert data["task_type"] == "research"

    @pytest.mark.asyncio
    async def test_get_nonexistent_task_returns_404(
        self,
        async_client,
    ) -> None:
        """Getting a non-existent task should return 404."""
        # Arrange / Act
        response = await async_client.get("/api/v1/task/nonexistent-task-id")
        # Assert
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_task_includes_trajectory_url(
        self,
        async_client,
        sample_research_request: ResearchRequest,
    ) -> None:
        """Task response should include trajectory_url."""
        # Arrange
        payload = sample_research_request.model_dump()
        await async_client.post("/api/v1/research", json=payload)
        # Act
        response = await async_client.get(
            f"/api/v1/task/{sample_research_request.task_id}"
        )
        # Assert
        assert response.status_code == 200
        data = response.json()
        expected = f"/api/v1/task/{sample_research_request.task_id}/trajectory"
        assert data["trajectory_url"] == expected

    @pytest.mark.asyncio
    async def test_get_completed_task_includes_result(
        self,
        async_client,
        test_task_store: dict[str, Any],
        sample_evaluation_result: dict[str, Any],
    ) -> None:
        """Completed task should include the result dict."""
        # Arrange: 🔧 Manually create completed task in store
        from inquiro.api.router import TaskState
        from inquiro.infrastructure.event_emitter import EventEmitter

        task_state = TaskState(
            task_id="completed-task-001",
            task_type=TaskType.RESEARCH,
            event_emitter=EventEmitter(),
        )
        task_state.status = TaskStatus.COMPLETED
        task_state.result = sample_evaluation_result
        task_state.cost = 0.5
        test_task_store["completed-task-001"] = task_state

        # Act
        response = await async_client.get("/api/v1/task/completed-task-001")
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"] is not None
        assert data["result"]["decision"] == "positive"
        assert data["cost"]["total_cost_usd"] == 0.5


# ============================================================
# 📡 GET /api/v1/task/{task_id}/stream Tests
# ============================================================


class TestStreamTask:
    """Tests for GET /api/v1/task/{task_id}/stream SSE endpoint 📡."""

    @pytest.mark.asyncio
    async def test_stream_nonexistent_task_returns_404(
        self,
        async_client,
    ) -> None:
        """Streaming a non-existent task should return 404."""
        # Arrange / Act
        response = await async_client.get("/api/v1/task/nonexistent/stream")
        # Assert
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_stream_returns_event_stream_content_type(
        self,
        async_client,
        test_task_store: dict[str, Any],
    ) -> None:
        """SSE endpoint should return text/event-stream content type."""
        # Arrange: 📡 Create task with terminal event in history
        from inquiro.api.router import TaskState
        from inquiro.infrastructure.event_emitter import EventEmitter

        emitter = EventEmitter()
        task_state = TaskState(
            task_id="stream-test-001",
            task_type=TaskType.RESEARCH,
            event_emitter=emitter,
        )
        test_task_store["stream-test-001"] = task_state

        # Emit events including a terminal one so stream will end
        emitter.emit("task_started", "stream-test-001", {"task_type": "research"})
        emitter.emit("task_completed", "stream-test-001", {"status": "completed"})

        # Act
        response = await async_client.get("/api/v1/task/stream-test-001/stream")
        # Assert
        assert response.headers["content-type"].startswith("text/event-stream")

    @pytest.mark.asyncio
    async def test_stream_includes_cache_control_headers(
        self,
        async_client,
        test_task_store: dict[str, Any],
    ) -> None:
        """SSE response should include no-cache headers."""
        # Arrange: 📡 Create task with terminal event
        from inquiro.api.router import TaskState
        from inquiro.infrastructure.event_emitter import EventEmitter

        emitter = EventEmitter()
        task_state = TaskState(
            task_id="stream-cache-001",
            task_type=TaskType.RESEARCH,
            event_emitter=emitter,
        )
        test_task_store["stream-cache-001"] = task_state

        emitter.emit("task_completed", "stream-cache-001", {"status": "completed"})

        # Act
        response = await async_client.get("/api/v1/task/stream-cache-001/stream")
        # Assert
        assert response.headers.get("cache-control") == "no-cache"


# ============================================================
# ⏹️ DELETE /api/v1/task/{task_id} Tests
# ============================================================


class TestCancelTask:
    """Tests for DELETE /api/v1/task/{task_id} endpoint ⏹️."""

    @pytest.mark.asyncio
    async def test_cancel_running_task_returns_200(
        self,
        async_client,
        test_task_store: dict[str, Any],
        mock_task_runner: MagicMock,
    ) -> None:
        """Cancelling a running task should return 200."""
        # Arrange: 🔧 Manually create RUNNING task (avoids background race)
        from inquiro.api.router import TaskState
        from inquiro.infrastructure.event_emitter import EventEmitter

        task_state = TaskState(
            task_id="cancel-running-001",
            task_type=TaskType.RESEARCH,
            event_emitter=EventEmitter(),
        )
        task_state.status = TaskStatus.RUNNING
        test_task_store["cancel-running-001"] = task_state

        # Act
        response = await async_client.delete("/api/v1/task/cancel-running-001")
        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cancelled"
        assert data["reason"] == "user_requested"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task_returns_404(
        self,
        async_client,
    ) -> None:
        """Cancelling a non-existent task should return 404."""
        # Arrange / Act
        response = await async_client.delete("/api/v1/task/nonexistent-task")
        # Assert
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cancel_completed_task_returns_409(
        self,
        async_client,
        test_task_store: dict[str, Any],
    ) -> None:
        """Cancelling a completed task should return 409 Conflict."""
        # Arrange: 🔧 Create task with COMPLETED status
        from inquiro.api.router import TaskState
        from inquiro.infrastructure.event_emitter import EventEmitter

        task_state = TaskState(
            task_id="cancel-completed-001",
            task_type=TaskType.RESEARCH,
            event_emitter=EventEmitter(),
        )
        task_state.status = TaskStatus.COMPLETED
        test_task_store["cancel-completed-001"] = task_state

        # Act
        response = await async_client.delete("/api/v1/task/cancel-completed-001")
        # Assert
        assert response.status_code == 409

    @pytest.mark.asyncio
    async def test_cancel_already_cancelled_task_returns_409(
        self,
        async_client,
        test_task_store: dict[str, Any],
    ) -> None:
        """Cancelling an already cancelled task should return 409."""
        # Arrange: 🔧 Create task with CANCELLED status
        from inquiro.api.router import TaskState
        from inquiro.infrastructure.event_emitter import EventEmitter

        task_state = TaskState(
            task_id="cancel-cancelled-001",
            task_type=TaskType.RESEARCH,
            event_emitter=EventEmitter(),
        )
        task_state.status = TaskStatus.CANCELLED
        test_task_store["cancel-cancelled-001"] = task_state

        # Act
        response = await async_client.delete("/api/v1/task/cancel-cancelled-001")
        # Assert
        assert response.status_code == 409


# ============================================================
# ❤️ GET /api/v1/health Tests
# ============================================================


class TestHealthCheck:
    """Tests for GET /api/v1/health endpoint ❤️."""

    @pytest.mark.asyncio
    async def test_health_returns_200(
        self,
        async_client,
    ) -> None:
        """Health check should return 200."""
        # Arrange / Act
        response = await async_client.get("/api/v1/health")
        # Assert
        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_health_includes_version(
        self,
        async_client,
    ) -> None:
        """Health response should include engine version."""
        # Arrange / Act
        response = await async_client.get("/api/v1/health")
        # Assert
        data = response.json()
        assert data["version"] == "2.0.0"

    @pytest.mark.asyncio
    async def test_health_includes_capabilities(
        self,
        async_client,
    ) -> None:
        """Health response should list research and synthesis capabilities."""
        # Arrange / Act
        response = await async_client.get("/api/v1/health")
        # Assert
        data = response.json()
        assert "research" in data["capabilities"]
        assert "synthesis" in data["capabilities"]

    @pytest.mark.asyncio
    async def test_health_includes_mcp_status(
        self,
        async_client,
    ) -> None:
        """Health response should include MCP server connectivity."""
        # Arrange / Act
        response = await async_client.get("/api/v1/health")
        # Assert
        data = response.json()
        assert "mcp_servers" in data
        assert data["mcp_servers"]["perplexity"] == "connected"

    @pytest.mark.asyncio
    async def test_health_includes_active_tasks(
        self,
        async_client,
    ) -> None:
        """Health response should include active task count."""
        # Arrange / Act
        response = await async_client.get("/api/v1/health")
        # Assert
        data = response.json()
        assert "active_tasks" in data
        assert isinstance(data["active_tasks"], int)


# ============================================================
# ❌ Error Handling Tests
# ============================================================


class TestErrorHandling:
    """Tests for API error handling patterns ❌."""

    @pytest.mark.asyncio
    async def test_validation_error_returns_422(
        self,
        async_client,
    ) -> None:
        """Pydantic validation errors should return 422."""
        # Arrange: Invalid payload
        # Act
        response = await async_client.post(
            "/api/v1/research",
            json={"task_id": 123},  # task_id should be string
        )
        # Assert
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_error_response_format(
        self,
        async_client,
    ) -> None:
        """Error responses should follow ErrorResponse schema."""
        # Arrange / Act: 🔍 Request non-existent task to trigger 404
        response = await async_client.get("/api/v1/task/nonexistent-error-test")
        # Assert: ✅ HTTPException wraps our ErrorResponse in 'detail'
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        detail = data["detail"]
        assert detail["code"] == "task_not_found"
        assert "message" in detail

    @pytest.mark.asyncio
    async def test_unknown_endpoint_returns_404(
        self,
        async_client,
    ) -> None:
        """Unknown endpoint should return 404 Not Found."""
        # Arrange / Act
        response = await async_client.get("/api/v1/unknown")
        # Assert
        assert response.status_code in (404, 405)
