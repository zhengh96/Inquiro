"""Inquiro API layer — FastAPI service endpoints ✨.

Provides REST API for the Inquiro evidence research & synthesis engine.
Endpoints: POST /research, POST /synthesize, GET /task/{id},
GET /task/{id}/stream (SSE), DELETE /task/{id}, GET /health.
"""

from inquiro.api.app import create_app
from inquiro.api.router import router
from inquiro.api.schemas import (
    ResearchRequest,
    SynthesizeRequest,
    TaskResponse,
    TaskSubmitResponse,
    TaskCancelResponse,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    "create_app",
    "router",
    "ResearchRequest",
    "SynthesizeRequest",
    "TaskResponse",
    "TaskSubmitResponse",
    "TaskCancelResponse",
    "HealthResponse",
    "ErrorResponse",
]
