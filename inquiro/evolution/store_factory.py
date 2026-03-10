"""ExperienceStore factory — lazy-initialized singleton for evolution persistence 🏭.

Provides a module-level async factory that creates and caches an
ExperienceStore instance using the ``EVOLUTION_DB_URL`` environment variable.

This decouples SearchExp and SynthesisExp from needing a pre-built
session factory in the evolution_profile dict. Instead, they call
``get_store()`` which lazily initializes the database connection on first use.

Usage::

    from inquiro.evolution.store_factory import get_store

    store = await get_store()  # Lazy init from EVOLUTION_DB_URL
    experiences = await store.query(query)

Environment Variables:
    EVOLUTION_DB_URL: Database URL for the evolution store.
        Default: ``sqlite+aiosqlite:///evolution_store.db``
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# 🔒 Module-level singleton state
# Uses asyncio.Lock to avoid blocking the event loop thread.
# IMPORTANT: threading.Lock caused deadlocks when multiple coroutines
# called get_store() concurrently — the second caller blocked the
# thread, preventing the first caller from completing its await.
_store_instance: Any = None
_init_lock = asyncio.Lock()


async def get_store():
    """Get or create the ExperienceStore singleton 🏭.

    Lazily initializes the database engine, schema, and session factory
    on first call. Subsequent calls return the cached instance.

    The database URL is read from the ``EVOLUTION_DB_URL`` environment
    variable. If not set, defaults to a local SQLite database.

    Concurrency-safe: uses ``asyncio.Lock`` to serialize initialization
    without blocking the event loop thread.

    Returns:
        ExperienceStore instance ready for use.

    Raises:
        RuntimeError: If database initialization fails.
    """
    global _store_instance

    if _store_instance is not None:
        return _store_instance

    async with _init_lock:
        # 🔍 Double-check after acquiring lock
        if _store_instance is not None:
            return _store_instance

        from sqlalchemy.ext.asyncio import (
            async_sessionmaker,
            create_async_engine,
        )

        from inquiro.evolution.store import ExperienceStore, init_store_schema

        # 🔧 Read DB URL from environment
        db_url = os.environ.get(
            "EVOLUTION_DB_URL",
            "sqlite+aiosqlite:///evolution_store.db",
        )

        try:
            # 🏗️ Create engine with connection pool
            engine_kwargs: dict[str, Any] = {"echo": False}
            if not db_url.startswith("sqlite"):
                # 📊 Connection pool settings for non-SQLite databases
                engine_kwargs["pool_size"] = 5
                engine_kwargs["max_overflow"] = 10

            engine = create_async_engine(db_url, **engine_kwargs)

            # 🛠️ Initialize schema (create tables if needed)
            await init_store_schema(engine)

            # 💾 Create session factory and store
            session_factory = async_sessionmaker(engine, expire_on_commit=False)
            _store_instance = ExperienceStore(session_factory)

            logger.info("🏭 ExperienceStore initialized: %s", db_url)
            return _store_instance

        except Exception as e:
            logger.error("❌ Failed to initialize ExperienceStore: %s", e)
            raise RuntimeError(f"ExperienceStore initialization failed: {e}") from e


async def reset_store() -> None:
    """Reset the singleton store instance (for testing) 🧪.

    Clears the cached store instance so the next ``get_store()``
    call will create a fresh one. Does NOT close the database
    connection — caller is responsible for cleanup.
    """
    global _store_instance
    async with _init_lock:
        _store_instance = None
    logger.debug("🧪 ExperienceStore singleton reset")
