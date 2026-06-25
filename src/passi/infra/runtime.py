"""Runtime dependency injection container.

Groups all dependencies (config, LLM clients, session, background tasks) into a
single injectable container — inspired by Kimi CLI's Runtime pattern.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from passi.config import PassiConfig
    from passi.infra.llm_client import LLMClient
    from passi.infra.session import SessionManager
    from passi.infra.context import ContextManager

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    """Dependency injection container holding all shared services.

    Services are initialized lazily on first access.
    """

    config: "PassiConfig"

    # Lazy-loaded services
    _llm_clients: dict[str, "LLMClient"] = field(default_factory=dict)
    _session_manager: "SessionManager | None" = field(default=None, init=False)
    _context_manager: "ContextManager | None" = field(default=None, init=False)
    _background_tasks: set[asyncio.Task[Any]] = field(default_factory=set)
    _initialized: bool = field(default=False, init=False)

    async def initialize(self) -> None:
        """Pre-initialize all services."""
        if self._initialized:
            return
        self._initialized = True
        logger.info("Runtime initialized.")

    async def shutdown(self) -> None:
        """Cancel background tasks and clean up."""
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        self._background_tasks.clear()
        self._initialized = False
        logger.info("Runtime shutdown complete.")

    def create_task(self, coro: Any) -> asyncio.Task:
        """Create a tracked background task."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @property
    def session(self) -> "SessionManager":
        if self._session_manager is None:
            from passi.infra.session import SessionManager

            self._session_manager = SessionManager(self.config)
        return self._session_manager

    @property
    def context(self) -> "ContextManager":
        if self._context_manager is None:
            from passi.infra.context import ContextManager

            self._context_manager = ContextManager(self.config)
        return self._context_manager

    def get_llm_client(self, provider: str | None = None) -> "LLMClient":
        """Get or create an LLM client for the given provider."""
        provider = provider or self.config.default_provider
        if provider not in self._llm_clients:
            from passi.infra.llm_client import create_llm_client

            self._llm_clients[provider] = create_llm_client(self.config, provider)
        return self._llm_clients[provider]
