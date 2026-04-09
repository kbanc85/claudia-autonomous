"""Claudia hybrid memory provider plugin.

Phase 2A.2 + 2A.3 complete. Exports:

- schema: SQLite DDL and migration runner (sub-task 2A.2a)
- embeddings: Ollama client for all-minilm:l6-v2 (sub-task 2A.2b)
- hybrid_search: 50/25/10/15 ranking formula (sub-task 2A.2c)
- entities: entity and relationship CRUD (sub-task 2A.2d)
- offline: three-tier degradation router (sub-task 2A.2e)
- provider: ClaudiaMemoryProvider ABC subclass (sub-task 2A.2f)
- writer: Serialized WriterQueue for concurrent-safe writes (Phase 2A.3)
- reader: Bounded ReaderPool for concurrent reads (Phase 2A.3)

The module-level ``register(ctx)`` function is the entry point
the memory manager calls when loading this plugin. It constructs
a ``ClaudiaMemoryProvider`` and registers it alongside the
always-on built-in memory provider.

Design reference: docs/decisions/memory-provider-design.md
"""

from plugins.memory.claudia import (  # noqa: F401
    embeddings,
    entities,
    hybrid_search,
    offline,
    provider,
    reader,
    schema,
    writer,
)
from plugins.memory.claudia.provider import ClaudiaMemoryProvider

__version__ = "0.1.0-dev"


def register(ctx) -> None:
    """Register Claudia's hybrid memory provider with the memory manager.

    Called by the plugin loader during agent startup. The memory
    manager owns the returned provider's lifecycle (initialize,
    prefetch, sync_turn, shutdown).
    """
    ctx.register_memory_provider(ClaudiaMemoryProvider())
