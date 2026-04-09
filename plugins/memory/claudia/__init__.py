"""Claudia hybrid memory provider plugin.

Phase 2A.2 work in progress. Currently exports:

- schema: SQLite DDL and migration runner (sub-task 2A.2a — done)
- embeddings: Ollama client for all-minilm:l6-v2 (sub-task 2A.2b — done)
- hybrid_search: 50/25/10/15 ranking formula (sub-task 2A.2c — done)
- entities: entity and relationship CRUD (sub-task 2A.2d — done)
- offline: three-tier degradation router (sub-task 2A.2e — done)

Planned exports once Phase 2A.2 completes:

- provider.ClaudiaMemoryProvider: the MemoryProvider ABC subclass
  (sub-task 2A.2f — the final wiring step that registers this
  plugin with the memory manager)

Until the provider registration sub-task (2A.2f) lands, this plugin
is not activated by the memory manager — importing this package has
no runtime side effects beyond making the five implementation
modules available.

Design reference: docs/decisions/memory-provider-design.md
"""

from plugins.memory.claudia import (  # noqa: F401
    embeddings,
    entities,
    hybrid_search,
    offline,
    schema,
)

__version__ = "0.1.0-dev"
