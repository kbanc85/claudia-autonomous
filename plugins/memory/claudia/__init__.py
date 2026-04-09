"""Claudia Hybrid Memory — local-first, trust-aware memory provider.

Author: Kamil Banc (https://github.com/kbanc85)
Part of: claudia-autonomous (https://github.com/kbanc85/claudia-autonomous)
License: MIT

A ``MemoryProvider`` plugin for the claudia-autonomous fork of
Hermes Agent v0.7.0. SQLite + vector embeddings via Ollama, with
trust-aware hybrid ranking, LLM entity extraction, hybrid pattern
+ LLM commitment detection, fuzzy consolidation, confidence decay,
and 18 ``memory.*`` tools covering read / write / state mutation
/ delete operations.

Runs 100% locally. No API keys, no cloud services. Degrades
gracefully when Ollama is unreachable (recall still works via
FTS + importance + recency ranking).

Modules:

    schema                — SQLite DDL, migrations, WAL, FTS5
    embeddings            — Ollama embedding client
    hybrid_search         — Trust-weighted hybrid ranking
    entities              — Entity and relationship CRUD
    offline               — Three-tier degradation router
    writer                — Serialized writer queue
    reader                — Bounded reader pool
    provider              — ClaudiaMemoryProvider ABC implementation
    extractor             — LLM entity extraction (Ollama-backed)
    commitments           — Commitment CRUD
    commitment_detector   — Hybrid pattern + LLM commitment detection
    consolidation         — Fuzzy entity dedup + FK resolution
    verification          — Confidence decay + stale flagging
    budget                — Token budget estimation and truncation
    retention             — Permanent removal of old soft-deletes
    metrics               — Thread-safe internal counters

Module-level ``register(ctx)`` is the entry point the fork's
memory manager calls during agent startup.

Documentation:
    README.md                  — reference docs
    CHANGELOG.md               — v1.0 release notes
    docs/getting-started.md    — 10-step first-time user tutorial
    demo.py                    — runnable scripted walkthrough
"""

from plugins.memory.claudia import (  # noqa: F401
    budget,
    commitment_detector,
    commitments,
    consolidation,
    embeddings,
    entities,
    extractor,
    hybrid_search,
    metrics,
    offline,
    provider,
    reader,
    retention,
    schema,
    verification,
    writer,
)
from plugins.memory.claudia.provider import ClaudiaMemoryProvider

__version__ = "1.0.0"
__author__ = "Kamil Banc"
__license__ = "MIT"


def register(ctx) -> None:
    """Register Claudia's hybrid memory provider with the memory manager.

    Called by the plugin loader during agent startup. The memory
    manager owns the returned provider's lifecycle (initialize,
    prefetch, sync_turn, shutdown).
    """
    ctx.register_memory_provider(ClaudiaMemoryProvider())
