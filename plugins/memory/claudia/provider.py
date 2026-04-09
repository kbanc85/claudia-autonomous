"""ClaudiaMemoryProvider — MemoryProvider ABC implementation.

Wires the seven implementation modules (schema, embeddings,
hybrid_search, entities, offline, writer, reader) into a single
class that satisfies ``agent/memory_provider.MemoryProvider`` and
registers with the plugin system via ``register(ctx)``.

Layered design (Phase 2A.3 concurrent update):

1. ``schema.ensure_database(db_path)`` — applies migrations on a
   one-shot connection during ``initialize``. The writer and reader
   pool each open their own connections after migrations are done.
2. ``writer.WriterQueue`` — single background thread owning one
   sqlite3 connection. All writes go through its enqueue path.
3. ``reader.ReaderPool`` — bounded pool of N reader connections
   (default 4) for concurrent prefetch / memory.recall / memory.about
   calls. Uses ``check_same_thread=False`` so connections travel
   between threads.
4. ``embeddings.OllamaEmbedder`` — lazy Ollama client. Probed on
   first embed call, cached thereafter. Embeddings are generated
   BEFORE enqueueing the write job so HTTP latency doesn't stall
   the writer thread.
5. ``offline.OfflineRouter(embedder)`` — decides which
   ``HybridWeights`` preset to use each query.
6. ``hybrid_search.search`` — called via ``router.search`` for the
   50/25/10/15 composite ranking.
7. ``entities.*`` — entity CRUD for ``memory.about``.

Thread safety:

- ``sync_turn`` is non-blocking: it generates the embedding on the
  caller thread, then enqueues a fire-and-forget job into the
  writer queue.
- ``memory.remember`` is blocking: the caller expects the new row
  id in the JSON response, so the provider enqueues a job and
  waits on a marker event until the writer processes it.
- ``prefetch``, ``memory.recall``, and ``memory.about`` run on the
  caller's thread but acquire a reader connection from the pool
  for each call, allowing N concurrent readers without cursor
  contention.
- ``system_prompt_block`` also uses the reader pool.
- ``shutdown()`` stops the writer (draining pending jobs) and
  closes the reader pool.

Profile resolution priority: ``user_id`` > ``agent_identity`` >
``agent_workspace`` > ``"default"``. Gateway sessions get
per-user memory isolation automatically.

agent_context filtering: writes are skipped entirely for
non-primary contexts (cron, subagent, flush). Reads are always
allowed — a subagent can recall from the parent's memory, it
just can't write back.

Testability:

The embedder, router, writer, and reader pool are constructed via
``_make_*`` factory methods. Tests subclass
``ClaudiaMemoryProvider`` and override these to inject scripted
fakes without touching httpx or spawning real threads.

Reference: docs/decisions/memory-provider-design.md (Phase 2A.3)
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from plugins.memory.claudia import (
    commitments as commitments_module,
    consolidation,
    entities,
    hybrid_search,
    retention,
    schema,
    verification,
)
from plugins.memory.claudia.budget import (
    BudgetDecision,
    BudgetState,
    decide_budget,
    estimate_tokens,
    truncate_to_budget,
    update_budget_state,
)
from plugins.memory.claudia.commitment_detector import (
    CommitmentDetector,
    DetectedCommitment,
    HybridCommitmentDetector,
)
from plugins.memory.claudia.consolidation import ConsolidationResult
from plugins.memory.claudia.retention import RetentionResult
from plugins.memory.claudia.verification import VerificationResult
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.extractor import (
    ExtractedEntity,
    LLMExtractor,
    OllamaLLMExtractor,
)
from plugins.memory.claudia.offline import MemoryMode, OfflineRouter
from plugins.memory.claudia.reader import DEFAULT_POOL_SIZE, ReaderPool
from plugins.memory.claudia.writer import WriterQueue

logger = logging.getLogger(__name__)


# ─── Tool schemas ────────────────────────────────────────────────────────
#
# OpenAI function-calling format, per the ABC contract. The agent
# loop converts between this and Anthropic's tool_use format as
# needed — see builtin_memory_provider.py for the reference shape.


MEMORY_RECALL_SCHEMA: Dict[str, Any] = {
    "name": "memory.recall",
    "description": (
        "Search Claudia's long-term memory for relevant context using "
        "hybrid ranking (semantic similarity, importance, recency, and "
        "full-text search). Returns the top memories as compact bullets "
        "with their source provenance. Use this to answer questions "
        "about the user's past work, relationships, and decisions."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for. Natural language works.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return. Default 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


MEMORY_REMEMBER_SCHEMA: Dict[str, Any] = {
    "name": "memory.remember",
    "description": (
        "Store a single fact in Claudia's long-term memory. Use this when "
        "the user explicitly asks you to remember something, or when you "
        "notice a fact worth preserving (preferences, commitments, "
        "relationships, decisions). Returns a confirmation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to remember. One sentence is ideal.",
            },
            "importance": {
                "type": "number",
                "description": "How important is this fact? 0.0 to 1.0. Default 0.5.",
                "default": 0.5,
            },
            "source_type": {
                "type": "string",
                "description": (
                    "Provenance tag: 'conversation', 'transcript', 'gmail', "
                    "'upload', or 'capture'. Default 'conversation'."
                ),
                "default": "conversation",
            },
        },
        "required": ["content"],
    },
}


MEMORY_ABOUT_SCHEMA: Dict[str, Any] = {
    "name": "memory.about",
    "description": (
        "Look up everything Claudia knows about a person, organization, "
        "project, location, or concept. Returns the entity record plus "
        "relationships. Pass a name or alias; the lookup is case-"
        "insensitive and checks both the canonical name and any stored "
        "aliases."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name or alias of the entity to look up.",
            },
            "kind": {
                "type": "string",
                "enum": ["person", "organization", "project", "location", "concept"],
                "description": "Optional filter to disambiguate same-name entities.",
            },
        },
        "required": ["name"],
    },
}


MEMORY_COMMITMENTS_SCHEMA: Dict[str, Any] = {
    "name": "memory.commitments",
    "description": (
        "List commitments Claudia has tracked from recent "
        "conversations. Use this when the user asks 'what did I "
        "promise?', 'what do I owe?', or 'what's outstanding?'. "
        "Returns a list ordered by deadline (NULLs last), each "
        "with id, content, deadline, and status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": [
                    "open", "completed", "overdue",
                    "dropped", "superseded", "all",
                ],
                "description": (
                    "Filter by status. Default 'open'. Pass 'all' "
                    "to return commitments regardless of status."
                ),
                "default": "open",
            },
            "limit": {
                "type": "integer",
                "description": "Max commitments to return. Default 50.",
                "default": 50,
            },
        },
    },
}


MEMORY_STATUS_SCHEMA: Dict[str, Any] = {
    "name": "memory.status",
    "description": (
        "Get Claudia's memory system status: counts per table, "
        "current offline mode, and session id. Useful for "
        "diagnostics ('how many memories do you have?') and for "
        "choosing between local and detailed recall."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


MEMORY_COMMITMENT_COMPLETE_SCHEMA: Dict[str, Any] = {
    "name": "memory.commitment_complete",
    "description": (
        "Mark a tracked commitment as completed. Use this when "
        "the user confirms they finished the task (e.g. 'I sent "
        "the proposal'). Pass the commitment id from a previous "
        "memory.commitments call. Returns the updated commitment "
        "with status='completed' and a completed_at timestamp."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": (
                    "The commitment id from memory.commitments."
                ),
            },
        },
        "required": ["id"],
    },
}


MEMORY_COMMITMENT_DROP_SCHEMA: Dict[str, Any] = {
    "name": "memory.commitment_drop",
    "description": (
        "Mark a tracked commitment as dropped (the user explicitly "
        "abandoning it, not completing it). Use this when the user "
        "says 'never mind about X' or 'forget that promise'. "
        "Returns the updated commitment with status='dropped'."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": (
                    "The commitment id from memory.commitments."
                ),
            },
        },
        "required": ["id"],
    },
}


MEMORY_VERIFY_MEMORY_SCHEMA: Dict[str, Any] = {
    "name": "memory.verify_memory",
    "description": (
        "Mark a memory as explicitly verified. Use this when the "
        "user confirms a fact Claudia extracted is correct ('yes, "
        "that's right'). Verified memories get the full trust "
        "weight in recall ranking and are exempt from stale-flagging."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The memory row id.",
            },
        },
        "required": ["id"],
    },
}


MEMORY_FLAG_MEMORY_SCHEMA: Dict[str, Any] = {
    "name": "memory.flag_memory",
    "description": (
        "Mark a memory as flagged for review. Use this when a "
        "memory looks suspicious or needs user attention but isn't "
        "definitively wrong. Flagged memories rank lower in recall "
        "(~0.5× multiplier) and are visible as a distinct status."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The memory row id.",
            },
        },
        "required": ["id"],
    },
}


MEMORY_CONTRADICTS_MEMORY_SCHEMA: Dict[str, Any] = {
    "name": "memory.contradicts_memory",
    "description": (
        "Mark a memory as contradicting another known memory. Use "
        "this when the user explicitly corrects a fact ('no, that's "
        "wrong, it's actually X'). Contradicted memories get a "
        "heavy demotion (~0.3× multiplier) in recall ranking."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The memory row id.",
            },
        },
        "required": ["id"],
    },
}


MEMORY_FORGET_ENTITY_SCHEMA: Dict[str, Any] = {
    "name": "memory.forget_entity",
    "description": (
        "Soft-delete an entity by name (case-insensitive). The "
        "entity row stays in the DB with deleted_at set but is "
        "excluded from lookups and recall ranking. Relationships "
        "that reference it are NOT cascaded — they become orphan "
        "references that retention purge will clean up later. "
        "Use when the user explicitly asks to remove a person, "
        "organization, or project from Claudia's knowledge."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Entity name (case-insensitive).",
            },
        },
        "required": ["name"],
    },
}


MEMORY_FORGET_MEMORY_SCHEMA: Dict[str, Any] = {
    "name": "memory.forget_memory",
    "description": (
        "Soft-delete a specific memory by id. The row stays in "
        "the DB with deleted_at set (so trace/audit queries can "
        "still see it) but is excluded from every recall path. "
        "Use when the user explicitly asks to remove a memory "
        "('forget what I said about X'). For permanent removal, "
        "wait for the retention purge to sweep it out."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The memory row id to forget.",
            },
        },
        "required": ["id"],
    },
}


MEMORY_TRACE_SCHEMA: Dict[str, Any] = {
    "name": "memory.trace",
    "description": (
        "Walk the correction chain for a memory. Returns every "
        "version of the fact — the original extracted memory, "
        "every subsequent correction, and the current authoritative "
        "version — ordered oldest-first. Used for provenance "
        "audits: 'why does Claudia think X?' shows the timeline of "
        "how the fact evolved."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": (
                    "Any memory id in the chain. The result is the "
                    "same regardless of which version you pass."
                ),
            },
        },
        "required": ["id"],
    },
}


MEMORY_SEARCH_ENTITIES_SCHEMA: Dict[str, Any] = {
    "name": "memory.search_entities",
    "description": (
        "Search entities by substring match on name or any alias. "
        "Complements memory.about (exact lookup) for queries like "
        "'who do I know at Acme' or 'list people named Sarah'. "
        "Use '*' as the query to return every entity in the profile."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Substring to match on name or aliases. Pass "
                    "'*' to return everything (wildcard)."
                ),
            },
            "kind": {
                "type": "string",
                "enum": ["person", "organization", "project", "location", "concept"],
                "description": "Optional filter to one entity kind.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results. Default 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}


MEMORY_RELATE_SCHEMA: Dict[str, Any] = {
    "name": "memory.relate",
    "description": (
        "Create a directed relationship between two existing "
        "entities. Use this when the user explicitly states a "
        "connection ('Sarah works at Acme', 'this project belongs "
        "to Q4', 'Bob reports to Alice'). Both entities must exist; "
        "if one doesn't, create it first via the extraction pipeline "
        "or a memory.remember with the entity name in the content. "
        "Returns the created relationship."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "from_name": {
                "type": "string",
                "description": "Source entity name (case-insensitive).",
            },
            "to_name": {
                "type": "string",
                "description": "Target entity name (case-insensitive).",
            },
            "type": {
                "type": "string",
                "description": (
                    "Relationship type (free-form: 'works_at', "
                    "'knows', 'reports_to', 'belongs_to', etc.)."
                ),
            },
            "notes": {
                "type": "string",
                "description": "Optional context about the relationship.",
            },
        },
        "required": ["from_name", "to_name", "type"],
    },
}


MEMORY_CORRECT_MEMORY_SCHEMA: Dict[str, Any] = {
    "name": "memory.correct_memory",
    "description": (
        "Replace the content of a stored memory with a corrected "
        "version. Creates a NEW memory row with the new content "
        "(origin='corrected', verification='verified', confidence=1.0) "
        "that references the old row via corrected_from. Marks the "
        "old row as contradicts so it ranks lower in recall while "
        "remaining visible in provenance audits. Use this when the "
        "user explicitly corrects a fact Claudia extracted."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The memory row id to correct.",
            },
            "new_content": {
                "type": "string",
                "description": (
                    "The corrected content. Must be non-empty."
                ),
            },
        },
        "required": ["id", "new_content"],
    },
}


_ALL_TOOL_SCHEMAS = [
    MEMORY_RECALL_SCHEMA,
    MEMORY_REMEMBER_SCHEMA,
    MEMORY_ABOUT_SCHEMA,
    MEMORY_COMMITMENTS_SCHEMA,
    MEMORY_STATUS_SCHEMA,
    MEMORY_COMMITMENT_COMPLETE_SCHEMA,
    MEMORY_COMMITMENT_DROP_SCHEMA,
    MEMORY_VERIFY_MEMORY_SCHEMA,
    MEMORY_FLAG_MEMORY_SCHEMA,
    MEMORY_CONTRADICTS_MEMORY_SCHEMA,
    MEMORY_CORRECT_MEMORY_SCHEMA,
    MEMORY_RELATE_SCHEMA,
    MEMORY_SEARCH_ENTITIES_SCHEMA,
    MEMORY_TRACE_SCHEMA,
    MEMORY_FORGET_MEMORY_SCHEMA,
    MEMORY_FORGET_ENTITY_SCHEMA,
]


# SQL for inserting a memory row. Shared between sync and async
# insert paths so there's one source of truth for the column order.
_INSERT_MEMORY_SQL = """
INSERT INTO memories (
    content, origin, confidence, importance, access_count,
    embedding, embedding_dim, source_type, source_ref,
    profile, created_at, accessed_at
) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
"""


# ─── Provider implementation ────────────────────────────────────────────


class ClaudiaMemoryProvider(MemoryProvider):
    """Claudia's hybrid memory provider plugin."""

    def __init__(self) -> None:
        self._writer: Optional[WriterQueue] = None
        self._reader_pool: Optional[ReaderPool] = None
        self._db_path: Optional[Path] = None
        self._embedder: Optional[OllamaEmbedder] = None
        self._router: Optional[OfflineRouter] = None
        self._session_id: Optional[str] = None
        self._profile: str = "default"
        self._agent_context: str = "primary"
        self._claudia_home: Optional[Path] = None

        # Phase 2C.3: loaded from memory/claudia/config.json on
        # initialize(). Empty dict when the file doesn't exist.
        # Factory methods and verify/consolidate read values from
        # here with sensible fallbacks.
        self._config: Dict[str, Any] = {}

        # Phase 2B cognitive pipeline: LLM entity extraction (2B.1) +
        # commitment detection (2B.2). One dedicated single-worker
        # executor shared by both tasks, because both have the same
        # latency profile (slow LLM calls) and the same non-blocking
        # requirement. The pool stays off the caller thread (invariant
        # #3: sync_turn non-blocking) AND off the writer thread (so
        # slow LLM calls don't stall queued writes). Pending futures
        # from BOTH pipelines are tracked under one lock so
        # flush()/shutdown() can drain everything together before
        # closing the writer.
        self._extractor: Optional[LLMExtractor] = None
        self._commitment_detector: Optional[CommitmentDetector] = None
        self._cognitive_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
        self._pending_cognitive_futures: List[concurrent.futures.Future] = []
        self._cognitive_lock = threading.Lock()

        # Phase 2B.5: cost governance state. ``on_turn_start``
        # updates this via update_budget_state. Every other hook
        # reads it and calls decide_budget() for a fresh decision.
        self._budget_state: BudgetState = BudgetState()

    # ── Required ABC members ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "claudia"

    def is_available(self) -> bool:
        """Always available. Uses only stdlib + sqlite3 at this layer.

        Ollama is optional (the plugin runs in FTS_IMP_REC mode when
        the daemon is unreachable), sqlite-vec is optional (my
        implementation uses pure-Python cosine), and httpx is already
        a fork-level dependency. There is no runtime precondition
        this provider cannot satisfy, so availability is unconditional.
        """
        return True

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Open the DB, apply migrations, start writer + reader pool.

        Required kwargs per the ABC: ``claudia_home``, ``platform``.
        Optional: ``agent_context``, ``agent_identity``,
        ``agent_workspace``, ``parent_session_id``, ``user_id``.
        """
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context", "primary")

        self._claudia_home = self._resolve_claudia_home(kwargs)
        self._profile = self._resolve_profile(kwargs)

        self._db_path = self._claudia_home / "memory" / "claudia" / "claudia.db"

        # Phase 2C.3: load user config BEFORE constructing factories
        # so _make_extractor / _make_commitment_detector can read it.
        self._config = self._load_config()

        # Apply migrations via a one-shot connection. The writer thread
        # and reader pool will each open their own connections after
        # this returns, so they never race on migration state.
        init_conn = schema.ensure_database(self._db_path)
        init_conn.close()

        self._embedder = self._make_embedder()
        self._router = self._make_router(self._embedder)

        self._writer = self._make_writer()
        self._writer.start()

        self._reader_pool = self._make_reader_pool()

        self._extractor = self._make_extractor()
        self._commitment_detector = self._make_commitment_detector()
        self._cognitive_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="claudia-cognitive",
        )

        logger.info(
            "Claudia memory provider initialized: session=%s profile=%s "
            "context=%s db=%s",
            session_id,
            self._profile,
            self._agent_context,
            self._db_path,
        )

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return list(_ALL_TOOL_SCHEMAS)

    # ── Factories (overridable for tests) ─────────────────────────────

    def _make_embedder(self) -> OllamaEmbedder:
        """Construct the embedder. Tests override to inject scripted fakes."""
        return OllamaEmbedder()

    def _make_router(self, embedder: OllamaEmbedder) -> OfflineRouter:
        """Construct the router. Tests override for custom presets or force modes."""
        return OfflineRouter(embedder)

    def _make_writer(self) -> WriterQueue:
        """Construct the writer queue. Tests can override for synchronous fakes."""
        assert self._db_path is not None
        return WriterQueue(self._db_path)

    def _make_reader_pool(self) -> ReaderPool:
        """Construct the reader pool. Tests can override pool size."""
        assert self._db_path is not None
        return ReaderPool(self._db_path, size=DEFAULT_POOL_SIZE)

    def _make_extractor(self) -> LLMExtractor:
        """Construct the LLM entity extractor (Phase 2B.1).

        Tests override this to inject a deterministic fake that
        returns scripted ``ExtractedEntity`` lists without touching
        the network. Real sessions get ``OllamaLLMExtractor``, which
        probes lazily on first use per invariant #1.

        Phase 2C.3: respects ``extraction_model`` and ``ollama_host``
        from ``self._config`` if set; otherwise uses the extractor's
        own defaults (qwen2.5:3b / http://localhost:11434).
        """
        kwargs: Dict[str, Any] = {}
        model = self._config.get("extraction_model")
        if model:
            kwargs["model"] = model
        host = self._config.get("ollama_host")
        if host:
            kwargs["host"] = host
        return OllamaLLMExtractor(**kwargs)

    def _make_commitment_detector(self) -> CommitmentDetector:
        """Construct the commitment detector (Phase 2B.2).

        Default is ``HybridCommitmentDetector`` which combines a
        regex pre-filter with optional LLM refinement. Tests override
        this to inject a scripted fake. Offline sessions get pattern
        output only; sessions with a running Ollama daemon get LLM
        refinement.

        Phase 2C.3: respects ``detection_model`` and ``ollama_host``
        from ``self._config`` if set.
        """
        kwargs: Dict[str, Any] = {}
        model = self._config.get("detection_model")
        if model:
            kwargs["model"] = model
        host = self._config.get("ollama_host")
        if host:
            kwargs["host"] = host
        return HybridCommitmentDetector(**kwargs)

    # ── Config (Phase 2C.3) ───────────────────────────────────────────

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Return the user-facing config fields for ``claudia memory setup``.

        All fields are non-secret (Claudia is local-first; there are
        no remote API keys to manage). Env vars are listed where
        existing modules honor them so the user can override without
        editing the config file.
        """
        return [
            {
                "key": "ollama_host",
                "description": (
                    "Ollama daemon URL. Default: http://localhost:11434. "
                    "Override for a shared or remote Ollama instance."
                ),
                "default": "http://localhost:11434",
                "env_var": "CLAUDIA_OLLAMA_HOST",
            },
            {
                "key": "embedding_model",
                "description": (
                    "Ollama model for memory embeddings. Default: "
                    "all-minilm:l6-v2 (384 dim, fast on CPU)."
                ),
                "default": "all-minilm:l6-v2",
            },
            {
                "key": "extraction_model",
                "description": (
                    "Ollama model for entity extraction (Phase 2B.1). "
                    "Default: qwen2.5:3b (small, fast, good enough)."
                ),
                "default": "qwen2.5:3b",
            },
            {
                "key": "detection_model",
                "description": (
                    "Ollama model for commitment detection (Phase 2B.2). "
                    "Default: qwen2.5:3b."
                ),
                "default": "qwen2.5:3b",
            },
            {
                "key": "decay_half_life_days",
                "description": (
                    "Memory confidence decay half-life (Phase 2B.4). "
                    "Memories accessed this long ago see their "
                    "confidence halved. Default: 30."
                ),
                "default": 30,
            },
            {
                "key": "stale_flag_days",
                "description": (
                    "Days of inactivity before auto-flagging pending "
                    "memories (Phase 2B.4). Default: 60."
                ),
                "default": 60,
            },
            {
                "key": "auto_merge_threshold",
                "description": (
                    "Fuzzy-match threshold for auto-merging entity "
                    "duplicates in consolidation (Phase 2B.3). "
                    "Higher = more conservative. Default: 0.92."
                ),
                "default": 0.92,
            },
            {
                "key": "retention_days",
                "description": (
                    "Days before soft-deleted rows are permanently "
                    "purged (Phase 2C.4). Only ``purge_old_soft_deletes`` "
                    "honors this; consolidation and verification do not "
                    "touch soft-deleted rows. Default: 90."
                ),
                "default": 90,
            },
        ]

    def save_config(
        self,
        values: Dict[str, Any],
        claudia_home: str,
    ) -> None:
        """Write non-secret config to ``{claudia_home}/memory/claudia/config.json``.

        Merges with any existing file so callers can write partial
        updates. Creates parent directories if missing. Corrupt
        existing files are silently replaced (logged at DEBUG).
        """
        config_path = (
            Path(claudia_home) / "memory" / "claudia" / "config.json"
        )
        config_path.parent.mkdir(parents=True, exist_ok=True)

        existing: Dict[str, Any] = {}
        if config_path.exists():
            try:
                existing = json.loads(config_path.read_text())
                if not isinstance(existing, dict):
                    existing = {}
            except Exception as exc:
                logger.debug(
                    "Overwriting corrupt config at %s: %s",
                    config_path, exc,
                )
                existing = {}

        existing.update(values)
        config_path.write_text(json.dumps(existing, indent=2))

    def _config_path(self) -> Optional[Path]:
        """Return the expected config.json location for this provider."""
        if self._claudia_home is None:
            return None
        return self._claudia_home / "memory" / "claudia" / "config.json"

    def _load_config(self) -> Dict[str, Any]:
        """Read and parse config.json if it exists.

        Returns an empty dict on missing file, unreadable file, or
        malformed JSON. Never raises — consumers of ``self._config``
        should always use ``.get(key, default)`` patterns.
        """
        path = self._config_path()
        if path is None or not path.exists():
            return {}
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            logger.debug("Failed to load config %s: %s", path, exc)
            return {}
        if not isinstance(data, dict):
            return {}
        return data

    # ── Profile and home resolution ───────────────────────────────────

    @staticmethod
    def _resolve_claudia_home(kwargs: Dict[str, Any]) -> Path:
        """Honor the explicit kwarg, falling back to the constants helper."""
        explicit = kwargs.get("claudia_home")
        if explicit:
            return Path(explicit).expanduser()
        try:
            from claudia_constants import get_claudia_home

            return get_claudia_home()
        except ImportError:  # pragma: no cover - defensive fallback
            return Path.home() / ".claudia"

    @staticmethod
    def _resolve_profile(kwargs: Dict[str, Any]) -> str:
        """Priority: user_id > agent_identity > agent_workspace > 'default'."""
        for key in ("user_id", "agent_identity", "agent_workspace"):
            value = kwargs.get(key)
            if value:
                return str(value)
        return "default"

    # ── Optional lifecycle hooks ──────────────────────────────────────

    def on_delegation(
        self,
        task: str,
        result: str,
        *,
        child_session_id: str = "",
        **kwargs: Any,
    ) -> None:
        """Parent-side observation of subagent work (Phase 2C.16).

        Called on the PARENT provider when a subagent completes.
        Claudia captures this as a ``memories`` row so the parent
        can recall what was delegated:

          content = "Delegated: <task> → Result: <result>"
          origin = 'inferred' (it's an observation, not a user fact)
          source_type = 'delegation'
          source_ref = child_session_id (for provenance back to the
                       subagent's log)
          importance = 0.6 (above default because delegation is
                       a deliberate user action with known intent)

        Skipped entirely for non-primary agent_context. Empty
        task AND empty result is a no-op (nothing to record). If
        only the task is non-empty (subagent returned nothing),
        we still record it — the parent observed that it tried.
        """
        if self._writer is None:
            return
        if self._agent_context != "primary":
            return

        task_stripped = (task or "").strip()
        result_stripped = (result or "").strip()
        if not task_stripped and not result_stripped:
            return

        if task_stripped and result_stripped:
            content = f"Delegated: {task_stripped} → Result: {result_stripped}"
        elif task_stripped:
            content = f"Delegated: {task_stripped} (no result)"
        else:
            content = f"Subagent result: {result_stripped}"

        source_ref = child_session_id or self._session_id or ""

        self._enqueue_insert_memory(
            content,
            origin="inferred",
            source_type="delegation",
            source_ref=source_ref,
            importance=0.6,
        )

    def on_session_end(
        self,
        messages: List[Dict[str, Any]],
    ) -> None:
        """End-of-session hook (Phase 2C.6).

        Flushes any pending cognitive work so in-flight extractions
        and commitment detections land in the DB before the session
        marker closes. This is softer than ``shutdown()`` because
        it leaves the writer and reader pool alive — the host may
        still reuse the provider for another session without
        re-initializing.

        No-op for non-primary agent_context (cron / subagent /
        flush) since those contexts don't write anyway. No-op for
        a shutdown provider (writer is gone).

        Does NOT run extraction or detection on the full message
        history — per-turn sync_turn calls already handled that,
        and re-processing the full history would be expensive and
        duplicative.
        """
        if self._writer is None:
            return
        if self._agent_context != "primary":
            return
        # Soft flush — host may continue the session
        self.flush(timeout=5.0)

    def on_pre_compress(
        self,
        messages: List[Dict[str, Any]],
    ) -> str:
        """Pre-compression hook (Phase 2C.6).

        Called by the host BEFORE context compression discards old
        messages. The return string is injected into the compressor's
        summary prompt so provider-extracted insights aren't lost
        when the raw conversation is thrown away.

        Claudia returns a bullet list of the top recent memories
        from the current profile, ordered by recency × importance,
        capped at 5 bullets. The compressor sees these as
        "high-signal content to preserve" and their information is
        kept in the summary even when the original turns are gone.

        Returns an empty string if:
        - The provider is shutdown (no reader pool)
        - The profile has no memories
        - An internal error occurs (logged at DEBUG)
        """
        if self._reader_pool is None:
            return ""

        try:
            with self._reader_pool.acquire() as conn:
                rows = conn.execute(
                    """
                    SELECT content, importance, source_type
                    FROM memories
                    WHERE profile = ? AND deleted_at IS NULL
                    ORDER BY accessed_at DESC, importance DESC
                    LIMIT 5
                    """,
                    (self._profile,),
                ).fetchall()
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("on_pre_compress reader query failed: %s", exc)
            return ""

        if not rows:
            return ""

        lines = ["Claudia-preserved memories (from compressed window):"]
        for row in rows:
            provenance = ""
            if row["source_type"]:
                provenance = f" [{row['source_type']}]"
            lines.append(
                f"- {row['content']} "
                f"(importance={row['importance']:.2f}){provenance}"
            )
        return "\n".join(lines)

    def on_turn_start(
        self,
        turn_number: int,
        message: str,
        **kwargs: Any,
    ) -> None:
        """Update the cost-governance state from ``remaining_tokens``.

        The ABC may pass ``remaining_tokens`` in kwargs. We read it,
        stash it on self, and let every subsequent hook in this
        turn (prefetch, sync_turn, system_prompt_block) call
        ``decide_budget(self._budget_state)`` to decide how much
        work to do.

        Per the ABC contract, this hook MUST NOT block the turn —
        real enforcement of budgets happens in ``run_agent.py``.
        Claudia just reads the signal and gracefully degrades her
        own contribution to the prompt when room is tight.
        """
        self._budget_state = update_budget_state(
            self._budget_state,
            kwargs,
            now=datetime.now(timezone.utc),
            turn_number=turn_number,
        )

    def system_prompt_block(self) -> str:
        """Inject mode + compact stats into the system prompt.

        Uses a reader pool connection for the schema introspection
        query so it doesn't contend with the writer thread.
        """
        if self._reader_pool is None or self._router is None:
            return ""

        with self._reader_pool.acquire() as conn:
            stats = schema.describe_schema(conn)

        decision = self._router.select_mode()

        return (
            "# Claudia Memory\n"
            f"Mode: {decision.mode.value}. "
            f"{stats['memories_count']} memories, "
            f"{stats['entities_count']} entities, "
            f"{stats['relationships_count']} relationships, "
            f"{stats['commitments_count']} commitments.\n"
            "Use memory.recall to search, memory.remember to store a fact, "
            "and memory.about to look up a person, organization, or project."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Run hybrid search and format the result as budgeted bullets.

        Acquires a reader connection, runs the router's one-shot
        search, formats results as compact bullets, then truncates
        to fit the active token budget.

        Phase 2B.5: result COUNT is governed by
        ``decide_budget(self._budget_state).prefetch_limit``.
        Phase 2B.6: result TOKEN TOTAL is bounded by
        ``decide_budget(self._budget_state).prefetch_budget_tokens``
        via ``truncate_to_budget``. If the formatted text would
        exceed the budget, bullets are dropped from the bottom and
        a ``(truncated)`` marker is appended. This ensures Claudia
        never blows past the prompt budget even if the SQL result
        set happens to contain verbose memory content.
        """
        if self._reader_pool is None or self._router is None:
            return ""
        if not query or not query.strip():
            return ""

        decision = decide_budget(self._budget_state)
        limit = decision.prefetch_limit

        with self._reader_pool.acquire() as conn:
            results = self._router.search(
                conn,
                query,
                profile=self._profile,
                limit=limit,
            )

        if not results:
            return ""

        lines = ["## Claudia Memory"]
        for r in results:
            provenance = ""
            if r.source_type:
                provenance = f" [{r.source_type}]"
            lines.append(
                f"- {r.content} "
                f"(score={r.score:.2f}, importance={r.importance:.2f}){provenance}"
            )
        formatted = "\n".join(lines)

        # Phase 2B.6: enforce the token budget. This is the LAST
        # step before returning so any downstream format change
        # is still counted.
        return truncate_to_budget(
            formatted,
            max_tokens=decision.prefetch_budget_tokens,
        )

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Store the completed turn + extract entities + detect commitments.

        Three async fire-and-forget pipelines are kicked off:

        1. **Memory insert.** Embedding is generated on the caller
           thread (invariant #11) and the INSERT is enqueued on the
           writer queue. This is the Phase 2A path, unchanged.
        2. **Entity extraction** (Phase 2B.1). The combined turn
           text is submitted to the cognitive executor; the worker
           calls ``LLMExtractor.extract()`` and enqueues entity
           upserts on the writer.
        3. **Commitment detection** (Phase 2B.2). The USER's content
           (not the combined turn) is submitted to the cognitive
           executor; the worker calls
           ``CommitmentDetector.detect()`` and enqueues commitment
           inserts on the writer. Only user content is scanned
           because commitments come from the user — Claudia's
           responses go through approval flow before becoming
           commitments.

        All three paths skip entirely for non-primary agent contexts
        (cron, subagent, flush) per the ABC guidance. ``sync_turn``
        stays non-blocking: it computes the embedding and submits
        both cognitive futures, then returns. Actual LLM latency
        lives on the cognitive worker thread.
        """
        if self._writer is None:
            return
        if self._agent_context != "primary":
            return

        combined = self._format_turn(user_content, assistant_content)
        if not combined:
            return

        source_ref = session_id or self._session_id or ""

        self._enqueue_insert_memory(
            combined,
            origin="extracted",
            source_type="conversation",
            source_ref=source_ref,
            importance=0.5,
        )

        # Phase 2B.5: cost governance. If remaining_tokens is
        # critically low, skip extraction and detection entirely —
        # both are best-effort background work that can be deferred
        # without corrupting the conversation. The memory insert
        # above is NOT skipped because it's the source of truth.
        decision = decide_budget(self._budget_state)

        if not decision.skip_extraction:
            self._enqueue_extract(combined, source_ref=source_ref)

        if not decision.skip_detection:
            self._enqueue_detect_commitments(
                user_content, source_ref=source_ref
            )

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for all pending extractions and writes to drain.

        Order matters: extraction produces writes (entity upserts),
        so we must drain the extraction pool FIRST, then flush the
        writer. If we reversed the order, an extraction that hasn't
        yet enqueued its upserts would silently race past the flush.

        Returns True if both stages drained within the timeout. The
        ``timeout`` budget is shared across both stages; if extraction
        consumed most of it, the writer flush gets whatever is left.
        """
        deadline = time.monotonic() + timeout

        if not self._drain_cognitive_futures(deadline):
            return False

        if self._writer is None:
            return True

        remaining = max(0.0, deadline - time.monotonic())
        return self._writer.flush(timeout=remaining)

    def _drain_cognitive_futures(self, deadline: float) -> bool:
        """Wait for all pending extraction futures to complete.

        Snapshots the pending-futures list under the lock, then waits
        on the snapshot without holding the lock so new futures can
        still register their completion callbacks. Returns True if
        all snapshotted futures finished before the deadline.
        """
        if self._cognitive_pool is None:
            return True

        with self._cognitive_lock:
            pending = list(self._pending_cognitive_futures)

        if not pending:
            return True

        remaining = max(0.01, deadline - time.monotonic())
        done, not_done = concurrent.futures.wait(
            pending,
            timeout=remaining,
            return_when=concurrent.futures.ALL_COMPLETED,
        )
        return not not_done

    def consolidate(
        self,
        *,
        timeout: float = 30.0,
    ) -> ConsolidationResult:
        """Run a full consolidation pass (Phase 2B.3).

        Flushes pending cognitive work first so the consolidator sees
        a consistent view (extracted entities from recent turns, not
        a half-processed state), then submits the consolidation job
        through the writer queue and waits for the result.

        Order:
          1. ``flush()`` drains any pending extractions and commitment
             detections. Their entity/commitment writes land in the
             tables before consolidation starts scanning.
          2. A single writer job runs
             ``consolidation.run_consolidation`` inside the writer's
             transaction, so all fuzzy merges and commitment-FK
             updates commit atomically alongside normal writes.
          3. The caller blocks (up to ``timeout`` seconds) until the
             job finishes and returns the ``ConsolidationResult``.

        Phase 2C.4: honors ``auto_merge_threshold`` from
        ``self._config`` when set, otherwise uses
        ``consolidation.AUTO_MERGE_THRESHOLD``.

        Designed to be called by an external scheduler (cron-style)
        rather than on every sync_turn. Safe to call concurrently
        with sync_turn calls — the writer queue serializes work.

        Returns an empty ConsolidationResult if the provider has
        been shut down or if the writer queue rejects the job.
        """
        if self._writer is None:
            return ConsolidationResult()

        if not self.flush(timeout=timeout):
            logger.warning(
                "consolidate: flush timed out, proceeding with "
                "possibly-stale state"
            )

        profile = self._profile
        threshold = self._config.get(
            "auto_merge_threshold",
            consolidation.AUTO_MERGE_THRESHOLD,
        )

        def _job(conn):
            return consolidation.run_consolidation(
                conn, profile=profile, threshold=threshold
            )

        result = self._writer.enqueue_and_wait(_job, timeout=timeout)
        if isinstance(result, ConsolidationResult):
            return result
        return ConsolidationResult()

    def purge_old_soft_deletes(
        self,
        *,
        timeout: float = 30.0,
        retention_days: Optional[int] = None,
    ) -> RetentionResult:
        """Permanently remove soft-deleted rows older than the window (Phase 2C.4).

        Destructive and irreversible: actual DELETE statements
        against the memories, entities, relationships, and
        commitments tables. Provided as a separate method from
        ``consolidate()`` so users opt into the irreversibility
        explicitly.

        ``retention_days`` resolves in this order:
          1. Explicit argument (caller override)
          2. ``retention_days`` from self._config
          3. ``retention.DEFAULT_RETENTION_DAYS`` (90)

        Returns an empty RetentionResult on a shut-down provider
        or writer-timeout.
        """
        if self._writer is None:
            return RetentionResult()

        if retention_days is None:
            retention_days = int(self._config.get(
                "retention_days", retention.DEFAULT_RETENTION_DAYS
            ))

        profile = self._profile

        def _job(conn):
            return retention.purge_old_soft_deletes(
                conn,
                profile=profile,
                retention_days=retention_days,
            )

        result = self._writer.enqueue_and_wait(_job, timeout=timeout)
        if isinstance(result, RetentionResult):
            return result
        return RetentionResult()

    def verify(
        self,
        *,
        timeout: float = 30.0,
    ) -> VerificationResult:
        """Run a full verification pass (Phase 2B.4).

        Applies confidence decay to eligible memories and flags
        stale pending memories. Same pattern as ``consolidate()``:
        flush pending cognitive work, then submit
        ``verification.run_verification`` as a single writer job.

        Phase 2C.3: honors ``decay_half_life_days`` and
        ``stale_flag_days`` from ``self._config`` when set,
        otherwise uses verification.py defaults.

        Designed for scheduled invocation. Running it on every
        sync_turn would be wasteful — the decay function is driven
        by elapsed time since ``accessed_at``, so sub-day cadence
        produces minimal state change. A daily schedule is typical.

        Returns an empty result on a shut-down provider or on
        writer timeout.
        """
        if self._writer is None:
            return VerificationResult()

        if not self.flush(timeout=timeout):
            logger.warning(
                "verify: flush timed out, proceeding with possibly-stale state"
            )

        profile = self._profile
        half_life_days = self._config.get(
            "decay_half_life_days",
            verification.DEFAULT_HALF_LIFE_DAYS,
        )
        stale_days = self._config.get(
            "stale_flag_days",
            verification.DEFAULT_STALE_DAYS,
        )

        def _job(conn):
            return verification.run_verification(
                conn,
                profile=profile,
                half_life_days=half_life_days,
                stale_days=stale_days,
            )

        result = self._writer.enqueue_and_wait(_job, timeout=timeout)
        if isinstance(result, VerificationResult):
            return result
        return VerificationResult()

    def shutdown(self) -> None:
        """Graceful shutdown.

        Order:
        1. Drain the extraction pool (extractions can still enqueue writes).
        2. Stop the writer (drains remaining jobs, including entity
           upserts submitted by the drained extractions).
        3. Close the reader pool.

        Each stage is independently wrapped so a failure in one does
        not skip the others.
        """
        if self._cognitive_pool is not None:
            try:
                # wait=True drains the executor's task queue before
                # joining its workers. Since we submit with fire-and-
                # forget futures, the worker processes every enqueued
                # extraction before shutdown returns.
                self._cognitive_pool.shutdown(wait=True)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Error shutting down claudia extraction pool")
            self._cognitive_pool = None

        if self._writer is not None:
            try:
                self._writer.stop(timeout=10.0)
            except Exception:  # pragma: no cover - defensive
                logger.debug("Error stopping claudia writer")
            self._writer = None

        if self._reader_pool is not None:
            try:
                self._reader_pool.close()
            except Exception:  # pragma: no cover - defensive
                logger.debug("Error closing claudia reader pool")
            self._reader_pool = None

    # ── Tool call dispatch ────────────────────────────────────────────

    def handle_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Dispatch a memory.* tool call to its handler. Returns JSON string."""
        if self._writer is None or self._reader_pool is None:
            return json.dumps({"error": "claudia memory provider not initialized"})

        try:
            if tool_name == "memory.recall":
                return self._handle_recall(args)
            if tool_name == "memory.remember":
                return self._handle_remember(args)
            if tool_name == "memory.about":
                return self._handle_about(args)
            if tool_name == "memory.commitments":
                return self._handle_commitments(args)
            if tool_name == "memory.status":
                return self._handle_status(args)
            if tool_name == "memory.commitment_complete":
                return self._handle_commitment_complete(args)
            if tool_name == "memory.commitment_drop":
                return self._handle_commitment_drop(args)
            if tool_name == "memory.verify_memory":
                return self._handle_verify_memory(args)
            if tool_name == "memory.flag_memory":
                return self._handle_flag_memory(args)
            if tool_name == "memory.contradicts_memory":
                return self._handle_contradicts_memory(args)
            if tool_name == "memory.correct_memory":
                return self._handle_correct_memory(args)
            if tool_name == "memory.relate":
                return self._handle_relate(args)
            if tool_name == "memory.search_entities":
                return self._handle_search_entities(args)
            if tool_name == "memory.trace":
                return self._handle_trace(args)
            if tool_name == "memory.forget_memory":
                return self._handle_forget_memory(args)
            if tool_name == "memory.forget_entity":
                return self._handle_forget_entity(args)
            return json.dumps({"error": f"unknown tool: {tool_name}"})
        except Exception as exc:
            logger.exception("claudia memory tool call failed: %s", tool_name)
            return json.dumps({"error": f"{type(exc).__name__}: {exc}"})

    def _handle_recall(self, args: Dict[str, Any]) -> str:
        query = str(args.get("query", "")).strip()
        if not query:
            return json.dumps({"error": "missing required parameter: query"})

        limit = int(args.get("limit", 10))
        if limit < 1:
            limit = 1
        if limit > 50:
            limit = 50

        assert self._reader_pool is not None and self._router is not None
        with self._reader_pool.acquire() as conn:
            results = self._router.search(
                conn,
                query,
                profile=self._profile,
                limit=limit,
            )

        return json.dumps(
            {
                "query": query,
                "results": [
                    {
                        "content": r.content,
                        "score": round(r.score, 3),
                        "importance": round(r.importance, 2),
                        "source_type": r.source_type,
                        "source_ref": r.source_ref,
                        "access_count": r.access_count,
                    }
                    for r in results
                ],
            }
        )

    def _handle_remember(self, args: Dict[str, Any]) -> str:
        content = str(args.get("content", "")).strip()
        if not content:
            return json.dumps({"error": "missing required parameter: content"})

        importance = float(args.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        source_type = str(args.get("source_type", "conversation"))

        memory_id = self._insert_memory_sync(
            content,
            origin="user_stated",
            source_type=source_type,
            source_ref=self._session_id or "",
            importance=importance,
        )

        if memory_id is None:
            return json.dumps({"error": "failed to store memory (writer queue)"})

        return json.dumps(
            {
                "result": "remembered",
                "memory_id": memory_id,
            }
        )

    def _handle_about(self, args: Dict[str, Any]) -> str:
        name = str(args.get("name", "")).strip()
        if not name:
            return json.dumps({"error": "missing required parameter: name"})

        kind = args.get("kind")
        if kind is not None:
            kind = str(kind)

        assert self._reader_pool is not None

        with self._reader_pool.acquire() as conn:
            entity = entities.find_entity(
                conn,
                name,
                kind=kind,
                profile=self._profile,
            )

            if entity is None:
                return json.dumps(
                    {
                        "result": None,
                        "message": f"no entity named {name!r}",
                    }
                )

            rels = entities.get_relationships(
                conn,
                entity.id,
                profile=self._profile,
            )

        return json.dumps(
            {
                "result": {
                    "id": entity.id,
                    "kind": entity.kind,
                    "name": entity.name,
                    "aliases": entity.aliases,
                    "attributes": entity.attributes,
                    "importance": entity.importance,
                    "access_count": entity.access_count,
                    "relationships": [
                        {
                            "type": r.type,
                            "from_entity_id": r.from_entity_id,
                            "to_entity_id": r.to_entity_id,
                            "health_score": round(r.health_score, 2),
                            "last_interaction_at": r.last_interaction_at,
                        }
                        for r in rels
                    ],
                }
            }
        )

    def _handle_commitments(self, args: Dict[str, Any]) -> str:
        """List commitments in the current profile (Phase 2C.5).

        Args:
          status: one of 'open', 'completed', 'overdue',
                  'dropped', 'superseded', or 'all'. Default 'open'.
          limit: max rows to return. Default 50, clamped to [1, 200].

        Returns a JSON string with a top-level ``commitments`` array.
        Each entry has id, content, deadline, status, source_type,
        source_ref, created_at, updated_at.
        """
        status_raw = args.get("status", "open")
        if not isinstance(status_raw, str):
            return json.dumps({"error": "status must be a string"})
        status: Optional[str] = status_raw.lower().strip()

        if status == "all":
            status_filter: Optional[str] = None
        elif status in commitments_module.VALID_COMMITMENT_STATUSES:
            status_filter = status
        else:
            return json.dumps({
                "error": (
                    f"invalid status: {status!r}. Must be one of "
                    f"{sorted(commitments_module.VALID_COMMITMENT_STATUSES)} "
                    "or 'all'."
                )
            })

        limit = int(args.get("limit", 50))
        if limit < 1:
            limit = 1
        if limit > 200:
            limit = 200

        assert self._reader_pool is not None
        with self._reader_pool.acquire() as conn:
            rows = commitments_module.list_commitments(
                conn,
                status=status_filter,
                profile=self._profile,
                limit=limit,
            )

        return json.dumps({
            "commitments": [
                {
                    "id": row.id,
                    "content": row.content,
                    "deadline": row.deadline,
                    "status": row.status,
                    "source_type": row.source_type,
                    "source_ref": row.source_ref,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]
        })

    def _handle_commitment_complete(
        self,
        args: Dict[str, Any],
    ) -> str:
        """Mark a commitment as completed (Phase 2C.7).

        Wraps commitments.update_commitment_status via the writer
        queue so the status transition commits in the writer's
        transaction. Returns JSON with the updated commitment or
        an error.
        """
        return self._handle_commitment_status_transition(
            args, "completed", "memory.commitment_complete"
        )

    def _handle_commitment_drop(
        self,
        args: Dict[str, Any],
    ) -> str:
        """Mark a commitment as dropped (Phase 2C.7).

        Dropped means the user explicitly abandoned the task, as
        opposed to completing it. No completed_at is set.
        """
        return self._handle_commitment_status_transition(
            args, "dropped", "memory.commitment_drop"
        )

    def _handle_forget_entity(self, args: Dict[str, Any]) -> str:
        """Soft-delete an entity by name (Phase 2C.15).

        Uses find_entity (case-insensitive name/alias match) to
        resolve the id, then issues a soft_delete_entity via the
        writer queue. Relationships referencing the entity are
        NOT cascaded — they'll be cleaned up by retention purge
        after the retention window expires.
        """
        name = args.get("name", "")
        if not isinstance(name, str) or not name.strip():
            return json.dumps({
                "error": "memory.forget_entity: 'name' is required"
            })
        name = name.strip()

        if self._writer is None:
            return json.dumps({
                "error": "memory.forget_entity: provider not initialized"
            })

        profile = self._profile

        def _job(conn):
            ent = entities.find_entity(conn, name, profile=profile)
            if ent is None:
                return None
            ok = entities.soft_delete_entity(
                conn, ent.id, profile=profile
            )
            if not ok:
                return None
            return {
                "id": ent.id,
                "name": ent.name,
                "kind": ent.kind,
            }

        result = self._writer.enqueue_and_wait(_job, timeout=5.0)
        if result is None:
            return json.dumps({
                "error": (
                    f"memory.forget_entity: no entity named {name!r} "
                    f"in profile {profile!r}"
                )
            })

        return json.dumps({"ok": True, "entity": result})

    def _handle_forget_memory(self, args: Dict[str, Any]) -> str:
        """Soft-delete a memory (Phase 2C.14).

        Issues an UPDATE setting deleted_at to the current time
        via the writer queue. Follows the standard profile-isolation
        path: cross-profile and already-deleted rows return "unknown
        id" errors. Returns ``{ok: true, id: <id>}`` on success.
        """
        raw_id = args.get("id")
        if raw_id is None:
            return json.dumps({
                "error": "memory.forget_memory: missing required parameter 'id'"
            })
        try:
            memory_id = int(raw_id)
        except (TypeError, ValueError):
            return json.dumps({
                "error": (
                    f"memory.forget_memory: 'id' must be an integer, "
                    f"got {raw_id!r}"
                )
            })

        if self._writer is None:
            return json.dumps({
                "error": "memory.forget_memory: provider not initialized"
            })

        profile = self._profile
        now_iso = datetime.now(timezone.utc).isoformat()

        def _job(conn):
            cur = conn.execute(
                """
                UPDATE memories
                SET deleted_at = ?
                WHERE id = ?
                  AND profile = ?
                  AND deleted_at IS NULL
                """,
                (now_iso, memory_id, profile),
            )
            return cur.rowcount > 0

        updated = self._writer.enqueue_and_wait(_job, timeout=5.0)
        if not updated:
            return json.dumps({
                "error": (
                    f"memory.forget_memory: no memory with id {memory_id} "
                    f"in profile {profile!r} (may already be deleted or "
                    "in another profile)"
                )
            })

        return json.dumps({"ok": True, "id": memory_id})

    def _handle_trace(self, args: Dict[str, Any]) -> str:
        """Walk a memory's correction chain (Phase 2C.12).

        Returns every version of the fact in one flat list,
        ordered by ``created_at`` ascending. Walks backward from
        the given id via the ``corrected_from`` column and forward
        via a reverse lookup on other memories' ``corrected_from``.

        The traversal is breadth-first but small — correction
        chains rarely exceed a few steps in practice. Safe-cap
        at 100 entries to prevent pathological cycles (shouldn't
        happen given the linear nature of corrections, but
        defensive).
        """
        raw_id = args.get("id")
        if raw_id is None:
            return json.dumps({
                "error": "memory.trace: missing required parameter 'id'"
            })
        try:
            memory_id = int(raw_id)
        except (TypeError, ValueError):
            return json.dumps({
                "error": (
                    f"memory.trace: 'id' must be an integer, got {raw_id!r}"
                )
            })

        if self._reader_pool is None:
            return json.dumps({
                "error": "memory.trace: provider not initialized"
            })

        profile = self._profile

        with self._reader_pool.acquire() as conn:
            # Verify the root exists in this profile
            root = conn.execute(
                """
                SELECT id FROM memories
                WHERE id = ? AND profile = ? AND deleted_at IS NULL
                """,
                (memory_id, profile),
            ).fetchone()
            if root is None:
                return json.dumps({
                    "error": (
                        f"memory.trace: no memory with id {memory_id} "
                        f"in profile {profile!r}"
                    )
                })

            # Collect the chain via a BFS over corrected_from links
            visited: set = set()
            to_visit = [memory_id]
            CAP = 100

            while to_visit and len(visited) < CAP:
                current = to_visit.pop(0)
                if current in visited:
                    continue
                visited.add(current)

                # Follow backward: predecessor via corrected_from
                row = conn.execute(
                    """
                    SELECT corrected_from FROM memories
                    WHERE id = ? AND profile = ?
                    """,
                    (current, profile),
                ).fetchone()
                if row is not None and row["corrected_from"] is not None:
                    to_visit.append(int(row["corrected_from"]))

                # Follow forward: any memory that corrected_from == current
                forward_rows = conn.execute(
                    """
                    SELECT id FROM memories
                    WHERE corrected_from = ? AND profile = ?
                      AND deleted_at IS NULL
                    """,
                    (current, profile),
                ).fetchall()
                for fwd in forward_rows:
                    to_visit.append(int(fwd["id"]))

            if not visited:
                return json.dumps({
                    "error": (
                        f"memory.trace: no memory with id {memory_id}"
                    )
                })

            # Fetch full rows for the collected ids and sort by time
            placeholders = ",".join("?" * len(visited))
            rows = conn.execute(
                f"""
                SELECT id, content, origin, confidence, verification,
                       corrected_from, source_type, source_ref,
                       created_at, accessed_at
                FROM memories
                WHERE id IN ({placeholders})
                  AND profile = ?
                ORDER BY created_at ASC, id ASC
                """,
                (*visited, profile),
            ).fetchall()

        chain = [
            {
                "id": row["id"],
                "content": row["content"],
                "origin": row["origin"],
                "confidence": float(row["confidence"]),
                "verification": row["verification"],
                "corrected_from": row["corrected_from"],
                "source_type": row["source_type"],
                "source_ref": row["source_ref"],
                "created_at": row["created_at"],
                "accessed_at": row["accessed_at"],
            }
            for row in rows
        ]

        return json.dumps({"chain": chain})

    def _handle_search_entities(self, args: Dict[str, Any]) -> str:
        """Fuzzy entity search (Phase 2C.11).

        Wraps entities.search_entities via the reader pool. Supports
        substring match, '*' wildcard, optional kind filter, and
        limit. Invalid kind returns a structured error (rather than
        letting entities._validate_kind raise).
        """
        query = args.get("query")
        if not isinstance(query, str):
            return json.dumps({
                "error": "memory.search_entities: 'query' is required"
            })

        kind = args.get("kind")
        if kind is not None:
            if not isinstance(kind, str):
                return json.dumps({
                    "error": "memory.search_entities: 'kind' must be a string"
                })
            if kind not in entities.VALID_KINDS:
                return json.dumps({
                    "error": (
                        f"memory.search_entities: invalid kind {kind!r}. "
                        f"Must be one of {sorted(entities.VALID_KINDS)}."
                    )
                })

        limit = int(args.get("limit", 10))
        if limit < 1:
            limit = 1
        if limit > 100:
            limit = 100

        if self._reader_pool is None:
            return json.dumps({
                "error": "memory.search_entities: provider not initialized"
            })

        with self._reader_pool.acquire() as conn:
            results = entities.search_entities(
                conn,
                query,
                kind=kind,
                profile=self._profile,
                limit=limit,
            )

        return json.dumps({
            "entities": [
                {
                    "id": e.id,
                    "name": e.name,
                    "kind": e.kind,
                    "aliases": e.aliases,
                    "attributes": e.attributes,
                    "importance": e.importance,
                    "access_count": e.access_count,
                }
                for e in results
            ]
        })

    def _handle_relate(self, args: Dict[str, Any]) -> str:
        """Create an explicit entity relationship (Phase 2C.10).

        Both entities must exist in the current profile. If either
        is missing, returns an error naming which entity could not
        be found (helpful for the LLM to decide whether to create
        the entity first).

        Wraps entities.find_entity + create_relationship inside a
        single writer job. Unique constraint collisions
        (profile, from, to, type) return a clean error JSON.
        """
        from_name = args.get("from_name", "")
        to_name = args.get("to_name", "")
        rel_type = args.get("type", "")
        notes = args.get("notes")

        if not isinstance(from_name, str) or not from_name.strip():
            return json.dumps({
                "error": "memory.relate: 'from_name' is required"
            })
        if not isinstance(to_name, str) or not to_name.strip():
            return json.dumps({
                "error": "memory.relate: 'to_name' is required"
            })
        if not isinstance(rel_type, str) or not rel_type.strip():
            return json.dumps({
                "error": "memory.relate: 'type' is required"
            })

        from_name = from_name.strip()
        to_name = to_name.strip()
        rel_type = rel_type.strip()
        notes_str: Optional[str] = None
        if isinstance(notes, str) and notes.strip():
            notes_str = notes.strip()

        if self._writer is None:
            return json.dumps({
                "error": "memory.relate: provider not initialized"
            })

        profile = self._profile

        def _job(conn):
            from_ent = entities.find_entity(
                conn, from_name, profile=profile
            )
            if from_ent is None:
                return {"error": f"unknown entity 'from_name'={from_name!r}"}

            to_ent = entities.find_entity(
                conn, to_name, profile=profile
            )
            if to_ent is None:
                return {"error": f"unknown entity 'to_name'={to_name!r}"}

            try:
                rel = entities.create_relationship(
                    conn,
                    from_ent.id,
                    to_ent.id,
                    rel_type,
                    notes=notes_str,
                    profile=profile,
                )
            except sqlite3.IntegrityError:
                return {
                    "error": (
                        f"relationship already exists: "
                        f"({from_name!r} -> {to_name!r}, type={rel_type!r})"
                    )
                }

            return {
                "relationship": {
                    "id": rel.id,
                    "from_entity_id": rel.from_entity_id,
                    "to_entity_id": rel.to_entity_id,
                    "type": rel.type,
                    "health_score": rel.health_score,
                    "notes": rel.notes,
                    "created_at": rel.created_at,
                },
                "from_entity": {
                    "id": from_ent.id,
                    "name": from_ent.name,
                    "kind": from_ent.kind,
                },
                "to_entity": {
                    "id": to_ent.id,
                    "name": to_ent.name,
                    "kind": to_ent.kind,
                },
            }

        result = self._writer.enqueue_and_wait(_job, timeout=5.0)
        if result is None:
            return json.dumps({
                "error": "memory.relate: writer queue timeout"
            })
        if "error" in result:
            return json.dumps({"error": result["error"]})
        return json.dumps({"ok": True, **result})

    def _handle_correct_memory(self, args: Dict[str, Any]) -> str:
        """Replace a memory with a corrected version (Phase 2C.9).

        Creates a new memory row linked to the old via
        ``corrected_from``. Marks the old row as ``contradicts``
        so the original content stays in the DB (audit trail) but
        ranks low in recall. Both operations happen inside one
        writer job so the correction is atomic.
        """
        raw_id = args.get("id")
        if raw_id is None:
            return json.dumps({
                "error": (
                    "memory.correct_memory: missing required parameter 'id'"
                )
            })
        try:
            old_id = int(raw_id)
        except (TypeError, ValueError):
            return json.dumps({
                "error": (
                    f"memory.correct_memory: 'id' must be an integer, "
                    f"got {raw_id!r}"
                )
            })

        new_content = args.get("new_content", "")
        if not isinstance(new_content, str) or not new_content.strip():
            return json.dumps({
                "error": (
                    "memory.correct_memory: 'new_content' is required "
                    "and must be a non-empty string"
                )
            })
        new_content = new_content.strip()

        if self._writer is None:
            return json.dumps({
                "error": "memory.correct_memory: provider not initialized"
            })

        profile = self._profile
        # Generate embedding on caller thread (invariant #11)
        embedding_blob = None
        embedding_dim = None
        if self._embedder is not None:
            embed_result = self._embedder.embed(new_content)
            if embed_result is not None:
                embedding_blob, embedding_dim = embed_result

        now_iso = datetime.now(timezone.utc).isoformat()
        session_ref = self._session_id or ""

        def _job(conn):
            # 1. Verify the old memory exists in this profile
            old_row = conn.execute(
                """
                SELECT id, content FROM memories
                WHERE id = ? AND profile = ? AND deleted_at IS NULL
                """,
                (old_id, profile),
            ).fetchone()
            if old_row is None:
                return None

            # 2. Insert the new corrected memory row
            cur = conn.execute(
                """
                INSERT INTO memories (
                    content, origin, confidence, importance, access_count,
                    embedding, embedding_dim, source_type, source_ref,
                    corrected_from, verification, profile,
                    created_at, accessed_at
                ) VALUES (?, 'corrected', 1.0, 0.8, 0, ?, ?,
                          'conversation', ?, ?, 'verified', ?, ?, ?)
                """,
                (
                    new_content,
                    embedding_blob,
                    embedding_dim,
                    session_ref,
                    old_id,
                    profile,
                    now_iso,
                    now_iso,
                ),
            )
            new_id = int(cur.lastrowid)

            # 3. Mark old memory as contradicts
            conn.execute(
                """
                UPDATE memories
                SET verification = 'contradicts'
                WHERE id = ? AND profile = ? AND deleted_at IS NULL
                """,
                (old_id, profile),
            )

            # 4. Read back the new row for the response
            new_row = conn.execute(
                """
                SELECT id, content, origin, confidence, verification,
                       corrected_from, source_type, source_ref,
                       created_at, accessed_at
                FROM memories WHERE id = ?
                """,
                (new_id,),
            ).fetchone()
            return {
                "id": new_row["id"],
                "content": new_row["content"],
                "origin": new_row["origin"],
                "confidence": float(new_row["confidence"]),
                "verification": new_row["verification"],
                "corrected_from": new_row["corrected_from"],
                "source_type": new_row["source_type"],
                "source_ref": new_row["source_ref"],
                "created_at": new_row["created_at"],
                "accessed_at": new_row["accessed_at"],
            }

        result = self._writer.enqueue_and_wait(_job, timeout=5.0)
        if result is None:
            return json.dumps({
                "error": (
                    f"memory.correct_memory: no memory with id {old_id} "
                    f"in profile {profile!r}"
                )
            })

        return json.dumps({"ok": True, "memory": result})

    def _handle_verify_memory(self, args: Dict[str, Any]) -> str:
        """Mark a memory as verified (Phase 2C.8)."""
        return self._handle_memory_verification_transition(
            args, verification.mark_verified, "memory.verify_memory", "verified"
        )

    def _handle_flag_memory(self, args: Dict[str, Any]) -> str:
        """Mark a memory as flagged (Phase 2C.8)."""
        return self._handle_memory_verification_transition(
            args, verification.mark_flagged, "memory.flag_memory", "flagged"
        )

    def _handle_contradicts_memory(self, args: Dict[str, Any]) -> str:
        """Mark a memory as contradicting another (Phase 2C.8)."""
        return self._handle_memory_verification_transition(
            args,
            verification.mark_contradicts,
            "memory.contradicts_memory",
            "contradicts",
        )

    def _handle_memory_verification_transition(
        self,
        args: Dict[str, Any],
        mark_fn,
        tool_name: str,
        expected_status: str,
    ) -> str:
        """Shared helper for memory verification state tools (2C.8).

        Validates the id, submits the verification.mark_* call
        through the writer queue, and reads back the updated row
        to return its current state. ``mark_fn`` is one of
        ``verification.mark_verified``, ``mark_flagged``, or
        ``mark_contradicts``.
        """
        raw_id = args.get("id")
        if raw_id is None:
            return json.dumps({
                "error": f"{tool_name}: missing required parameter 'id'"
            })
        try:
            memory_id = int(raw_id)
        except (TypeError, ValueError):
            return json.dumps({
                "error": (
                    f"{tool_name}: 'id' must be an integer, got {raw_id!r}"
                )
            })

        if self._writer is None:
            return json.dumps({"error": f"{tool_name}: provider not initialized"})

        profile = self._profile

        def _job(conn):
            ok = mark_fn(conn, memory_id, profile=profile)
            if not ok:
                return None
            row = conn.execute(
                """
                SELECT id, content, verification, confidence, origin,
                       source_type, source_ref, created_at, accessed_at
                FROM memories
                WHERE id = ? AND profile = ? AND deleted_at IS NULL
                """,
                (memory_id, profile),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": row["id"],
                "content": row["content"],
                "verification": row["verification"],
                "confidence": float(row["confidence"]),
                "origin": row["origin"],
                "source_type": row["source_type"],
                "source_ref": row["source_ref"],
                "created_at": row["created_at"],
                "accessed_at": row["accessed_at"],
            }

        result = self._writer.enqueue_and_wait(_job, timeout=5.0)
        if result is None:
            return json.dumps({
                "error": (
                    f"{tool_name}: no memory with id {memory_id} in "
                    f"profile {profile!r} (may be deleted or in "
                    "another profile)"
                )
            })

        return json.dumps({"ok": True, "memory": result})

    def _handle_commitment_status_transition(
        self,
        args: Dict[str, Any],
        new_status: str,
        tool_name: str,
    ) -> str:
        """Shared helper for commitment state-mutation tools.

        Validates the id argument, submits the status update
        through the writer queue, and returns a JSON result.
        Missing/invalid id returns an error. Unknown id (not in
        profile, soft-deleted, etc.) returns an error.
        """
        raw_id = args.get("id")
        if raw_id is None:
            return json.dumps({
                "error": f"{tool_name}: missing required parameter 'id'"
            })
        try:
            commitment_id = int(raw_id)
        except (TypeError, ValueError):
            return json.dumps({
                "error": (
                    f"{tool_name}: 'id' must be an integer, "
                    f"got {raw_id!r}"
                )
            })

        if self._writer is None:
            return json.dumps({"error": f"{tool_name}: provider not initialized"})

        profile = self._profile

        def _job(conn):
            return commitments_module.update_commitment_status(
                conn,
                commitment_id,
                new_status,
                profile=profile,
            )

        updated = self._writer.enqueue_and_wait(_job, timeout=5.0)
        if updated is None:
            return json.dumps({
                "error": (
                    f"{tool_name}: no commitment with id {commitment_id} "
                    f"in profile {profile!r} (may be deleted or in "
                    "another profile)"
                )
            })

        return json.dumps({
            "ok": True,
            "commitment": {
                "id": updated.id,
                "content": updated.content,
                "status": updated.status,
                "deadline": updated.deadline,
                "source_type": updated.source_type,
                "source_ref": updated.source_ref,
                "created_at": updated.created_at,
                "updated_at": updated.updated_at,
                "completed_at": updated.completed_at,
            },
        })

    def _handle_status(self, args: Dict[str, Any]) -> str:
        """Return memory system status (Phase 2C.5).

        JSON fields:
          - memories_count / entities_count / relationships_count /
            commitments_count: row counts (non-deleted)
          - mode: current offline mode (FULL_HYBRID, FTS_IMP_REC, PURE_FTS)
          - session_id: the initialize()-time session id
          - profile: active profile key
        """
        if self._reader_pool is None or self._router is None:
            return json.dumps({"error": "provider not initialized"})

        with self._reader_pool.acquire() as conn:
            stats = schema.describe_schema(conn)

        decision = self._router.select_mode()

        return json.dumps({
            "memories_count": stats.get("memories_count", 0),
            "entities_count": stats.get("entities_count", 0),
            "relationships_count": stats.get("relationships_count", 0),
            "commitments_count": stats.get("commitments_count", 0),
            "mode": decision.mode.value,
            "session_id": self._session_id or "",
            "profile": self._profile,
        })

    # ── Internal write helpers ────────────────────────────────────────

    @staticmethod
    def _format_turn(user: str, assistant: str) -> str:
        """Format a conversation turn for storage as a memory row."""
        user = (user or "").strip()
        assistant = (assistant or "").strip()
        if not user and not assistant:
            return ""
        if user and assistant:
            return f"User: {user}\nAssistant: {assistant}"
        return user or assistant

    def _build_insert_params(
        self,
        content: str,
        *,
        origin: str,
        source_type: str,
        source_ref: str,
        importance: float,
    ) -> tuple:
        """Build the parameter tuple for an _INSERT_MEMORY_SQL job.

        Generates the embedding on the CALLER'S thread (not the
        writer thread) so Ollama HTTP latency doesn't stall queue
        processing. If the embedder is offline, ``embedding_blob``
        and ``embedding_dim`` are None and hybrid_search will fall
        back to the non-vector signals.
        """
        embedding_blob = None
        embedding_dim = None
        if self._embedder is not None:
            result = self._embedder.embed(content)
            if result is not None:
                embedding_blob, embedding_dim = result

        now_iso = datetime.now(timezone.utc).isoformat()
        return (
            content,
            origin,
            0.9,  # confidence
            importance,
            embedding_blob,
            embedding_dim,
            source_type,
            source_ref,
            self._profile,
            now_iso,
            now_iso,
        )

    def _enqueue_insert_memory(
        self,
        content: str,
        *,
        origin: str,
        source_type: str,
        source_ref: str,
        importance: float,
    ) -> None:
        """Fire-and-forget async insert. Used by sync_turn."""
        assert self._writer is not None

        params = self._build_insert_params(
            content,
            origin=origin,
            source_type=source_type,
            source_ref=source_ref,
            importance=importance,
        )

        def _job(conn):
            conn.execute(_INSERT_MEMORY_SQL, params)

        self._writer.enqueue(_job, block=False)

    def _insert_memory_sync(
        self,
        content: str,
        *,
        origin: str,
        source_type: str,
        source_ref: str,
        importance: float,
    ) -> Optional[int]:
        """Enqueue an insert and block until the writer processes it.

        Used by memory.remember because the caller expects the new
        row id in the tool response. Returns None on timeout or
        enqueue failure.
        """
        assert self._writer is not None

        params = self._build_insert_params(
            content,
            origin=origin,
            source_type=source_type,
            source_ref=source_ref,
            importance=importance,
        )

        def _job(conn):
            cur = conn.execute(_INSERT_MEMORY_SQL, params)
            return int(cur.lastrowid)

        return self._writer.enqueue_and_wait(_job, timeout=5.0)

    # ── Extraction pipeline (Phase 2B.1) ──────────────────────────────

    def _enqueue_extract(self, text: str, *, source_ref: str) -> None:
        """Submit text to the extraction executor (fire-and-forget).

        The submitted future runs ``_run_extract`` on a worker thread.
        Its result (any ``ExtractedEntity`` objects the LLM returned)
        are pushed to the writer queue as entity upsert jobs. The
        caller thread returns immediately after ``submit`` — that's
        what makes ``sync_turn`` non-blocking (invariant #3) despite
        LLM latency.

        Pending futures are tracked under ``_cognitive_lock`` so
        ``flush`` and ``shutdown`` can wait for them to drain before
        closing the writer. A done-callback removes each future from
        the pending list as soon as it completes.
        """
        if self._cognitive_pool is None or self._extractor is None:
            return
        if not text or not text.strip():
            return

        future = self._cognitive_pool.submit(
            self._run_extract, text, source_ref
        )

        with self._cognitive_lock:
            self._pending_cognitive_futures.append(future)

        future.add_done_callback(self._on_cognitive_done)

    def _on_cognitive_done(
        self,
        future: "concurrent.futures.Future",
    ) -> None:
        """Done-callback: drop the future from the pending list."""
        with self._cognitive_lock:
            try:
                self._pending_cognitive_futures.remove(future)
            except ValueError:
                # Already removed (e.g. shutdown drained and cleared).
                pass

    def _run_extract(self, text: str, source_ref: str) -> None:
        """Worker body: call the extractor and enqueue entity upserts.

        Runs on the single-worker extraction thread. Any exception
        is logged and swallowed — extraction is best-effort and must
        never block the memory insert path or kill the worker.
        """
        if self._extractor is None:
            return

        try:
            extracted = self._extractor.extract(text, source_ref=source_ref)
        except Exception:  # pragma: no cover - extractor contract says no raise
            logger.exception("claudia extractor raised (contract violation)")
            return

        if not extracted:
            return

        for ent in extracted:
            self._enqueue_upsert_entity(ent)

    def _enqueue_upsert_entity(self, ent: ExtractedEntity) -> None:
        """Enqueue an entity upsert on the writer queue.

        Runs on the extraction worker thread (caller of ``_run_extract``),
        not on the writer thread. The writer processes the enqueued
        job serially alongside memory inserts, so entity writes are
        consistent with the Phase 2A single-writer design.

        Uses the stored provider profile — captured at enqueue time
        rather than read from ``self`` inside the job — so a profile
        change between submit and execute cannot corrupt the write.
        """
        writer = self._writer
        if writer is None:
            return

        profile = self._profile
        kind = ent.kind
        name = ent.name
        aliases = list(ent.aliases) if ent.aliases else None
        attributes = dict(ent.attributes) if ent.attributes else None
        importance = ent.confidence

        def _job(conn):
            try:
                entities.upsert_entity(
                    conn,
                    kind,
                    name,
                    aliases=aliases,
                    attributes=attributes,
                    importance=importance,
                    profile=profile,
                )
            except ValueError as exc:
                # Invalid kind slipped through (shouldn't happen —
                # _coerce_json_to_entities filters invalid kinds).
                logger.debug(
                    "upsert_entity rejected kind=%r name=%r: %s",
                    kind, name, exc,
                )

        writer.enqueue(_job, block=False)

    # ── Built-in memory mirror (Phase 2C.2) ───────────────────────────

    #: Maps the ``target`` parameter of ``on_memory_write`` to the
    #: source_type we use for mirrored rows. Unknown targets are
    #: silently ignored (defensive: the ABC contract says 'memory'
    #: or 'user', but a future extension could add more).
    _BUILTIN_TARGET_SOURCE_TYPES = {
        "memory": "builtin_memory",
        "user": "builtin_user",
    }

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
    ) -> None:
        """Mirror built-in memory writes into Claudia's store (Phase 2C.2).

        The ABC calls this whenever the built-in memory tool (MEMORY.md
        or USER.md) is modified. Claudia mirrors the change so her
        recall surface stays in sync with whatever the user stored via
        the built-in path:

        - ``action='add'``: insert a new memory row with
          ``origin='user_stated'`` (explicit user assertion) and
          ``source_type='builtin_memory'`` or ``'builtin_user'``.
        - ``action='replace'``: soft-delete all existing mirrors for
          the same target, then insert the new content as a fresh row.
          This matches the built-in semantic of "the section now
          contains exactly this".
        - ``action='remove'``: soft-delete all existing mirrors for
          the same target.

        Unknown actions and targets are no-ops (defensive). Empty
        content on an ``add`` or ``replace`` is a no-op; empty
        content on ``remove`` still executes the soft-delete step.

        Thread safety: all writes go through the writer queue so this
        method is safe to call from any thread. It is non-blocking:
        enqueue and return.
        """
        if self._writer is None:
            return

        source_type = self._BUILTIN_TARGET_SOURCE_TYPES.get(target)
        if source_type is None:
            return

        action = (action or "").lower()

        if action == "add":
            if not content or not content.strip():
                return
            self._enqueue_mirror_insert(content, source_type)
        elif action == "replace":
            self._enqueue_mirror_soft_delete(source_type)
            if content and content.strip():
                self._enqueue_mirror_insert(content, source_type)
        elif action == "remove":
            self._enqueue_mirror_soft_delete(source_type)
        # Unknown action → no-op

    def _enqueue_mirror_insert(
        self,
        content: str,
        source_type: str,
    ) -> None:
        """Enqueue an insert of a mirrored memory row."""
        assert self._writer is not None

        params = self._build_insert_params(
            content,
            origin="user_stated",
            source_type=source_type,
            source_ref=self._session_id or "",
            importance=0.6,
        )

        def _job(conn):
            conn.execute(_INSERT_MEMORY_SQL, params)

        self._writer.enqueue(_job, block=False)

    def _enqueue_mirror_soft_delete(self, source_type: str) -> None:
        """Enqueue a soft-delete of all non-deleted mirrors for a target.

        Soft-deletes rows matching the source_type and the current
        profile. Other source types (conversation turns, captures,
        etc.) are untouched.
        """
        if self._writer is None:
            return

        profile = self._profile
        now_iso = datetime.now(timezone.utc).isoformat()

        def _job(conn):
            conn.execute(
                """
                UPDATE memories
                SET deleted_at = ?
                WHERE source_type = ?
                  AND profile = ?
                  AND deleted_at IS NULL
                """,
                (now_iso, source_type, profile),
            )

        self._writer.enqueue(_job, block=False)

    # ── Commitment detection pipeline (Phase 2B.2) ────────────────────

    def _enqueue_detect_commitments(
        self,
        text: str,
        *,
        source_ref: str,
    ) -> None:
        """Submit text to the cognitive executor for commitment detection.

        Shares the executor with entity extraction (2B.1). Both
        pipelines push their results back through the writer queue,
        so flush() and shutdown() drain them together.

        Called with USER content only (not the combined turn). See
        the sync_turn docstring for the rationale: commitments come
        from the user; Claudia's responses go through approval
        before becoming commitments.
        """
        if self._cognitive_pool is None or self._commitment_detector is None:
            return
        if not text or not text.strip():
            return

        future = self._cognitive_pool.submit(
            self._run_detect_commitments, text, source_ref
        )

        with self._cognitive_lock:
            self._pending_cognitive_futures.append(future)

        future.add_done_callback(self._on_cognitive_done)

    def _run_detect_commitments(
        self,
        text: str,
        source_ref: str,
    ) -> None:
        """Worker body: call detector, enqueue commitment inserts.

        Runs on the cognitive worker thread. Any exception is logged
        and swallowed. Detection is best-effort and must never
        block the memory insert path or kill the worker.
        """
        if self._commitment_detector is None:
            return

        try:
            detected = self._commitment_detector.detect(
                text, source_ref=source_ref
            )
        except Exception:  # pragma: no cover - detector contract says no raise
            logger.exception(
                "claudia commitment detector raised (contract violation)"
            )
            return

        if not detected:
            return

        for c in detected:
            self._enqueue_insert_commitment(c)

    def _enqueue_insert_commitment(
        self,
        c: DetectedCommitment,
    ) -> None:
        """Enqueue a commitment insert on the writer queue.

        Runs on the cognitive worker thread. Captures all needed
        values at enqueue time so profile/state changes between
        submit and execute cannot corrupt the write.

        FK fields (owner_entity_id, target_entity_id) are left NULL
        on first write. Phase 2B.3 consolidation will resolve
        entity references once extraction has populated the
        entities table.
        """
        writer = self._writer
        if writer is None:
            return

        profile = self._profile
        content = c.content
        deadline = c.deadline_iso  # only the parsed ISO form; None if unparseable
        source_type = "conversation"
        source_ref = c.source_ref

        def _job(conn):
            try:
                commitments_module.create_commitment(
                    conn,
                    content,
                    deadline=deadline,
                    source_type=source_type,
                    source_ref=source_ref,
                    profile=profile,
                )
            except ValueError as exc:
                # Shouldn't happen — coercer produces only valid
                # statuses — but log defensively.
                logger.debug(
                    "create_commitment rejected content=%r: %s",
                    content, exc,
                )

        writer.enqueue(_job, block=False)
