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

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from plugins.memory.claudia import entities, hybrid_search, schema
from plugins.memory.claudia.embeddings import OllamaEmbedder
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


_ALL_TOOL_SCHEMAS = [
    MEMORY_RECALL_SCHEMA,
    MEMORY_REMEMBER_SCHEMA,
    MEMORY_ABOUT_SCHEMA,
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
        """Run hybrid search on the current turn's query and format the result.

        Acquires a reader connection from the pool, runs the router's
        one-shot search, and formats results as compact bullets. The
        reader connection is returned to the pool on exit, allowing
        other threads to run prefetch in parallel.
        """
        if self._reader_pool is None or self._router is None:
            return ""
        if not query or not query.strip():
            return ""

        with self._reader_pool.acquire() as conn:
            results = self._router.search(
                conn,
                query,
                profile=self._profile,
                limit=10,
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
        return "\n".join(lines)

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
    ) -> None:
        """Store the completed turn as a memory row (async, non-blocking).

        Skipped entirely for non-primary agent contexts (cron,
        subagent, flush) per the ABC guidance — those contexts
        should read but not write.

        Generates the embedding on the caller's thread so HTTP to
        Ollama doesn't stall the writer, then enqueues a
        fire-and-forget insert job.
        """
        if self._writer is None:
            return
        if self._agent_context != "primary":
            return

        combined = self._format_turn(user_content, assistant_content)
        if not combined:
            return

        self._enqueue_insert_memory(
            combined,
            origin="extracted",
            source_type="conversation",
            source_ref=session_id or self._session_id or "",
            importance=0.5,
        )

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for all pending writes to drain.

        Useful for tests, for callers that need read-your-writes
        consistency, and for the shutdown path before stopping the
        writer. Returns True if the queue drained within the timeout.
        """
        if self._writer is None:
            return True
        return self._writer.flush(timeout=timeout)

    def shutdown(self) -> None:
        """Stop the writer (draining pending jobs) and close the reader pool."""
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
