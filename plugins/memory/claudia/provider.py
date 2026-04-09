"""ClaudiaMemoryProvider — MemoryProvider ABC implementation (Phase 2A.2f).

The final sub-task of Phase 2A.2: wires the five implementation
modules (schema, embeddings, hybrid_search, entities, offline) into
a single class that satisfies ``agent/memory_provider.MemoryProvider``
and registers with the plugin system via ``register(ctx)``.

Layered design, from bottom to top:

1. ``schema.ensure_database(db_path)`` — opens a WAL-mode connection,
   applies pending migrations. One per provider instance. Lives for
   the duration of the session.
2. ``embeddings.OllamaEmbedder`` — lazy Ollama client. Probed on
   first embed call, cached thereafter.
3. ``offline.OfflineRouter(embedder)`` — decides which
   ``HybridWeights`` preset to use each query.
4. ``hybrid_search.search`` — called via ``router.search`` or
   directly for the 50/25/10/15 composite scoring.
5. ``entities.*`` — entity CRUD for the memory.about tool handler
   and for extraction during sync_turn.

Lifecycle:

- ``initialize(session_id, **kwargs)`` reads ``claudia_home``,
  resolves a profile name, opens the DB, and constructs the
  embedder + router. Does NOT probe Ollama (the ABC forbids
  network I/O in initialize).
- ``prefetch(query)`` and ``sync_turn(u, a)`` are the hot paths —
  called before and after each API call respectively.
- ``handle_tool_call`` dispatches memory.recall / memory.remember
  / memory.about.
- ``shutdown()`` closes the connection.

Profile resolution priority: ``user_id`` > ``agent_identity`` >
``agent_workspace`` > ``"default"``. The design doc calls for
per-user memory isolation in gateway sessions, which means the
platform user id wins when present; otherwise the provider falls
back to the agent's identity scope.

agent_context filtering: writes are skipped for non-primary
contexts (cron, subagent, flush) per the ABC guidance. Reads are
always allowed — a subagent can recall from the parent's memory,
it just can't write back.

Testability:

The embedder and router are constructed via ``_make_embedder`` and
``_make_router`` methods. Tests subclass ``ClaudiaMemoryProvider``
and override these to inject scripted fakes without touching
``httpx`` or a real Ollama daemon.

Reference: docs/decisions/memory-provider-design.md (Phase 2A.2f)
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from plugins.memory.claudia import entities, hybrid_search, schema
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.offline import MemoryMode, OfflineRouter

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


# ─── Provider implementation ────────────────────────────────────────────


class ClaudiaMemoryProvider(MemoryProvider):
    """Claudia's hybrid memory provider plugin."""

    def __init__(self) -> None:
        self._conn: Optional[sqlite3.Connection] = None
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
        """Open the DB, build the embedder, and wire up the router.

        Required kwargs per the ABC: ``claudia_home``, ``platform``.
        Optional: ``agent_context``, ``agent_identity``,
        ``agent_workspace``, ``parent_session_id``, ``user_id``.
        """
        self._session_id = session_id
        self._agent_context = kwargs.get("agent_context", "primary")

        self._claudia_home = self._resolve_claudia_home(kwargs)
        self._profile = self._resolve_profile(kwargs)

        self._db_path = self._claudia_home / "memory" / "claudia" / "claudia.db"
        self._conn = schema.ensure_database(self._db_path)

        self._embedder = self._make_embedder()
        self._router = self._make_router(self._embedder)

        logger.info(
            "Claudia memory provider initialized: session=%s profile=%s context=%s db=%s",
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

    # ── Profile and home resolution ───────────────────────────────────

    @staticmethod
    def _resolve_claudia_home(kwargs: Dict[str, Any]) -> Path:
        """Honor the explicit kwarg, falling back to the constants helper."""
        explicit = kwargs.get("claudia_home")
        if explicit:
            return Path(explicit).expanduser()
        # Lazy import — avoid forcing the claudia_constants dependency
        # at module import time (tests may not have the full fork on
        # sys.path when they subclass this provider).
        try:
            from claudia_constants import get_claudia_home

            return get_claudia_home()
        except ImportError:  # pragma: no cover - defensive fallback
            return Path.home() / ".claudia"

    @staticmethod
    def _resolve_profile(kwargs: Dict[str, Any]) -> str:
        """Priority: user_id > agent_identity > agent_workspace > 'default'.

        Platform user id wins when present so a gateway session with
        multiple users gets per-user memory isolation automatically.
        Otherwise the agent's identity or workspace scope is used.
        """
        for key in ("user_id", "agent_identity", "agent_workspace"):
            value = kwargs.get(key)
            if value:
                return str(value)
        return "default"

    # ── Optional lifecycle hooks ──────────────────────────────────────

    def system_prompt_block(self) -> str:
        """Inject mode + compact stats into the system prompt.

        Budgeted ~500 tokens per the design doc, but in practice this
        stays well under 100 tokens because the stats are just counts.
        """
        if self._conn is None or self._router is None:
            return ""

        stats = schema.describe_schema(self._conn)
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

        The router handles mode selection and the embed-then-search
        orchestration; we just format the ``SearchResult`` list as
        compact bullets with score, importance, and provenance.
        Returns an empty string when there are no matches or the
        provider is not yet initialized.
        """
        if self._conn is None or self._router is None:
            return ""
        if not query or not query.strip():
            return ""

        results = self._router.search(
            self._conn,
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
        """Store the completed turn as a memory row.

        Skipped entirely for non-primary agent contexts (cron,
        subagent, flush) per the ABC guidance — those contexts
        should read but not write. A subagent's ephemeral work
        does not belong in the user's long-term memory.
        """
        if self._conn is None:
            return
        if self._agent_context != "primary":
            return

        combined = self._format_turn(user_content, assistant_content)
        if not combined:
            return

        self._insert_memory(
            combined,
            origin="extracted",
            source_type="conversation",
            source_ref=session_id or self._session_id or "",
            importance=0.5,
        )

    def shutdown(self) -> None:
        """Close the DB connection and clear state."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as exc:  # pragma: no cover - defensive
                logger.debug("Error closing claudia memory DB: %s", exc)
            self._conn = None

    # ── Tool call dispatch ────────────────────────────────────────────

    def handle_tool_call(
        self,
        tool_name: str,
        args: Dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Dispatch a memory.* tool call to its handler. Returns JSON string."""
        if self._conn is None:
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

        assert self._conn is not None and self._router is not None
        results = self._router.search(
            self._conn,
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

        memory_id = self._insert_memory(
            content,
            origin="user_stated",
            source_type=source_type,
            source_ref=self._session_id or "",
            importance=importance,
        )

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

        assert self._conn is not None
        entity = entities.find_entity(
            self._conn,
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
            self._conn,
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

    # ── Internal helpers ──────────────────────────────────────────────

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

    def _insert_memory(
        self,
        content: str,
        *,
        origin: str,
        source_type: str,
        source_ref: str,
        importance: float,
    ) -> int:
        """Insert a memory row, generating an embedding if Ollama is available.

        Returns the new memory id. Safe to call even when the embedder
        is offline — the embedding columns will be NULL and
        hybrid_search will fall back to the non-vector signals.
        """
        assert self._conn is not None

        embedding_blob: Optional[bytes] = None
        embedding_dim: Optional[int] = None
        if self._embedder is not None:
            result = self._embedder.embed(content)
            if result is not None:
                embedding_blob, embedding_dim = result

        now_iso = datetime.now(timezone.utc).isoformat()
        cur = self._conn.execute(
            """
            INSERT INTO memories (
                content, origin, confidence, importance, access_count,
                embedding, embedding_dim, source_type, source_ref,
                profile, created_at, accessed_at
            ) VALUES (?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                content,
                origin,
                0.9,
                importance,
                embedding_blob,
                embedding_dim,
                source_type,
                source_ref,
                self._profile,
                now_iso,
                now_iso,
            ),
        )
        return int(cur.lastrowid)
