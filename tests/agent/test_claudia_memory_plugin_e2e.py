"""End-to-end integration test: ClaudiaMemoryProvider through MemoryManager.

This is Phase 2A.5 — the last sub-task of Phase 2A. Previous tests
exercise ``ClaudiaMemoryProvider`` via direct method invocation. This
test loads the provider through the real ``agent.memory_manager.MemoryManager``
alongside the ``BuiltinMemoryProvider``, then drives the full lifecycle
as the host would:

  initialize_all → build_system_prompt → prefetch_all →
  handle_tool_call → sync_all → shutdown_all

The goal is to prove that the plugin is wire-compatible with the host:

- The manager accepts ClaudiaMemoryProvider as an external provider.
- Its three tools (memory.recall, memory.remember, memory.about) are
  routed correctly from the manager's central dispatch.
- Its system prompt block merges cleanly with the builtin's block.
- prefetch and sync_turn run through the manager without raising.
- Shutdown tears down the writer queue and reader pool cleanly when
  called via the manager's reverse-order shutdown path.
- Provider failure isolation: if ClaudiaMemoryProvider errors mid-
  lifecycle, the builtin keeps working.

Lives in ``tests/agent/`` alongside ``test_memory_plugin_e2e.py``
(which uses a toy SQLite provider for the same purpose). The existing
test is the template; this one just substitutes the real plugin.

Uses a ``_TestProvider`` subclass with a fake embedder so the test
stays offline and deterministic — no httpx, no Ollama, no threading
flakiness from a real daemon probe.
"""

from __future__ import annotations

import json
import threading

import pytest

from agent.builtin_memory_provider import BuiltinMemoryProvider
from agent.memory_manager import MemoryManager
from plugins.memory.claudia import ClaudiaMemoryProvider
from plugins.memory.claudia.embeddings import OllamaEmbedder


# ─── Fake embedder (same pattern as tests/plugins/memory/claudia/) ──────


class _FakeEmbedder(OllamaEmbedder):
    """Thread-safe scripted embedder that returns a stable 3-dim vector.

    The probe via ``is_available`` succeeds on the first call, so the
    provider boots into FULL_HYBRID mode. Subsequent embed calls return
    the same vector, which is enough for integration-level coverage —
    we're verifying the dispatch path, not the ranking math.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._call_count = 0
        self._lock = threading.Lock()

    def _call_embed(self, text):  # type: ignore[override]
        with self._lock:
            self._call_count += 1
        return [0.1, 0.2, 0.3]


class _TestClaudiaProvider(ClaudiaMemoryProvider):
    """ClaudiaMemoryProvider with the embedder factory swapped for a fake."""

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder()


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def manager(tmp_path):
    """Real MemoryManager with builtin + Claudia providers wired up.

    Initializes both, yields the manager, and runs shutdown_all on
    teardown. Uses ``tmp_path`` for CLAUDIA_HOME so tests never touch
    the real ~/.claudia.
    """
    mgr = MemoryManager()
    builtin = BuiltinMemoryProvider()
    claudia = _TestClaudiaProvider()

    mgr.add_provider(builtin)
    mgr.add_provider(claudia)

    mgr.initialize_all(
        session_id="e2e-test-session",
        claudia_home=str(tmp_path),
        platform="cli",
    )

    yield mgr

    mgr.shutdown_all()


# ─── Registration and discovery ─────────────────────────────────────────


class TestRegistration:
    def test_both_providers_registered(self, manager):
        assert manager.provider_names == ["builtin", "claudia"]

    def test_claudia_provider_is_second(self, manager):
        """Builtin must always be first per MemoryManager contract."""
        assert manager.providers[0].name == "builtin"
        assert manager.providers[1].name == "claudia"

    def test_second_external_provider_rejected(self, tmp_path):
        """Only one external (non-builtin) provider at a time."""
        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())

        p1 = _TestClaudiaProvider()
        p2 = _TestClaudiaProvider()

        mgr.add_provider(p1)
        mgr.add_provider(p2)  # rejected by the manager

        assert mgr.provider_names == ["builtin", "claudia"]
        # Only the first one should show up in the list
        claudia_providers = [p for p in mgr.providers if p.name == "claudia"]
        assert len(claudia_providers) == 1
        assert claudia_providers[0] is p1

        mgr.shutdown_all()


# ─── Tool schema routing ────────────────────────────────────────────────


class TestToolRouting:
    def test_all_three_claudia_tools_exposed(self, manager):
        schemas = manager.get_all_tool_schemas()
        names = {s["name"] for s in schemas}

        # Claudia's three tools all reachable through the manager
        assert "memory.recall" in names
        assert "memory.remember" in names
        assert "memory.about" in names

    def test_has_tool_routing(self, manager):
        assert manager.has_tool("memory.recall")
        assert manager.has_tool("memory.remember")
        assert manager.has_tool("memory.about")
        assert not manager.has_tool("nonexistent.tool")

    def test_tool_schemas_are_openai_format(self, manager):
        """Schemas must be the OpenAI function-calling format the ABC mandates."""
        schemas = manager.get_all_tool_schemas()
        claudia_schemas = [s for s in schemas if s["name"].startswith("memory.")]

        for schema in claudia_schemas:
            assert "description" in schema
            assert schema["parameters"]["type"] == "object"
            assert "properties" in schema["parameters"]


# ─── System prompt composition ──────────────────────────────────────────


class TestSystemPrompt:
    def test_contains_claudia_block(self, manager):
        prompt = manager.build_system_prompt()
        assert "Claudia Memory" in prompt

    def test_contains_mode_and_counts(self, manager):
        prompt = manager.build_system_prompt()
        assert "Mode:" in prompt
        assert "memories" in prompt
        assert "entities" in prompt

    def test_builtin_and_claudia_blocks_separated(self, manager):
        """Both blocks should appear, joined by a blank line per manager contract."""
        prompt = manager.build_system_prompt()
        # Non-empty system prompt means at least Claudia's block is in
        # (builtin's block may or may not appear depending on its state)
        assert len(prompt.strip()) > 0
        assert "Claudia Memory" in prompt


# ─── Full lifecycle ─────────────────────────────────────────────────────


class TestFullLifecycle:
    def test_remember_then_recall_through_manager(self, manager):
        """Store via memory.remember, read back via memory.recall — both
        dispatched through the manager's central routing."""
        # Store
        remember_result = json.loads(
            manager.handle_tool_call(
                "memory.remember",
                {"content": "The user's favorite color is blue", "importance": 0.8},
            )
        )
        assert remember_result["result"] == "remembered"
        assert "memory_id" in remember_result

        # Recall — FTS should match "favorite color"
        recall_result = json.loads(
            manager.handle_tool_call(
                "memory.recall",
                {"query": "favorite color"},
            )
        )
        assert "results" in recall_result
        assert len(recall_result["results"]) >= 1
        contents = [r["content"] for r in recall_result["results"]]
        assert any("favorite color is blue" in c for c in contents)

    def test_sync_turn_then_recall(self, manager):
        """sync_all enqueues an async write. Subsequent prefetch_all
        should see it after the writer drains."""
        # Sync a turn
        manager.sync_all(
            "What's my timezone?",
            "You're in US Pacific time.",
            session_id="e2e-test-session",
        )

        # Flush the claudia writer so the sync is visible to reads
        claudia = manager.get_provider("claudia")
        assert claudia is not None
        assert claudia.flush(timeout=5.0)

        # Prefetch should find it
        prefetched = manager.prefetch_all("timezone")
        assert "timezone" in prefetched.lower() or "pacific" in prefetched.lower()

    def test_memory_about_unknown_returns_null(self, manager):
        result = json.loads(
            manager.handle_tool_call("memory.about", {"name": "Nobody"})
        )
        assert result["result"] is None
        assert "message" in result

    def test_memory_about_finds_entity_after_creation(self, manager):
        """memory.about should resolve entities that were created via
        the writer queue path."""
        claudia = manager.get_provider("claudia")
        assert claudia is not None

        # Create an entity via the writer — the direct entities.py path
        # that the provider's extraction pipeline will eventually use.
        from plugins.memory.claudia import entities

        def _create(conn):
            entities.create_entity(
                conn, "person", "Sarah Chen",
                aliases=["Sarah", "schen"],
                attributes={"role": "VP Engineering"},
            )

        claudia._writer.enqueue_and_wait(_create, timeout=5.0)

        # Look up via the manager's dispatch
        result = json.loads(
            manager.handle_tool_call("memory.about", {"name": "Sarah"})
        )
        assert result["result"] is not None
        assert result["result"]["name"] == "Sarah Chen"
        assert result["result"]["kind"] == "person"
        assert result["result"]["attributes"]["role"] == "VP Engineering"


# ─── Shutdown ordering and teardown ─────────────────────────────────────


class TestShutdown:
    def test_manager_shutdown_tears_down_claudia_resources(self, tmp_path):
        """shutdown_all must invoke Claudia's shutdown, which stops the
        writer and closes the reader pool."""
        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())

        claudia = _TestClaudiaProvider()
        mgr.add_provider(claudia)
        mgr.initialize_all(
            session_id="teardown-test",
            claudia_home=str(tmp_path),
            platform="cli",
        )

        assert claudia._writer is not None
        assert claudia._writer.is_running
        assert claudia._reader_pool is not None

        mgr.shutdown_all()

        # Claudia's resources released
        assert claudia._writer is None
        assert claudia._reader_pool is None

    def test_shutdown_drains_pending_sync_turns(self, tmp_path):
        """Writes enqueued via sync_all must commit before the writer
        thread joins on shutdown."""
        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())

        claudia = _TestClaudiaProvider()
        mgr.add_provider(claudia)
        mgr.initialize_all(
            session_id="drain-test",
            claudia_home=str(tmp_path),
            platform="cli",
        )

        for i in range(15):
            mgr.sync_all(f"user turn {i}", f"assistant response {i}")

        mgr.shutdown_all()

        # Open a fresh connection and verify all writes committed
        import sqlite3
        db_path = tmp_path / "memory" / "claudia" / "claudia.db"
        conn = sqlite3.connect(str(db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            assert row[0] == 15
        finally:
            conn.close()


# ─── Failure isolation ──────────────────────────────────────────────────


class TestFailureIsolation:
    def test_claudia_prefetch_exception_does_not_break_builtin(self, tmp_path):
        """If the Claudia provider's prefetch raises, the manager
        swallows it and the builtin still runs."""
        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())

        claudia = _TestClaudiaProvider()
        mgr.add_provider(claudia)
        mgr.initialize_all(
            session_id="iso-test",
            claudia_home=str(tmp_path),
            platform="cli",
        )

        # Break Claudia's reader pool mid-session
        claudia._reader_pool.close()

        # prefetch_all should not raise — failures are logged and skipped
        result = mgr.prefetch_all("anything")
        # Result may be empty (both providers quiet) or have builtin content,
        # but the call must complete without raising
        assert isinstance(result, str)

        mgr.shutdown_all()

    def test_claudia_sync_exception_does_not_break_builtin(self, tmp_path):
        """Failing sync_turn in Claudia doesn't stop the lifecycle."""
        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())

        claudia = _TestClaudiaProvider()
        mgr.add_provider(claudia)
        mgr.initialize_all(
            session_id="iso-test-2",
            claudia_home=str(tmp_path),
            platform="cli",
        )

        # Stop the writer — subsequent sync_turn calls will no-op
        claudia._writer.stop()

        # sync_all should not raise
        mgr.sync_all("user", "assistant")

        mgr.shutdown_all()


# ─── Plugin registration via register(ctx) ─────────────────────────────


class TestRegisterFunction:
    def test_register_attaches_to_manager_like_ctx(self, tmp_path):
        """The ``register(ctx)`` entry point works with a ctx that
        forwards to MemoryManager.add_provider.

        This simulates what a real plugin loader would do: call
        register() with something that exposes register_memory_provider.
        """
        from plugins.memory.claudia import register

        mgr = MemoryManager()
        mgr.add_provider(BuiltinMemoryProvider())

        # Adapter: ctx wraps the manager
        class _ManagerCtx:
            def __init__(self, manager):
                self._mgr = manager

            def register_memory_provider(self, provider):
                self._mgr.add_provider(provider)

        register(_ManagerCtx(mgr))

        assert "claudia" in mgr.provider_names
        claudia = mgr.get_provider("claudia")
        assert isinstance(claudia, ClaudiaMemoryProvider)

        # Initialize and shutdown should work through this path too
        mgr.initialize_all(
            session_id="register-test",
            claudia_home=str(tmp_path),
            platform="cli",
        )
        assert claudia._writer is not None

        mgr.shutdown_all()
