"""Unit tests for plugins/memory/claudia/provider.py (Phase 2A.2f).

Covers the full ABC surface of ``ClaudiaMemoryProvider``:

- name and is_available
- initialize creates DB, wires embedder + router
- Profile resolution priority (user_id > agent_identity > agent_workspace > default)
- claudia_home precedence (explicit kwarg wins)
- agent_context filters sync_turn writes (skips cron/subagent/flush)
- get_tool_schemas returns the three memory.* tools in OpenAI format
- system_prompt_block contains mode + stats
- prefetch returns formatted bullets with score/importance/provenance
- prefetch returns empty string when nothing is stored
- handle_tool_call dispatches memory.recall / memory.remember / memory.about
- memory.recall with real data
- memory.remember stores a row with optional importance + source_type
- memory.about looks up entities by name/alias, returns relationships
- memory.about with unknown name returns null result
- handle_tool_call on unknown tool returns error JSON
- handle_tool_call before initialize returns error JSON
- shutdown closes the connection
- register() attaches a provider to a fake ctx
- ClaudiaMemoryProvider is re-exported from the package __init__

Tests use a ``_TestProvider`` subclass that overrides ``_make_embedder``
to inject a scripted ``_FakeEmbedder`` (same pattern as the other
sub-task tests). That keeps the tests offline and fast.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plugins.memory.claudia import (
    ClaudiaMemoryProvider,
    entities as entities_module,
    register,
)
from plugins.memory.claudia.embeddings import OllamaEmbedder
from plugins.memory.claudia.offline import MemoryMode
from plugins.memory.claudia.provider import (
    MEMORY_ABOUT_SCHEMA,
    MEMORY_RECALL_SCHEMA,
    MEMORY_REMEMBER_SCHEMA,
)


# ─── Scripted fake embedder ─────────────────────────────────────────────


class _FakeEmbedder(OllamaEmbedder):
    """OllamaEmbedder with a scripted ``_call_embed``.

    Unlike the other test files, this one defaults to an unlimited
    script of successful probes so the provider can insert memories
    and run recalls without script-exhaustion assertions. Pass an
    explicit ``script`` if you want to test specific failure
    sequences.
    """

    def __init__(self, script=None, **kwargs):
        super().__init__(**kwargs)
        self._script = list(script) if script is not None else None
        self.call_count = 0

    def _call_embed(self, text):  # type: ignore[override]
        self.call_count += 1
        if self._script is None:
            # Default: return a stable 3-dim vector for every call
            return [0.1, 0.2, 0.3]
        if not self._script:
            raise AssertionError(
                f"_call_embed script exhausted; text={text!r}"
            )
        result = self._script.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class _TestProvider(ClaudiaMemoryProvider):
    """Provider subclass that injects a scripted fake embedder.

    Saves the embedder instance on the class so tests can assert on
    ``provider.embedder.call_count`` after exercising the provider.
    """

    def __init__(self, script=None) -> None:
        super().__init__()
        self._script = script

    def _make_embedder(self) -> OllamaEmbedder:  # type: ignore[override]
        return _FakeEmbedder(script=self._script)


# ─── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture()
def provider(tmp_path) -> _TestProvider:
    p = _TestProvider()
    p.initialize(session_id="test-session", claudia_home=str(tmp_path), platform="cli")
    yield p
    p.shutdown()


@pytest.fixture()
def fresh_provider(tmp_path):
    """Factory fixture for tests that need multiple provider instances."""

    instances = []

    def make(**kwargs) -> _TestProvider:
        p = kwargs.pop("_cls", _TestProvider)()
        defaults = {
            "session_id": "test-session",
            "claudia_home": str(tmp_path),
            "platform": "cli",
        }
        defaults.update(kwargs)
        session_id = defaults.pop("session_id")
        p.initialize(session_id=session_id, **defaults)
        instances.append(p)
        return p

    yield make
    for p in instances:
        p.shutdown()


# ─── Required ABC members ───────────────────────────────────────────────


class TestBasicProperties:
    def test_name(self, provider):
        assert provider.name == "claudia"

    def test_is_available(self, provider):
        assert provider.is_available() is True

    def test_is_available_before_initialize(self):
        p = ClaudiaMemoryProvider()
        # Still True — availability is unconditional, not based on init state
        assert p.is_available() is True

    def test_get_tool_schemas_returns_three(self, provider):
        schemas = provider.get_tool_schemas()
        assert len(schemas) == 3
        names = {s["name"] for s in schemas}
        assert names == {"memory.recall", "memory.remember", "memory.about"}

    def test_tool_schemas_are_openai_format(self, provider):
        for schema in provider.get_tool_schemas():
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema
            assert schema["parameters"]["type"] == "object"
            assert "properties" in schema["parameters"]


# ─── initialize() ───────────────────────────────────────────────────────


class TestInitialize:
    def test_creates_db_file(self, tmp_path):
        p = _TestProvider()
        p.initialize(session_id="s1", claudia_home=str(tmp_path), platform="cli")
        db_path = tmp_path / "memory" / "claudia" / "claudia.db"
        assert db_path.exists()
        p.shutdown()

    def test_nested_dirs_created(self, tmp_path):
        p = _TestProvider()
        p.initialize(session_id="s1", claudia_home=str(tmp_path / "nested"), platform="cli")
        assert (tmp_path / "nested" / "memory" / "claudia").is_dir()
        p.shutdown()

    def test_session_id_tracked(self, provider):
        assert provider._session_id == "test-session"

    def test_default_profile(self, provider):
        assert provider._profile == "default"

    def test_default_agent_context(self, provider):
        assert provider._agent_context == "primary"


# ─── Profile resolution ─────────────────────────────────────────────────


class TestProfileResolution:
    def test_user_id_wins(self, fresh_provider):
        p = fresh_provider(
            user_id="alice",
            agent_identity="coder",
            agent_workspace="claudia",
        )
        assert p._profile == "alice"

    def test_agent_identity_wins_over_workspace(self, fresh_provider):
        p = fresh_provider(agent_identity="coder", agent_workspace="claudia")
        assert p._profile == "coder"

    def test_workspace_fallback(self, fresh_provider):
        p = fresh_provider(agent_workspace="claudia")
        assert p._profile == "claudia"

    def test_default_when_none_provided(self, fresh_provider):
        p = fresh_provider()
        assert p._profile == "default"

    def test_empty_values_skipped(self, fresh_provider):
        """Empty strings shouldn't win over later fields."""
        p = fresh_provider(user_id="", agent_identity="coder")
        assert p._profile == "coder"


# ─── agent_context filtering ────────────────────────────────────────────


class TestAgentContextFilter:
    def test_primary_writes(self, fresh_provider):
        p = fresh_provider(agent_context="primary")
        p.sync_turn("hello", "world")
        # One memory written
        conn = p._conn
        row = conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        assert row["n"] == 1

    def test_cron_skips_writes(self, fresh_provider):
        p = fresh_provider(agent_context="cron")
        p.sync_turn("hello", "world")
        row = p._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        assert row["n"] == 0

    def test_subagent_skips_writes(self, fresh_provider):
        p = fresh_provider(agent_context="subagent")
        p.sync_turn("hello", "world")
        row = p._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        assert row["n"] == 0

    def test_flush_skips_writes(self, fresh_provider):
        p = fresh_provider(agent_context="flush")
        p.sync_turn("hello", "world")
        row = p._conn.execute("SELECT COUNT(*) AS n FROM memories").fetchone()
        assert row["n"] == 0


# ─── system_prompt_block ────────────────────────────────────────────────


class TestSystemPromptBlock:
    def test_contains_mode(self, provider):
        block = provider.system_prompt_block()
        assert "Claudia Memory" in block
        # With a fake embedder that returns successful probes, mode
        # should be full_hybrid
        assert MemoryMode.FULL_HYBRID.value in block

    def test_contains_counts(self, provider):
        block = provider.system_prompt_block()
        assert "memories" in block
        assert "entities" in block
        assert "relationships" in block
        assert "commitments" in block

    def test_empty_before_initialize(self):
        p = ClaudiaMemoryProvider()
        assert p.system_prompt_block() == ""


# ─── prefetch ───────────────────────────────────────────────────────────


class TestPrefetch:
    def test_empty_query_returns_empty(self, provider):
        provider.sync_turn("hello world", "hi there")
        assert provider.prefetch("") == ""
        assert provider.prefetch("   ") == ""

    def test_no_results_returns_empty(self, provider):
        assert provider.prefetch("anything") == ""

    def test_formats_results_as_bullets(self, provider):
        provider.sync_turn("my favorite color is blue", "noted")
        block = provider.prefetch("favorite color")
        assert "Claudia Memory" in block
        assert "my favorite color is blue" in block
        assert "score=" in block
        assert "importance=" in block

    def test_includes_source_type(self, provider):
        provider.sync_turn("test memory", "ok")
        block = provider.prefetch("test")
        assert "[conversation]" in block


# ─── handle_tool_call dispatch ──────────────────────────────────────────


class TestHandleToolCall:
    def test_unknown_tool_returns_error(self, provider):
        result = provider.handle_tool_call("memory.nonexistent", {})
        data = json.loads(result)
        assert "error" in data
        assert "unknown tool" in data["error"].lower()

    def test_before_initialize_returns_error(self):
        p = ClaudiaMemoryProvider()
        result = p.handle_tool_call("memory.recall", {"query": "x"})
        data = json.loads(result)
        assert "error" in data
        assert "not initialized" in data["error"].lower()

    def test_returns_json_string(self, provider):
        result = provider.handle_tool_call("memory.recall", {"query": "test"})
        assert isinstance(result, str)
        json.loads(result)  # must parse


# ─── memory.recall ──────────────────────────────────────────────────────


class TestMemoryRecall:
    def test_missing_query_returns_error(self, provider):
        result = provider.handle_tool_call("memory.recall", {})
        data = json.loads(result)
        assert "error" in data

    def test_empty_query_returns_error(self, provider):
        result = provider.handle_tool_call("memory.recall", {"query": "  "})
        data = json.loads(result)
        assert "error" in data

    def test_recalls_stored_memories(self, provider):
        provider.sync_turn("the cat sat on the mat", "ok")
        provider.sync_turn("the sky is blue", "noted")

        result = provider.handle_tool_call(
            "memory.recall", {"query": "cat mat"}
        )
        data = json.loads(result)
        assert "results" in data
        assert len(data["results"]) >= 1
        contents = [r["content"] for r in data["results"]]
        assert any("cat sat on the mat" in c for c in contents)

    def test_limit_clamped(self, provider):
        for i in range(20):
            provider.sync_turn(f"memory number {i}", "ok")

        # Way over the clamp ceiling
        result = provider.handle_tool_call(
            "memory.recall", {"query": "memory", "limit": 999}
        )
        data = json.loads(result)
        assert len(data["results"]) <= 50

    def test_result_fields(self, provider):
        provider.handle_tool_call(
            "memory.remember",
            {"content": "important fact", "importance": 0.9},
        )
        result = provider.handle_tool_call(
            "memory.recall", {"query": "important"}
        )
        data = json.loads(result)
        assert len(data["results"]) >= 1
        row = data["results"][0]
        assert "content" in row
        assert "score" in row
        assert "importance" in row
        assert "source_type" in row
        assert "source_ref" in row
        assert "access_count" in row


# ─── memory.remember ────────────────────────────────────────────────────


class TestMemoryRemember:
    def test_missing_content_returns_error(self, provider):
        result = provider.handle_tool_call("memory.remember", {})
        data = json.loads(result)
        assert "error" in data

    def test_stores_with_default_importance(self, provider):
        result = provider.handle_tool_call(
            "memory.remember", {"content": "Claude is helpful"}
        )
        data = json.loads(result)
        assert data["result"] == "remembered"
        assert "memory_id" in data

    def test_importance_clamped(self, provider):
        # Above 1.0
        result = provider.handle_tool_call(
            "memory.remember", {"content": "x", "importance": 5.0}
        )
        mem_id = json.loads(result)["memory_id"]
        row = provider._conn.execute(
            "SELECT importance FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        assert row["importance"] == 1.0

        # Below 0.0
        result = provider.handle_tool_call(
            "memory.remember", {"content": "y", "importance": -0.5}
        )
        mem_id = json.loads(result)["memory_id"]
        row = provider._conn.execute(
            "SELECT importance FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        assert row["importance"] == 0.0

    def test_custom_source_type(self, provider):
        result = provider.handle_tool_call(
            "memory.remember",
            {"content": "from email", "source_type": "gmail"},
        )
        mem_id = json.loads(result)["memory_id"]
        row = provider._conn.execute(
            "SELECT source_type FROM memories WHERE id = ?", (mem_id,)
        ).fetchone()
        assert row["source_type"] == "gmail"


# ─── memory.about ───────────────────────────────────────────────────────


class TestMemoryAbout:
    def test_missing_name_returns_error(self, provider):
        result = provider.handle_tool_call("memory.about", {})
        data = json.loads(result)
        assert "error" in data

    def test_unknown_entity_returns_null(self, provider):
        result = provider.handle_tool_call(
            "memory.about", {"name": "Nobody"}
        )
        data = json.loads(result)
        assert data["result"] is None
        assert "message" in data

    def test_finds_entity_by_name(self, provider):
        entities_module.create_entity(
            provider._conn,
            "person",
            "Sarah",
            attributes={"role": "PM"},
        )
        result = provider.handle_tool_call(
            "memory.about", {"name": "Sarah"}
        )
        data = json.loads(result)
        assert data["result"] is not None
        assert data["result"]["name"] == "Sarah"
        assert data["result"]["kind"] == "person"
        assert data["result"]["attributes"] == {"role": "PM"}

    def test_finds_entity_by_alias(self, provider):
        entities_module.create_entity(
            provider._conn,
            "person",
            "Sarah Chen",
            aliases=["Sarah", "schen"],
        )
        result = provider.handle_tool_call(
            "memory.about", {"name": "schen"}
        )
        data = json.loads(result)
        assert data["result"] is not None
        assert data["result"]["name"] == "Sarah Chen"

    def test_kind_filter(self, provider):
        entities_module.create_entity(provider._conn, "person", "Mercury")
        entities_module.create_entity(provider._conn, "project", "Mercury")

        person_result = provider.handle_tool_call(
            "memory.about", {"name": "Mercury", "kind": "person"}
        )
        project_result = provider.handle_tool_call(
            "memory.about", {"name": "Mercury", "kind": "project"}
        )

        assert json.loads(person_result)["result"]["kind"] == "person"
        assert json.loads(project_result)["result"]["kind"] == "project"

    def test_includes_relationships(self, provider):
        alice = entities_module.create_entity(provider._conn, "person", "Alice")
        bob = entities_module.create_entity(provider._conn, "person", "Bob")
        entities_module.create_relationship(
            provider._conn, alice.id, bob.id, "colleague", health_score=0.8
        )

        result = provider.handle_tool_call(
            "memory.about", {"name": "Alice"}
        )
        data = json.loads(result)
        rels = data["result"]["relationships"]
        assert len(rels) == 1
        assert rels[0]["type"] == "colleague"
        assert rels[0]["to_entity_id"] == bob.id


# ─── shutdown ───────────────────────────────────────────────────────────


class TestShutdown:
    def test_closes_connection(self, tmp_path):
        p = _TestProvider()
        p.initialize(session_id="s1", claudia_home=str(tmp_path), platform="cli")
        assert p._conn is not None

        p.shutdown()
        assert p._conn is None

    def test_shutdown_idempotent(self, tmp_path):
        p = _TestProvider()
        p.initialize(session_id="s1", claudia_home=str(tmp_path), platform="cli")
        p.shutdown()
        p.shutdown()  # no-op, no error


# ─── register() ─────────────────────────────────────────────────────────


class _FakeCtx:
    def __init__(self):
        self.providers = []

    def register_memory_provider(self, provider):
        self.providers.append(provider)


class TestRegister:
    def test_register_attaches_provider(self):
        ctx = _FakeCtx()
        register(ctx)
        assert len(ctx.providers) == 1
        assert isinstance(ctx.providers[0], ClaudiaMemoryProvider)
        assert ctx.providers[0].name == "claudia"

    def test_each_register_creates_fresh_instance(self):
        ctx1 = _FakeCtx()
        ctx2 = _FakeCtx()
        register(ctx1)
        register(ctx2)
        assert ctx1.providers[0] is not ctx2.providers[0]


# ─── Tool schema module-level constants ─────────────────────────────────


class TestToolSchemaConstants:
    def test_recall_schema_name(self):
        assert MEMORY_RECALL_SCHEMA["name"] == "memory.recall"

    def test_remember_schema_name(self):
        assert MEMORY_REMEMBER_SCHEMA["name"] == "memory.remember"

    def test_about_schema_name(self):
        assert MEMORY_ABOUT_SCHEMA["name"] == "memory.about"

    def test_about_kind_enum(self):
        kind_prop = MEMORY_ABOUT_SCHEMA["parameters"]["properties"]["kind"]
        assert set(kind_prop["enum"]) == {
            "person",
            "organization",
            "project",
            "location",
            "concept",
        }
