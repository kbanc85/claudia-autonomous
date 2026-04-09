# Claudia memory provider design (Phase 2A.1 output)

**Author**: Phase 2A.1 study session, 2026-04-09
**Source of truth**: `agent/memory_provider.py` ABC (231 lines) as of submodule commit `98578fc`

## Executive summary

The v0.7.0 pluggable memory provider interface is clean, well-documented, and gives Claudia exactly what it needs. **Claudia's hybrid memory system can ship as a plugin at `plugins/memory/claudia/` without touching any core Hermes code.** The Phase 2A.2 implementation work is eight well-defined Python modules behind a known abstraction boundary.

The roadmap's 7-10 day Phase 2A estimate is achievable for one Claude Code session per sub-component (2A.2a through 2A.2f) plus integration.

## ABC contract summary

### Required (abstract) methods

| Method | Purpose | Notes |
|---|---|---|
| `name` (property) | Short identifier | Must return `"claudia"` for this plugin |
| `is_available()` | Config + dep check | No network calls. Check for SQLite, sqlite-vec extension, ollama binary |
| `initialize(session_id, **kwargs)` | Session startup | Create DB, schema, load embedding model, start background threads |
| `get_tool_schemas()` | Tool definitions | OpenAI function-calling format |

### Lifecycle hooks (optional but Claudia needs them)

| Method | Claudia's use |
|---|---|
| `system_prompt_block()` | Inject ~500 tokens of memory guidance + memory stats into system prompt |
| `prefetch(query, *, session_id)` | **Primary recall path**. Run hybrid search, return ~1500 tokens of top memories |
| `queue_prefetch(query, *, session_id)` | Background embedding of next query for warm cache |
| `sync_turn(user_content, assistant_content, *, session_id)` | Store conversation turn, extract commitments, update entity relationships |
| `handle_tool_call(tool_name, args, **kwargs)` | Dispatch `memory.recall`, `memory.remember`, `memory.about`, etc. |
| `shutdown()` | Flush write queue, close DB, stop threads |

### Optional hooks (Claudia should implement)

| Hook | Claudia's use |
|---|---|
| `on_turn_start(turn, message, **kwargs)` | Detect commitment language in user message before LLM call |
| `on_session_end(messages)` | Generate session summary, update relationship health scores |
| `on_pre_compress(messages)` | Extract high-importance facts from about-to-be-compressed history |
| `on_delegation(task, result, **kwargs)` | Track subagent work as ephemeral context |
| `on_memory_write(action, target, content)` | Mirror built-in memory writes into Claudia's SQLite store so nothing gets lost |
| `get_config_schema()` | Config fields for `claudia memory setup`: embedding model choice, importance weights, offline mode |
| `save_config(values, claudia_home)` | Write `plugins/memory/claudia/config.yaml` inside CLAUDIA_HOME |

## Architecture constraints from the ABC

1. **Built-in memory (MEMORY.md/USER.md) is always active and cannot be replaced.** Claudia's provider is additive. The `on_memory_write` hook lets Claudia mirror built-in writes into its SQLite store so a user searching Claudia's memory can find things written via the built-in tool.

2. **Only one external provider runs at a time.** Users pick `claudia` OR `honcho` OR `hindsight`, not multiple. This avoids tool schema bloat and conflicting backends.

3. **Initialize is called once per session with rich context** — `claudia_home`, `platform`, `agent_context`, `agent_identity`, `agent_workspace`, `parent_session_id`, `user_id`. All these let Claudia do profile-scoped and per-user memory isolation without bespoke plumbing.

4. **`agent_context` signals the runtime context**: `"primary"`, `"subagent"`, `"cron"`, or `"flush"`. Providers should skip writes for non-primary contexts so cron system prompts don't corrupt user representations. Claudia must honour this.

5. **`prefetch()` is called before each API call** and should be fast. The recommendation is to use background threads for actual recall and return cached results. Claudia's hybrid search will use a read-side thread pool (WAL mode makes concurrent reads free) and keep a per-session cache.

6. **`sync_turn()` is called after each turn** and should be non-blocking — queue for background processing. Claudia's write queue takes each turn and:
   - Generates embeddings for user and assistant content
   - Extracts entities mentioned in the conversation
   - Detects commitment language and queues for commitment lifecycle
   - Writes to SQLite via a single serialized writer (WAL + write lock)

7. **`handle_tool_call()` returns a JSON string**, not a dict. Keep formatting consistent with how Hermes's built-in tools return results.

8. **`get_tool_schemas()` returns OpenAI function-calling format**, not Anthropic tool_use format. The agent loop converts between formats as needed.

## Proposed plugin layout

```
plugins/memory/claudia/
├── __init__.py              Exports ClaudiaMemoryProvider class
├── plugin.yaml              Plugin metadata (name, version, ABC version, description)
├── provider.py              ClaudiaMemoryProvider(MemoryProvider) implementation
├── schema.py                SQLite schema DDL + migration runner (sub-task 2A.2a)
├── embeddings.py            Ollama integration for all-minilm:l6-v2 (sub-task 2A.2b)
├── hybrid_search.py         Ranking algorithm: 50% vec + 25% imp + 10% recency + 15% fts (2A.2c)
├── entities.py              Entity CRUD for people, orgs, projects (2A.2d)
├── offline.py               Degradation path: full hybrid / fts+imp+recency / fts-only (2A.2e)
├── writer.py                Serialized write queue + WAL concurrency (2A.3)
├── reader.py                Concurrent read path with connection pooling (2A.3)
├── tools.py                 memory.recall / memory.remember / memory.about tool definitions
└── config.py                ClaudiaMemoryConfig dataclass + defaults + user override
```

### plugin.yaml template

```yaml
name: claudia
display_name: "Claudia Hybrid Memory"
version: "0.1.0"
abc_version: "1.0"  # Match agent/memory_provider.py ABC version
description: |
  Claudia's chief-of-staff memory system. SQLite + vector embeddings
  (Ollama all-minilm:l6-v2) with hybrid ranking: 50% vector similarity,
  25% importance, 10% recency, 15% FTS. Includes entity CRUD, commitment
  lifecycle, relationship graphs, and offline degradation.
requires:
  - sqlite-vec
  - ollama  # optional at runtime; plugin degrades gracefully
```

## Sub-task mapping

Each Phase 2A.2 sub-task corresponds to one module above:

| Sub-task | Module | Purpose | Test target |
|---|---|---|---|
| 2A.2a SQLite schema | `schema.py` | Entities, memories, relationships, commitments tables. WAL mode. Importance + access_count columns. | `tests/plugins/memory/claudia/test_schema.py` |
| 2A.2b Embedding pipeline | `embeddings.py` | Ollama integration, offline fallback | `test_embeddings.py` |
| 2A.2c Hybrid search | `hybrid_search.py` | The 50/25/10/15 ranking formula + rehearsal boost | `test_hybrid_search.py` |
| 2A.2d Entity CRUD | `entities.py` | Create/read/update/delete + profile isolation | `test_entities.py` |
| 2A.2e Offline degradation | `offline.py` | Three-tier fallback (full / FTS+imp+recency / FTS-only) | `test_offline.py` |
| 2A.2f Provider registration | `provider.py` + `__init__.py` | Wire the plugin into the MemoryProvider ABC | `test_provider.py` |

## Concurrency design (Phase 2A.3 preview)

The ABC's `prefetch()` and `sync_turn()` both run concurrently from gateway sessions, cron jobs, interactive sessions, and subagents. Claudia's SQLite store must handle 3+ concurrent writers safely.

Approach (to be implemented in `writer.py` and `reader.py`):

1. **WAL mode enabled at DB open.** Concurrent readers are free in WAL; concurrent writers need serialization.
2. **Single serialized writer** — a dedicated `threading.Thread` consuming a queue of write jobs. `sync_turn()` enqueues and returns immediately.
3. **Connection pooling for readers** — `prefetch()` acquires a read connection from a pool (configurable size, default 4).
4. **Per-session reader cache** — `prefetch()` caches recent query results for 30 seconds per session, so the same query from the same gateway session doesn't re-run hybrid search.
5. **Write lock timeout** — writers wait up to 5s on SQLite's write lock, then retry. On persistent timeout, log and drop (we're not a source of truth for the conversation — the user's messages are preserved in the session DB).

Synthetic load test in 2A.3 will simulate 3 concurrent writers (cron, gateway, interactive) hitting the store for 60 seconds and verify no data corruption and bounded latency.

## Prompt budget accounting (Phase 2B.6 preview)

The ABC gives Claudia two injection points:

- `system_prompt_block()` — static per-provider block, budgeted ~500 tokens. Claudia uses this for memory guidance + stats (N memories stored, M entities, P commitments).
- `prefetch(query)` — dynamic per-turn recall, budgeted ~1500 tokens. Claudia returns the top N memories from hybrid search formatted as compact bullets with provenance.

Total Claudia memory contribution to the prompt: ~2000 tokens baseline, ~3300 including relationship context when entities in the current message are known. The roadmap's total baseline (~9300 tokens for all components) has headroom.

## Offline degradation path (Phase 2A.2e preview)

Three execution modes based on runtime availability:

| Mode | When | Recall ranking | Notes |
|---|---|---|---|
| **Full hybrid** | Ollama running, embeddings available | 50% vec + 25% imp + 10% recency + 15% FTS + rehearsal | Normal operation |
| **FTS + importance + recency** | Ollama down, no new embeddings possible | Reweight to 100%: ~50% imp + 20% recency + 30% FTS | Lose semantic recall, keep lexical + importance |
| **Pure FTS** | No vec0 extension loaded | 100% FTS | Minimal but functional; entity CRUD still works |

Mode is detected at `initialize()` and re-checked if an embedding call fails at runtime. Mode switches log a warning but don't error — Claudia keeps running.

## Open questions for Phase 2A.2 sub-tasks

1. **sqlite-vec as hard dep or optional?** Installing `sqlite-vec` is a Python extension + a compiled .so. If we make it optional, the "no vec0 extension loaded" mode becomes a first-class path. Recommendation: optional, with a clear warning in `claudia memory setup` if absent.

2. **Ollama host discovery**: the ABC says `initialize()` shouldn't make network calls. So ollama availability should be checked lazily on the first `prefetch()` call, not in `initialize()`. That means embedding generation on first recall may be slow; acceptable since subsequent calls are warm.

3. **Consolidation job scope (Phase 2B.4)**: should the plugin run its own consolidation cron, or should it hook into the existing Hermes cron scheduler? Recommendation: use Hermes cron so the user sees Claudia consolidation jobs alongside their other scheduled tasks.

4. **Migration from built-in MEMORY.md**: when a user switches from the default built-in provider to the Claudia provider, should old MEMORY.md entries be imported automatically? Recommendation: yes, as a one-time ingestion during first `initialize()`. Use `on_memory_write` to keep MEMORY.md and Claudia in sync thereafter.

5. **Cost governance integration (Phase 2B.5)**: the ABC doesn't expose a hook for token budget enforcement. The `on_turn_start()` hook gets `remaining_tokens` in kwargs — Claudia can read it but can't block the call. Enforcement has to happen in `run_agent.py` itself, which is Phase 2B.5 territory, not Phase 2A.

6. **Subagent memory isolation**: `initialize()` gets `agent_context` ("primary" / "subagent" / "cron") and `parent_session_id`. Subagents should probably SHARE the parent's memory for read (prefetch) but NOT write (skip `sync_turn`). That way a subagent's ephemeral work doesn't pollute the primary memory, but it can still recall context. This is a design call, not an ABC constraint.

## Next steps

Phase 2A.1 is complete. Phase 2A.2 implementation proceeds in six sequential Claude Code sessions:

- **2A.2a**: Implement `schema.py` — SQLite DDL, migration runner, WAL mode setup
- **2A.2b**: Implement `embeddings.py` — Ollama client, all-minilm:l6-v2, offline fallback
- **2A.2c**: Implement `hybrid_search.py` — ranking formula, rehearsal boost, cached per-session
- **2A.2d**: Implement `entities.py` — CRUD with profile isolation
- **2A.2e**: Implement `offline.py` — three-tier fallback logic
- **2A.2f**: Implement `provider.py` + `__init__.py` + `plugin.yaml` — register as `MemoryProvider` subclass

Then Phase 2A.3 (concurrency) and 2A.4 (unit tests ported from Claudia v1's 756-test suite where applicable).

## Reference

- `agent/memory_provider.py` — the ABC source of truth (231 lines)
- `agent/builtin_memory_provider.py` — the always-on reference implementation
- `plugins/memory/hindsight/` — external provider example (registers via plugin system, minimal implementation)
- `plugins/memory/honcho/` — more elaborate external provider with full CLI integration and native config file
- `plugins/memory/byterover/`, `plugins/memory/holographic/` — additional external provider examples
