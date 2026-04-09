# Phase 2A → Phase 2B Handoff

**Created**: 2026-04-09 by the Phase 2A implementation session
**Purpose**: Pick up cleanly after `/clear` with zero context loss

---

## TL;DR

Phase 2A of the Claudia memory provider plugin is **complete and committed** across both repos. The plugin loads through the real `agent.memory_manager.MemoryManager` alongside the builtin provider and exercises its full ABC surface with 305 plugin-specific unit/integration tests passing.

**Next up**: Phase 2B — LLM extraction, commitment detection, auto-consolidation, verification service, cost governance, and prompt budget accounting.

---

## Current state (as of handoff)

| Thing | Value |
|---|---|
| Outer repo branch | `claude/setup-project-structure-bvdSj` |
| Outer repo HEAD | `a5037de autonomous: Phase 2A.5 - e2e via real MemoryManager (Phase 2A COMPLETE)` |
| Submodule (`autonomous/fork`) branch | `main` |
| Submodule HEAD | `b0adcb9 Phase 2A.5: end-to-end integration test via real MemoryManager` |
| Working tree | **Clean** on both repos for the 2A work |
| Plugin suite | **305 tests passing** (full fork + plugin + agent suites: 352 combined) |
| Test runtime | ~6 seconds on Python 3.13 |

**Verify on session resume:**

```bash
cd "/Users/kamil/Documents/Claude Code/Open Source Claudia Agent/autonomous/fork"
git log --oneline -1   # should show b0adcb9
python3 -m pytest tests/plugins/memory/claudia/ tests/agent/test_claudia_memory_plugin_e2e.py -q -p no:xdist -o addopts="-m 'not integration and not frontier and not local'"
# Expect: 305 passed (or 352 if you also run test_memory_plugin_e2e.py and test_memory_provider.py)
```

If the count differs or anything fails, STOP and diagnose before proceeding. Phase 2A is the foundation everything else builds on.

---

## Phase 2A rollback chain

10 commits. Each is a legitimate "ready to ship" state. Any of them can be `git reset --hard` targets in both repos.

| # | Sub-task | Outer | Submodule | What landed |
|---|---|---|---|---|
| 1 | 2A.1 design doc | `6fbe1ab` | (analytical) | `docs/decisions/memory-provider-design.md` |
| 2 | 2A.2a schema | `8de79bc` | `432028b` | DDL + migrations + WAL |
| 3 | 2A.2b embeddings | `becba65` | `e8e509c` | Ollama client + BLOB codec |
| 4 | 2A.2c hybrid_search | `beefda9` | `847d08a` | 50/25/10/15 ranking formula |
| 5 | 2A.2d entities | `18c66ba` | `40e1d7e` | Entity + relationship CRUD |
| 6 | 2A.2e offline | `8ab5a09` | `a1a2d05` | Three-tier degradation router |
| 7 | 2A.2f provider | `dd861d2` | `1e7eb1a` | MemoryProvider ABC subclass |
| 8 | 2A.3 concurrency | `2c777bf` | `d69307e` | Writer queue + reader pool |
| 9 | 2A.4 v1 port | `2daf4bf` | `208174a` | Schema tests + wildcard search |
| 10 | 2A.5 e2e | `a5037de` | `b0adcb9` | Real MemoryManager integration |

---

## File map

### Implementation (`autonomous/fork/plugins/memory/claudia/`)

| File | Role | Approx lines |
|---|---|---|
| `schema.py` | SQLite DDL, migrations, WAL mode, FTS5 | ~330 |
| `embeddings.py` | `OllamaEmbedder` + `pack_embedding`/`unpack_embedding` | ~260 |
| `hybrid_search.py` | `HybridWeights`, `ScoreBreakdown`, `SearchResult`, `search()`, `cosine_similarity`, `build_fts_query` | ~430 |
| `entities.py` | `Entity`/`Relationship` dataclasses + 13 CRUD functions | ~560 |
| `offline.py` | `MemoryMode` enum, `ModeDecision`, `OfflineRouter` | ~240 |
| `writer.py` | `WriterQueue` (serialized background thread) | ~270 |
| `reader.py` | `ReaderPool` (bounded connection pool) | ~180 |
| `provider.py` | `ClaudiaMemoryProvider(MemoryProvider)` + tool schemas | ~530 |
| `__init__.py` | Re-exports + `register(ctx)` entry point | ~50 |
| `plugin.yaml` | Metadata | ~15 |

### Tests

| Location | File | Tests |
|---|---|---|
| `tests/plugins/memory/claudia/` | `test_schema.py` | 17 |
| | `test_embeddings.py` | 31 |
| | `test_hybrid_search.py` | 36 |
| | `test_entities.py` | 72 |
| | `test_offline.py` | 24 |
| | `test_provider.py` | 56 |
| | `test_writer.py` | 23 |
| | `test_reader.py` | 21 |
| | `test_concurrency.py` | 7 |
| `tests/agent/` | `test_claudia_memory_plugin_e2e.py` | 18 |
| **Total** | | **305** |

### Reference material

| What | Where |
|---|---|
| MemoryProvider ABC source | `autonomous/fork/agent/memory_provider.py` |
| MemoryManager | `autonomous/fork/agent/memory_manager.py` |
| Builtin provider (reference impl) | `autonomous/fork/agent/builtin_memory_provider.py` |
| Phase 2A.1 design doc | `autonomous/fork/docs/decisions/memory-provider-design.md` |
| Existing e2e template | `autonomous/fork/tests/agent/test_memory_plugin_e2e.py` |
| Claudia v1 daemon source | `claudia/memory-daemon/claudia_memory/` (outer repo) |
| Claudia v1 daemon tests | `claudia/memory-daemon/tests/` (outer repo, ~60 files) |
| Other external plugins to reference | `autonomous/fork/plugins/memory/{hindsight,honcho,openviking}/` |

---

## Critical design invariants (DO NOT violate in Phase 2B)

These constraints shaped Phase 2A and Phase 2B must honor them. Breaking any of these is a regression, even if tests pass.

### From the MemoryProvider ABC

1. **`initialize()` must not make network calls.** Ollama probing happens lazily on first `embed()` call via `OllamaEmbedder.is_available()`.
2. **`get_tool_schemas()` returns OpenAI function-calling format**, not Anthropic tool_use format. The agent loop converts.
3. **`sync_turn()` must be non-blocking.** The writer queue absorbs latency.
4. **`handle_tool_call()` returns a JSON string**, never a dict.
5. **Non-primary `agent_context` (cron, subagent, flush) skips writes** but reads are always allowed.
6. **Only one external memory provider at a time** — MemoryManager rejects the second one with a warning.

### From Phase 2A design choices

7. **Profile isolation via `profile` column on every row.** Every query filters on it. Default is `"default"`.
8. **Soft deletes only.** Every read path filters `WHERE deleted_at IS NULL`. Indexes use partial-index `WHERE deleted_at IS NULL` to skip them.
9. **Provenance is mandatory.** Every memory has `source_type` + `source_ref`. Every commitment does too.
10. **All writes go through `WriterQueue`.** Reads go through `ReaderPool`. The provider never holds a direct `sqlite3.Connection`.
11. **Embedding generation happens on the caller's thread**, not the writer thread. Ollama HTTP latency never stalls the queue.
12. **`check_same_thread=False` is quarantined to `ReaderPool` only.** Every other sqlite3.Connection uses the safe default.
13. **`""` is NOT a wildcard** for `search_entities`. Use explicit `"*"` for "all entities" — the empty-string-returns-all behavior is a footgun.
14. **FTS5 default boolean is AND, not OR.** `build_fts_query` must insert explicit `OR` operators between tokens.
15. **bm25 returns negative numbers; smaller = better.** Normalization must min-max invert within the candidate set, not globally.
16. **`pack_embedding`/`unpack_embedding` use `struct` little-endian float32.** This format is compatible with `sqlite-vec` virtual columns for future vec0 migration.

### From the Claudia persona

17. **Trust is the north star.** Every memory has traceable provenance. Never present inferences as stated facts. Surface contradictions rather than silently picking.
18. **Reads are stateless.** `get_entity`, `find_entity`, etc. do NOT bump `access_count` or `updated_at`. Rehearsal tracking goes through a separate explicit `bump_access` call on the writer queue.

---

## Phase 2B plan (forward-looking)

Phase 2B reproduces the "intelligent" parts of Claudia v1 — the things that require LLM calls, not just SQL. Phase 2A gave us the wire; 2B gives us the voltage.

### Sub-tasks (tentative ordering)

| # | Sub-task | Scope | v1 test files ready |
|---|---|---|---|
| 2B.1 | LLM entity extraction | Replace v1's spaCy+regex `entity_extractor` with LLM prompting. Extract entities from message content, return `ExtractedEntity` list with confidence. Wire into `provider.sync_turn`. | `test_extractor_stopwords.py`, `test_entity_type_inference.py`, `test_entity_paths.py` |
| 2B.2 | Commitment detection | Pattern + LLM hybrid to surface "I'll send the proposal by Friday" → commitment row. Track in `commitments` table (schema already has it). Wire into `provider.sync_turn`. | (new — v1 used pattern-only) |
| 2B.3 | Auto-consolidation | Fuzzy dedup via `SequenceMatcher` + embedding similarity + auto-merge on confidence > threshold. Background scheduled job. | `test_alias_filter.py`, `test_alias_specificity.py`, `test_dedup_fuzzy.py`, `test_fuzzy_entity.py`, `test_consolidation.py`, `test_consolidation_v2.py`, `test_merge.py` |
| 2B.4 | Verification service | Memory confidence decay over time + background verification of stale facts. Uses `verification` column (already in schema). | `test_verify.py`, `test_invalidated_filter.py` |
| 2B.5 | Cost governance | Token budget enforcement via `on_turn_start(remaining_tokens)` hook. The ABC passes `remaining_tokens` in kwargs; the provider reads but cannot block. Enforcement has to happen in `run_agent.py` itself. | (new — v1 didn't have this) |
| 2B.6 | Prompt budget accounting | Measure + bound the `~2000 tokens` memory contribution to the prompt. May require changes to `system_prompt_block` and `prefetch` output format. | (new) |

### v1 test files waiting for 2B (from 2A.4 audit)

These were deferred. When each Phase 2B sub-task lands, cherry-pick the matching v1 tests:

**For 2B.1 (LLM extraction):**
- `test_cognitive.py` — cognitive ingest service
- `test_extractor_stopwords.py` — spaCy stopword filter (may port as LLM prompt guidance)
- `test_entity_type_inference.py` — kind detection
- `test_entity_paths.py` — extraction from nested content

**For 2B.2 (commitments):**
- (v1 had commitment detection scattered — not in a single file; design fresh)

**For 2B.3 (consolidation + dedup):**
- `test_alias_filter.py` — single-token alias filter (Joel Salinas vs Joel Hart)
- `test_alias_specificity.py` — multi-token alias dedup
- `test_consolidation.py` — core consolidation logic
- `test_consolidation_v2.py` — LLM-based consolidation
- `test_dedup_fuzzy.py` — fuzzy matching thresholds
- `test_fuzzy_entity.py` — SequenceMatcher on write
- `test_merge.py` — entity merge with memory and relationship transfer
- `test_llm_consolidation.py` — LLM consolidation specifically
- `test_llm_coercion.py` — LLM output coercion

**For 2B.4 (verification):**
- `test_verify.py` — verification workflow
- `test_invalidated_filter.py` — invalidated memory filtering
- `test_temporal.py` — temporal queries
- `test_temporal_recall.py` — time-based recall

**For 2B.5 (cost governance):**
- (no v1 equivalent)

**For 2B.6 (prompt budget):**
- (no v1 equivalent — v1 had a different injection model)

### Scope not addressed in 2B (later phases or never)

- **Obsidian vault sync** (`test_vault_operations.py`) — v1 projected SQLite into a PARA-structured markdown vault. Plugin model doesn't have this.
- **Canvas generation** (`test_canvas_*.py`) — v1 generated Obsidian canvas files. Skip.
- **Briefing service** (`test_briefing*.py`) — `/morning-brief` skill lives outside the plugin.
- **TUI data source** (`test_tui_data_source.py`) — brain-monitor is a separate binary.
- **Daemon lifecycle** (`test_daemon_lifecycle.py`) — the plugin IS the daemon equivalent.
- **HTTP health endpoint** (`test_health_endpoint.py`) — the plugin exposes health via `system_prompt_block`, not HTTP.
- **Scheduler** (`test_scheduler_misfire.py`) — Hermes cron is the scheduler; plugin hooks into it later if needed.
- **Predictions** (`test_prediction*.py`) — removed per v1 robustness refactor; not porting.
- **Metrics** (`test_metrics.py`) — separate Phase 2C concern.
- **vec0 extension** (`test_vec0_loading.py`) — we use pure-Python cosine; vec0 is an optimization for later.

---

## Working patterns established in Phase 2A

These patterns are load-bearing for how the plugin grew. Keep them in 2B.

### Per-sub-task commit discipline

Each sub-task gets its own commit in both repos with a matching message shape:

- **Submodule message**: `Phase 2X.Y: <what>` with a detailed body listing new files, public API, design notes, what's NOT included, and next sub-task.
- **Outer message**: `autonomous: Phase 2X.Y - <what> (rollback point)` with a short summary + submodule bump + next sub-task.

The detailed submodule message is the primary documentation — read it later to understand "why did this sub-task make THIS choice?"

### Test-first (mostly)

Phase 2A.2b-2A.2e followed TDD: write the module, write exhaustive unit tests in the same commit, verify green before committing. 2A.2a deferred tests and that debt was repaid in 2A.4 — don't do that again.

### Test file conventions

- Tests go in `tests/plugins/memory/claudia/test_<module>.py` for unit tests
- Integration tests against `MemoryManager` go in `tests/agent/test_<name>_e2e.py`
- Pytest organization: top-level classes group by concern (`TestFoo`, `TestBar`), methods test one thing
- Use pinned `NOW = datetime(...)` anchor for time-dependent tests so they're deterministic
- Real SQLite via `schema.ensure_database(tmp_path / "x.db")` — no mocks for DB
- Fake embedder via `_FakeEmbedder(OllamaEmbedder)` subclass with scripted `_call_embed` — offline, deterministic
- Run tests with: `python3 -m pytest tests/plugins/memory/claudia/ -q -p no:xdist -o addopts="-m 'not integration and not frontier and not local'"`

The `-p no:xdist -o addopts=...` override is needed because the fork's `pyproject.toml` has `-n auto` in `addopts` for parallel execution but `pytest-xdist` isn't installed.

### Factory methods for testability

`ClaudiaMemoryProvider._make_embedder()`, `_make_router()`, `_make_writer()`, `_make_reader_pool()` are overridable hooks. Tests subclass the provider and override these to inject fakes. Don't change this pattern — it's what makes the test suite fast and deterministic.

### Design invariants in test form

Several tests are tripwires locking in specific design decisions:

- `test_fts_respects_multiword_and_operator` in `test_schema.py` — FTS5 default-AND semantics
- `test_empty_string_stays_empty_even_with_other_entities` in `test_entities.py` — wildcard vs. empty
- `test_no_substring_false_positive` in `test_entities.py` — alias exact-match, not substring
- `test_shutdown_drains_pending_writes` in `test_provider.py` — graceful writer drain

When you write a test like this, add a comment explaining what it's preventing. Future you will thank you.

---

## Pitfalls I hit during Phase 2A (future-me heads up)

1. **`pytest -n auto`** — the repo's pyproject enables xdist but it's not installed. Always pass `-p no:xdist -o addopts=...` when running tests locally.

2. **`_conn` direct access in tests** — when I migrated to writer/pool in 2A.3, ~11 tests broke because they read `provider._conn.execute(...)`. The fix was helpers `_count_memories(provider)` and `_read(provider, sql, params)` that call `provider.flush()` then acquire a reader. Use these helpers from the start in 2B tests.

3. **`sync_turn` is now async** — if you add a test that writes via `sync_turn` and then reads, ALWAYS call `provider.flush()` before the read or you'll get flaky empty results.

4. **Entity creation in tests** — if you need to create entities for a test, enqueue the write via `provider._writer.enqueue_and_wait(_create, timeout=5.0)` instead of direct `entities.create_entity(conn, ...)`. That matches the real write path and flushes automatically.

5. **`check_same_thread` on schema migrations** — when you call `schema.ensure_database(path)` it returns a conn that's bound to the calling thread. If you plan to share that conn with other threads (like passing it to a fixture that runs under xdist), use `open_connection(path, check_same_thread=False)` instead.

6. **JSON column ordering** — `aliases_json` and `attributes_json` are stored as canonical JSON via `json.dumps(..., ensure_ascii=False)`. If you compare these columns byte-for-byte in a test, normalize both sides.

7. **Submodule + outer commit pattern** — when committing a phase, ALWAYS stage + commit in the submodule first, then go to the outer repo and run `git add autonomous/fork` to pick up the submodule bump, then commit in the outer. Reverse order creates a dirty-submodule state that's annoying to clean up.

8. **Test file for provider has 56 tests** — don't keep adding to it without discipline. If you're adding >10 tests for a new concept in 2B, consider a new `test_provider_<concept>.py` file instead.

9. **The `_TestProvider` subclass pattern** — both `test_provider.py` and `test_claudia_memory_plugin_e2e.py` define their own `_TestProvider`/`_TestClaudiaProvider` subclass with `_make_embedder` overridden. This duplication is OK for now but if 2B adds more test files that need the same pattern, move it to a `tests/plugins/memory/claudia/_provider_fixtures.py` shared helper.

---

## How to resume after `/clear`

**Short version**: tell the fresh session to read this doc:

> Read `autonomous/fork/plans/phase-2b-handoff.md` and continue the plan. Phase 2A is complete. Start Phase 2B.1 (LLM entity extraction).

**Verbose version** if you want more guidance:

> I'm continuing the Claudia memory provider plugin work. Phase 2A is complete (10 rollback points, 305 tests passing). Read the handoff doc at `autonomous/fork/plans/phase-2b-handoff.md` first to get up to speed on what's done, the design invariants, the file layout, and the pitfalls. Then start Phase 2B.1 (LLM-based entity extraction): replace v1's spaCy+regex `entity_extractor` with an LLM-prompted extractor that plugs into `ClaudiaMemoryProvider.sync_turn`. Use the v1 tests in `claudia/memory-daemon/tests/test_extractor_*.py` as a reference but adapt them to the plugin API. Follow the per-sub-task commit pattern (submodule first, then outer repo rollback point).

### Session-start checklist for the fresh session

Before writing any code:

- [ ] `cd "/Users/kamil/Documents/Claude Code/Open Source Claudia Agent"` and verify you're on branch `claude/setup-project-structure-bvdSj`
- [ ] `git log --oneline -3` and confirm the last commit is `a5037de autonomous: Phase 2A.5 - e2e via real MemoryManager (Phase 2A COMPLETE)`
- [ ] `git -C autonomous/fork log --oneline -3` and confirm submodule HEAD is `b0adcb9 Phase 2A.5: ...`
- [ ] `cd autonomous/fork && python3 -m pytest tests/plugins/memory/claudia/ tests/agent/test_claudia_memory_plugin_e2e.py -q -p no:xdist -o addopts="-m 'not integration and not frontier and not local'"` and confirm 305 passed
- [ ] Read `docs/decisions/memory-provider-design.md` for the Phase 2A.1 design context (only ~180 lines)
- [ ] Read the "Critical design invariants" section of this handoff
- [ ] Read the submodule commit message of the previous sub-task (`git log b0adcb9 --format="%B" -1`) for the immediate handoff context
- [ ] Read `agent/memory_provider.py` to re-familiarize with the ABC
- [ ] Check the v1 reference files for the sub-task you're about to start

Only THEN write code.

---

## One last thing

The main value of this handoff isn't the summary — it's the **invariants list**. Phase 2A took 10 commits to shake those out; a fresh session that ignores them will introduce subtle regressions that the test suite won't catch until much later. Re-read "Critical design invariants" whenever you're making a non-obvious design choice in Phase 2B. If something you're about to write contradicts an invariant, STOP and either justify the deviation in a commit message or pick a different approach.

Good luck. Adapt and create.
