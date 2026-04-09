# Phase 2B → Phase 2C Handoff

**Created**: 2026-04-09 by the Phase 2B-2C implementation session
**Purpose**: Pick up cleanly after `/clear` with zero context loss
**Companion doc**: `phase-2b-handoff.md` (Phase 2A → 2B boundary)

---

## TL;DR

Phase 2B is **complete** (all six sub-tasks shipped). Phase 2C has landed 12 polish sub-tasks on top of 2B, covering trust-aware ranking, built-in memory sync, configurability, retention hygiene, the full LLM tool surface for state mutation, and lifecycle hooks. **Everything is green, 840 tests, ~10s runtime on Python 3.13.**

The plugin is now feature-complete for a trust-aware local memory provider:
- Memories are extracted (2B.1), mutated with user corrections (2C.9), and ranked by trust (2C.1)
- Commitments are detected (2B.2), listed (2C.5), and mutated by state transitions (2C.7)
- Duplicates are merged (2B.3), pending memories decay (2B.4), stale rows are purged (2C.4)
- Cost governance shrinks output under budget pressure (2B.5 / 2B.6)
- Built-in MEMORY.md writes are mirrored (2C.2)
- 14 LLM-facing tools cover read, write, mutation, graph, and provenance operations

---

## Current state

| Thing | Value |
|---|---|
| Outer repo branch | `claude/setup-project-structure-bvdSj` |
| Outer repo HEAD | `4757320 autonomous: Phase 2C.12 - memory.trace tool (rollback point)` |
| Submodule branch | `main` |
| Submodule HEAD | `5441b3b Phase 2C.12: memory.trace tool` |
| Working tree | Clean for the 2B/2C work |
| Plugin suite | **840 tests passing** |
| Test runtime | ~10 seconds on Python 3.13 |

**Verify on session resume:**

```bash
cd "/Users/kamil/Documents/Claude Code/Open Source Claudia Agent/autonomous/fork"
git log --oneline -1   # should show 5441b3b
python3 -m pytest tests/plugins/memory/claudia/ tests/agent/test_claudia_memory_plugin_e2e.py tests/agent/test_memory_plugin_e2e.py tests/agent/test_memory_provider.py -q -p no:xdist -o addopts="-m 'not integration and not frontier and not local'"
# Expect: 840 passed
```

If the count differs, STOP and diagnose. The rollback chain below has a clean state at every sub-task.

---

## Full rollback chain (Phase 2B + 2C)

Every entry is a clean ship state. Any commit can be the target of `git reset --hard` in both repos.

### Phase 2B (sub-module HEAD / outer HEAD)

| # | Sub-task | Submodule | Outer | What landed |
|---|---|---|---|---|
| 2A.5 | e2e baseline | `b0adcb9` | `a5037de` | Phase 2A COMPLETE |
| 2B.1 | LLM entity extraction | `a6847e1` | `dd20985` | `extractor.py` + dedicated cognitive executor |
| 2B.2 | commitment detection | `10ee027` | `ddbd3b7` | `commitments.py` CRUD + `commitment_detector.py` |
| 2B.3 | auto-consolidation | `27acae2` | `33c8612` | `consolidation.py` with `provider.consolidate()` |
| 2B.4 | verification service | `6f8a165` | `3b3b4ea` | `verification.py` with `provider.verify()` |
| 2B.5 | cost governance | `c402427` | `d67e01d` | `budget.py` reading `remaining_tokens` |
| 2B.6 | prompt budget accounting | `1f3edff` | `a660405` | `estimate_tokens` + `truncate_to_budget` |

### Phase 2C (polish)

| # | Sub-task | Submodule | Outer | What landed |
|---|---|---|---|---|
| 2C.1 | verification weighting | `6ffcdaf` | `563b03e` | trust_factor in hybrid_search |
| 2C.2 | on_memory_write mirror | `adcdd02` | `b1fa65e` | built-in MEMORY.md sync |
| 2C.3 | config schema + persist | `680bcd2` | `2f94485` | `get_config_schema` / `save_config` / `_load_config` |
| 2C.4 | retention purge | `f4a81b8` | `cd7ef5d` | `retention.py` + `provider.purge_old_soft_deletes` |
| 2C.5 | commitments + status tools | `d64bedc` | `35777ed` | `memory.commitments`, `memory.status` |
| 2C.6 | lifecycle hooks | `da30ebb` | `fdba39d` | `on_session_end`, `on_pre_compress` |
| 2C.7 | commitment mutation | `d54537f` | `d199a22` | `memory.commitment_complete`, `memory.commitment_drop` |
| 2C.8 | memory verification mutation | `ba1c9c1` | `81d2793` | `memory.verify_memory`, `memory.flag_memory`, `memory.contradicts_memory` |
| 2C.9 | memory.correct_memory | `c654145` | `728c8da` | atomic correction with `corrected_from` link |
| 2C.10 | memory.relate | `16ab487` | `cd5f2a6` | explicit graph edges |
| 2C.11 | memory.search_entities | `f4172dd` | `8f2dab9` | fuzzy substring search |
| 2C.12 | memory.trace | `5441b3b` | `4757320` | correction chain walker |

---

## File map (Phase 2A + 2B + 2C)

### Implementation (`autonomous/fork/plugins/memory/claudia/`)

| File | Role | Approx lines | Phase |
|---|---|---|---|
| `schema.py` | SQLite DDL, migrations, WAL, FTS5 | ~330 | 2A.2a |
| `embeddings.py` | Ollama embedder + BLOB codec | ~330 | 2A.2b |
| `hybrid_search.py` | Ranking + verification weighting | ~720 | 2A.2c + 2C.1 |
| `entities.py` | Entity CRUD + relationships | ~840 | 2A.2d |
| `offline.py` | Three-tier degradation router | ~240 | 2A.2e |
| `writer.py` | Serialized writer queue | ~270 | 2A.3 |
| `reader.py` | Bounded reader pool | ~180 | 2A.3 |
| `provider.py` | MemoryProvider ABC + 14 tools | ~1900 | 2A.2f + 2B + 2C |
| `extractor.py` | LLM entity extraction | ~430 | 2B.1 |
| `commitments.py` | Commitment CRUD | ~340 | 2B.2 |
| `commitment_detector.py` | Hybrid pattern+LLM detector | ~620 | 2B.2 |
| `consolidation.py` | Fuzzy dedup + FK resolution | ~560 | 2B.3 |
| `verification.py` | Decay + stale flagging + marks | ~380 | 2B.4 |
| `budget.py` | Token budget + truncation | ~300 | 2B.5 + 2B.6 |
| `retention.py` | Purge old soft-deletes | ~180 | 2C.4 |
| `__init__.py` | Exports + `register(ctx)` | ~50 | 2A |
| `plugin.yaml` | Metadata | ~15 | 2A |

### Tests (`autonomous/fork/tests/plugins/memory/claudia/`)

Unit tests by module:

| File | Tests | Covers |
|---|---|---|
| `test_schema.py` | 17 | Schema DDL + migrations |
| `test_embeddings.py` | 31 | Ollama + BLOB codec |
| `test_hybrid_search.py` | 46 | Ranking + verification weighting (2C.1) |
| `test_entities.py` | 72 | Entity CRUD |
| `test_offline.py` | 24 | Three-tier degradation |
| `test_provider.py` | 56 | Provider ABC surface |
| `test_writer.py` | 24 | Writer queue |
| `test_reader.py` | 20 | Reader pool |
| `test_concurrency.py` | 6 | Concurrent access |
| `test_extractor.py` | 64 | LLM extraction |
| `test_commitments.py` | 42 | Commitment CRUD |
| `test_commitment_detector.py` | 66 | Commitment detection |
| `test_consolidation.py` | 43 | Fuzzy dedup + merge |
| `test_verification.py` | 26 | Decay + stale + marks |
| `test_budget.py` | 27 | Token budget + truncation |
| `test_retention.py` | 20 | Purge |

Provider integration tests:

| File | Tests | Covers |
|---|---|---|
| `test_provider_extraction.py` | 15 | sync_turn → entity extraction |
| `test_provider_commitments.py` | 18 | sync_turn → commitment detection |
| `test_provider_consolidation.py` | 11 | `provider.consolidate()` |
| `test_provider_verification.py` | 7 | `provider.verify()` |
| `test_provider_budget.py` | 13 | `on_turn_start` budget handling |
| `test_provider_retention.py` | 6 | `provider.purge_old_soft_deletes()` |
| `test_provider_on_memory_write.py` | 14 | built-in memory mirror |
| `test_provider_config.py` | 14 | config schema + load |
| `test_provider_new_tools.py` | 16 | `memory.commitments`, `memory.status` |
| `test_provider_lifecycle_hooks.py` | 9 | `on_session_end`, `on_pre_compress` |
| `test_provider_commitment_tools.py` | 12 | commitment mutation tools |
| `test_provider_verification_tools.py` | 10 | memory verification tools |
| `test_provider_correct_memory.py` | 11 | `memory.correct_memory` |
| `test_provider_relate.py` | 11 | `memory.relate` |
| `test_provider_search_entities.py` | 12 | `memory.search_entities` |
| `test_provider_trace.py` | 11 | `memory.trace` |

MemoryManager integration:

| File | Tests | Covers |
|---|---|---|
| `tests/agent/test_claudia_memory_plugin_e2e.py` | 18 | Through real MemoryManager |
| `tests/agent/test_memory_plugin_e2e.py` | 14 | Toy provider (adjacent) |
| `tests/agent/test_memory_provider.py` | 34 | ABC base |

**Total: 840 tests.** Runtime ~10s on Python 3.13.

---

## Tool surface (14 tools)

All tools use OpenAI function-calling format (ABC contract) and are exposed via `provider.get_tool_schemas()`.

### Read tools (5)

| Tool | What it does |
|---|---|
| `memory.recall` | Hybrid search with query + optional limit |
| `memory.about` | Exact entity lookup by name (case-insensitive) |
| `memory.search_entities` | Fuzzy substring search over entities + `*` wildcard |
| `memory.commitments` | List commitments by status |
| `memory.trace` | Walk a memory's correction chain (timeline) |
| `memory.status` | Diagnostic counts + mode + session_id |

### Write tools (3)

| Tool | What it does |
|---|---|
| `memory.remember` | Store a single fact with origin=user_stated |
| `memory.correct_memory` | Replace a memory with a corrected version; old row → contradicts |
| `memory.relate` | Create a directed relationship between two existing entities |

### State mutation tools (5)

| Tool | What it does |
|---|---|
| `memory.commitment_complete` | Mark commitment as completed (sets completed_at) |
| `memory.commitment_drop` | Mark commitment as dropped (user abandoned) |
| `memory.verify_memory` | Mark memory as explicitly verified (full trust weight) |
| `memory.flag_memory` | Mark memory as flagged (0.5× recall weight) |
| `memory.contradicts_memory` | Mark memory as contradicting another (0.3× weight) |

**Total: 14 tools.**

---

## Critical design invariants (Phase 2B + 2C additions)

These are in addition to the 18 invariants from the Phase 2B handoff. Honor them in any future work on the plugin.

### From Phase 2B

19. **Cognitive executor is shared.** Entity extraction (2B.1) and commitment detection (2B.2) submit to the same single-worker `_cognitive_pool`. Future cognitive tasks should join the pool rather than spawning new ones. Drains happen together via `_drain_cognitive_futures`.

20. **Extraction and detection NEVER run on non-primary agent_context.** The filter lives in `sync_turn` BEFORE the enqueue, not inside the worker. Cron/subagent/flush contexts still insert memories (source of truth) but skip all cognitive work.

21. **Commitment detection runs on user_content ONLY.** Not the combined turn. Commitments come from the user; Claudia's responses go through approval before becoming commitments.

22. **Cost governance is read-only on the provider side.** `on_turn_start` populates `self._budget_state`; every other hook calls `decide_budget()` to consult it. The provider never blocks a turn — real enforcement is in `run_agent.py`.

23. **Idempotent decay targets.** Confidence decay in 2B.4 uses ABSOLUTE targets (from `DEFAULT_INITIAL_CONFIDENCE * 2^(-elapsed/half_life)`), NOT relative decay from current value. This keeps daily cron runs idempotent. Don't change to relative decay.

24. **user_stated and corrected memories are protected from decay.** Only `extracted` and `inferred` origins decay. Corrected rows (origin=corrected) are the gold standard — explicit user assertions with confidence 1.0.

25. **Prompt budget truncation is line-oriented, drops from bottom, includes a marker.** `truncate_to_budget` never leaves a dangling mid-bullet cut. The marker (`"... (truncated)"`) is visible to the LLM so it knows content was dropped.

### From Phase 2C

26. **Trust factor = confidence × verification_multiplier.** Applied to `(base + boost)` in `hybrid_search.search`. Don't multiply component scores separately — the trust factor is a single multiplier on the final composite. Tripwire: `test_verified_high_relevance_outranks_low_relevance`.

27. **Joel Salinas / Joel Hart safety gate is permanent.** For kind=person entities with multi-word names, the last token must match before SequenceMatcher's score is trusted. Don't weaken this. Single-word people never fuzzy-merge — only exact matches. Tripwire: `test_joel_salinas_vs_joel_hart_not_merged`.

28. **Profile isolation in cross-profile tool calls returns "unknown id" errors.** Never reveal that an id exists in a different profile. This is a privacy guarantee for multi-user gateway sessions. Tripwires exist in every tool's test file.

29. **Config is read at initialize() BEFORE factories.** `_load_config()` populates `self._config` first so `_make_extractor`, `_make_commitment_detector`, etc. can read values from it. Don't reorder this in `initialize()`.

30. **Retention purge is destructive and explicit.** It's a SEPARATE method from `consolidate()`. Don't bundle it into consolidation. Users opt into permanent deletion deliberately.

31. **Correction chains are symmetric.** `memory.trace` walks both backward (via `corrected_from`) and forward (via reverse lookup). Calling trace from any version in the chain returns the same full chain ordered oldest-first.

32. **memory.correct_memory is atomic.** The new row insert AND the old row contradicts-marking happen in one writer job. A concurrent reader never sees "both live" or "both dead" states.

33. **VERIFICATION_WEIGHTS dict is tripwire-locked.** 1.0 / 1.0 / 0.5 / 0.3 for verified/pending/flagged/contradicts. Don't drift.

34. **Tools return structured JSON errors, not exceptions.** Every handler validates args and returns `{"error": "..."}` on failure. Missing id, wrong type, unknown id, cross-profile — all return clean JSON. Never let a raw exception escape `handle_tool_call` to the caller.

---

## Key tripwire tests (must not break)

These tests lock specific design decisions and will catch regressions early:

### Phase 2A / 2B tripwires (from previous handoff)

- `test_valid_entity_kinds_matches_entities_module` — extractor kinds locked to schema
- `test_joel_salinas_vs_joel_hart_not_merged` — consolidation last-token safety
- `test_user_stated_memory_not_decayed` — verification respects human assertions
- `test_sync_turn_returns_before_extraction_completes` — non-blocking contract
- `TestPromptInvariants` (extractor and detector) — prompt guardrails
- `test_valid_statuses_matches_schema` (commitments and verification) — enum drift prevention

### Phase 2C tripwires

- `test_verification_weights_constants` — VERIFICATION_WEIGHTS dict values locked
- `test_verified_high_relevance_outranks_low_relevance` — relevance still matters even with trust weighting
- `test_profile_isolation` (in every tool's test file) — cross-profile access blocked
- `test_old_memory_content_unchanged` (correct_memory) — audit trail tripwire
- `test_chain_ordered_oldest_first` (trace) — timeline order
- `test_remove_does_not_touch_non_builtin_memories` (on_memory_write) — source_type filter precision
- `test_fk_integrity_rejects_bogus_ids` (commitments) — schema FK enforcement
- `test_idempotent_close_enough` (verification decay) — absolute vs relative targets
- `test_non_deleted_never_touched` (retention purge) — safety
- `test_threshold_is_reasonable` (consolidation) — AUTO_MERGE_THRESHOLD bounds

---

## Working patterns established in 2B/2C

### Per-sub-task commit discipline (unchanged from 2A)

Each sub-task gets one commit in both repos:

- **Submodule message**: `Phase 2X.Y: <what>` with detailed body listing new files, public API, tests, design notes, invariants upheld, known gaps, rollback chain
- **Outer message**: `autonomous: Phase 2X.Y - <what> (rollback point)` with a short summary and the submodule bump

### Tool-adding pattern

Every new LLM-facing tool follows this template:

1. Define `MEMORY_X_SCHEMA` constant at module level
2. Add to `_ALL_TOOL_SCHEMAS` list
3. Add dispatch branch in `handle_tool_call`
4. Write handler `_handle_x` that:
   - Validates required args (returns error JSON on invalid)
   - Resolves profile from `self._profile`
   - Wraps the operation in a writer-queue job via `enqueue_and_wait` (for state mutations) or uses the reader pool (for reads)
   - Returns `{"ok": True, ...}` on success or `{"error": "..."}` on failure
5. Update `test_provider.py::TestBasicProperties::test_get_tool_schemas_returns_all_X` to match new total and name set
6. Write a dedicated test file `test_provider_<tool>.py`

### Test file split discipline

- Unit tests go in `test_<module>.py` (covers a single plugin module)
- Provider integration tests go in `test_provider_<concept>.py` (not `test_provider.py`, which stays focused on the ABC surface)
- Any new "concept" with >5 tests gets its own file

### Tool count assertion lives in ONE place

`test_provider.py::TestBasicProperties::test_get_tool_schemas_returns_all_N` is the single source of truth for total tool count. Individual tool test files do NOT assert the total — they do focused `test_X_tool_present` checks. This prevents the "every new tool breaks N-1 other test files" anti-pattern.

### Factory method overrides for testability

The `_make_embedder`, `_make_extractor`, `_make_commitment_detector` factory methods on `ClaudiaMemoryProvider` are overridable hooks. Tests inject `_NoOp*` subclasses to stay offline. The template for a test provider:

```python
class _TestProvider(ClaudiaMemoryProvider):
    def _make_embedder(self):
        return _FakeEmbedder()
    def _make_extractor(self):
        return _NoOpExtractor()
    def _make_commitment_detector(self):
        return _NoOpDetector()
```

Extracted into `tests/plugins/memory/claudia/_provider_fixtures.py` is a good idea if the number of test files needing this grows further.

---

## Pitfalls hit during 2B/2C (future-me heads up)

These are additions to the 9 pitfalls from the Phase 2B handoff:

10. **Tool count assertion propagates across test files.** Originally every tool test file asserted the total count. Adding one tool broke all of them. Fix: centralize the total assertion in `test_provider.py`, use focused "present" checks elsewhere.

11. **sqlite3.IntegrityError on relationship conflicts.** `memory.relate` hit this when the same (profile, from, to, type) was inserted twice. Handler must catch it and return a clean error JSON. Same pattern applies to commitment uniqueness.

12. **on_memory_write 'replace' must handle empty content.** The semantic is "replace the section's content with X" — if X is empty, that's a remove, not a no-op insert. Handler drops the insert but still does the soft-delete step.

13. **Config load BEFORE factory construction.** If `self._config = self._load_config()` runs after `self._extractor = self._make_extractor()`, the factory sees an empty config and uses defaults. Always load config first in `initialize()`.

14. **Absolute vs relative decay confusion.** Writing "compound decay each run" instead of "absolute target from elapsed time" would make verify() destroy confidence on every run. The tripwire test `test_idempotent_close_enough` locks the correct behavior.

15. **Correction chain traversal loops.** The traversal in `memory.trace` uses an explicit visited set + cap to avoid pathological cycles. Corrections SHOULD always be linear, but never trust that — defensive cap at 100.

16. **Tool dispatch + _ALL_TOOL_SCHEMAS drift.** It's easy to add a schema constant but forget the dispatch branch, or vice versa. The `TestDispatch::test_unknown_tool_returns_error` test is a weak check — explicit tool-exists tests in each tool's test file catch the missing-dispatch case.

17. **insert_memory test helper must accept new columns.** When I added `verification` to the insert helper in test_hybrid_search.py, existing tests started inheriting the new default. Make helpers explicit about defaults and update tests that care.

---

## Possible future work (no explicit plan)

Polish items that could become 2C.13+ or Phase 2D:

- **memory.forget_memory(id)** — soft-delete a specific memory (complements correct_memory)
- **memory.forget_entity(name)** — soft-delete an entity, cascading to its relationships
- **memory.recent(limit)** — list recently accessed memories (complements recall)
- **memory.briefing()** — summary output for "what's been going on lately?"
- **Scheduler integration** — hook verify/consolidate/purge into Hermes cron (requires coordination outside the plugin)
- **Config hot-reload** — detect config.json changes without a restart
- **Metrics/observability** — internal counters for extraction, detection, consolidation, trust decisions
- **Prompt caching** — cache the system_prompt_block output per session rather than re-building it

### Permanently deferred (handoff doc decisions, NOT future work)

Explicitly out of scope:
- Vault sync (PARA-structured markdown export)
- Canvas generation (Obsidian)
- Briefing service (/morning-brief skill lives outside the plugin)
- TUI data source (brain-monitor is a separate binary)
- Daemon lifecycle (plugin IS the daemon equivalent)
- HTTP health endpoint (exposed via `memory.status` instead)
- Standalone scheduler (Hermes cron owns scheduling)
- Predictions (removed per v1 robustness refactor)
- Metrics module (separate Phase 2C+ concern)
- vec0 extension (pure-Python cosine works; vec0 is an optimization for later)

---

## How to resume after `/clear`

**Short version**: tell the fresh session to read this doc:

> Read `autonomous/fork/plans/phase-2c-handoff.md` and continue the plan. Phase 2B and Phase 2C (through 2C.12) are complete. Plugin is feature-complete for trust-aware local memory.

**Session-start checklist** (before any new code):

- [ ] `cd "/Users/kamil/Documents/Claude Code/Open Source Claudia Agent"` and verify branch `claude/setup-project-structure-bvdSj`
- [ ] `git log --oneline -3` and confirm the latest rollback point (2C.12)
- [ ] `git -C autonomous/fork log --oneline -3` and confirm submodule HEAD `5441b3b`
- [ ] Run the full suite, confirm 840 passed
- [ ] Read the "Critical design invariants" section of THIS doc (not the 2B one)
- [ ] Check the tool surface section — 14 tools, all named above
- [ ] Only THEN write code

---

## One last thing

The main value here is the **invariants list and the tripwire tests**. Those encode the design decisions that took 18 commits across Phase 2B and Phase 2C to shake out. A fresh session that ignores them will introduce subtle regressions the suite won't catch until much later.

If you're about to make a non-obvious design choice, re-read the relevant invariant section. If what you're writing contradicts an invariant, either justify the deviation in a commit message or pick a different approach.

Good luck. Adapt and create.
