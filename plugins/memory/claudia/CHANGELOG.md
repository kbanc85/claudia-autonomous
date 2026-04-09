# Claudia Hybrid Memory — Changelog

All notable changes to the plugin will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [1.0.0] — 2026-04-09

First user-facing release. Feature-complete for a trust-aware
local memory provider. 902 tests, ~10s runtime on Python 3.13.

### The memory store

- SQLite-backed memories, entities, relationships, commitments
- FTS5 full-text search with porter/unicode61 tokenizer
- Hybrid ranking: 50% vector + 25% importance + 10% recency + 15% FTS + rehearsal boost
- Three-tier offline degradation: full hybrid → FTS + importance + recency → pure FTS
- WAL mode with a single serialized writer and a bounded reader pool for concurrent reads
- Profile isolation on every table and every query
- Soft deletes everywhere; retention purge for permanent cleanup after a configurable window

### Cognitive pipeline

- **LLM entity extraction** via local Ollama (default `qwen2.5:3b`), running on a dedicated single-worker executor so `sync_turn` stays non-blocking
- **Hybrid commitment detection** — regex pre-filter for a fast offline path, LLM refinement when Ollama is available. Drops vague intentions ("someday", "maybe"), preserves explicit and implicit commitments, parses natural-language deadlines ("by Friday", "next week", "in 2 hours") to ISO 8601
- **Fuzzy entity consolidation** via `difflib.SequenceMatcher` with a last-token safety gate for people (prevents "Joel Salinas" from merging with "Joel Hart"). Relationships and commitments transfer atomically on merge
- **Confidence decay** with absolute targets so daily `verify()` runs are idempotent. `user_stated` and `corrected` origins are protected from decay
- **Stale flagging** for memories that stay `pending` past a configurable threshold (default 60 days)

### Trust model

- Every memory has `origin` (user_stated / extracted / inferred / corrected), `confidence` (0.0–1.0), `verification` (pending / verified / flagged / contradicts), and `source_type` + `source_ref` for provenance
- Final recall score = `(base + boost) × confidence × verification_multiplier`
- Verification multipliers: verified 1.0, pending 1.0, flagged 0.5, contradicts 0.3
- `memory.correct_memory` creates a new `origin='corrected'` row linked via `corrected_from` and demotes the old row to `contradicts`. `memory.trace` walks the full chain for audits

### Lifecycle hooks

Full `MemoryProvider` ABC implementation:

- `initialize`, `shutdown`, `name`, `is_available`, `get_tool_schemas`, `handle_tool_call`
- `system_prompt_block` with live counts and mode
- `prefetch` with hybrid search + token budget truncation
- `sync_turn` enqueues memory insert + entity extraction + commitment detection
- `on_turn_start` reads `remaining_tokens` for cost governance
- `on_session_end` soft-flushes pending cognitive work
- `on_pre_compress` returns recent memories for the compressor
- `on_memory_write` mirrors built-in MEMORY.md / USER.md changes
- `on_delegation` captures subagent observations as `origin='inferred'`
- `get_config_schema` + `save_config` for `claudia memory setup`

### Tool surface (18 tools)

**Read**: `memory.recall`, `memory.about`, `memory.search_entities`, `memory.commitments`, `memory.recent`, `memory.trace`, `memory.status`, `memory.metrics`

**Write**: `memory.remember`, `memory.correct_memory`, `memory.relate`

**State mutation**: `memory.commitment_complete`, `memory.commitment_drop`, `memory.verify_memory`, `memory.flag_memory`, `memory.contradicts_memory`

**Delete**: `memory.forget_memory`, `memory.forget_entity`

### Cost governance

- `on_turn_start(remaining_tokens=...)` updates the budget state
- Critical budget (< 2000 tokens): skip extraction, detection, minimize prefetch
- Low budget (< 5000 tokens): reduce prefetch limit, keep cognitive work
- Normal budget: full 10-result prefetch, full cognitive work
- `prefetch` output is truncated to `prefetch_budget_tokens` via line-oriented truncation with a visible `(truncated)` marker
- Memory inserts are NEVER skipped; only best-effort cognitive work degrades

### Configuration (8 fields)

- `ollama_host` — Ollama URL, env override `CLAUDIA_OLLAMA_HOST`
- `embedding_model` — default `all-minilm:l6-v2`
- `extraction_model` — default `qwen2.5:3b`
- `detection_model` — default `qwen2.5:3b`
- `decay_half_life_days` — default 30
- `stale_flag_days` — default 60
- `auto_merge_threshold` — default 0.92
- `retention_days` — default 90

Persisted to `$CLAUDIA_HOME/memory/claudia/config.json` via `save_config`. Factory methods and maintenance calls honor the loaded values.

### Observability

- `memory.metrics` tool and `provider.get_metrics()` accessor
- 14 counter categories: sync_turn calls/skipped, memories inserted, entities upserted, commitments inserted, cognitive extractions/detections runs and errors, consolidate/verify/purge runs, per-tool dispatch counts, total tool errors
- Thread-safe counter registry in pure stdlib (no prometheus dependency)

### Maintenance API

- `provider.consolidate(timeout=30)` — fuzzy entity dedup + commitment FK resolution. Idempotent.
- `provider.verify(timeout=30)` — confidence decay + stale flagging. Idempotent.
- `provider.purge_old_soft_deletes(timeout=30, retention_days=None)` — permanent removal past the retention window. Uses real DELETE statements (unlike the soft-delete write paths).

All three flush pending cognitive work before running and commit inside a single writer-queue job.

### Integration with claudia-autonomous

- `claudia memory setup` auto-discovers the plugin via `plugins/memory/__init__.py`
- `claudia doctor` reports plugin discovery, DB state, config presence, Ollama reachability, and required model availability (Claudia Memory section)
- Runs alongside the built-in MEMORY.md / USER.md provider — both can be active simultaneously with no interference

### Documentation

- `README.md` — end-user setup, config, tool reference, troubleshooting
- `CHANGELOG.md` — this file
- `demo.py` — runnable end-to-end walkthrough, offline via scripted fakes
- `plan/phase-2c-handoff.md` (in the fork root) — full design + invariants + rollback chain for future contributors

### Test coverage

- 902 tests (unit + integration + MemoryManager e2e)
- ~10 seconds runtime on Python 3.13
- 18 rollback points across Phase 2B (6 sub-tasks) and Phase 2C (18 sub-tasks)
- Tripwire tests lock key design decisions: last-token safety for people, user_stated-not-decayed, correction audit trail, non-blocking sync_turn, verification multipliers, empty-string-is-not-wildcard, prompt guardrail invariants

### Known gaps

These were explicitly deferred or are out of scope for v1.0:

- **v1 claudia-memory daemon migration tool** — needs a real v1 SQLite file to test against; deferred pending a user requesting it
- **Automated scheduler integration** — the maintenance API is opt-in via `provider.consolidate()` etc. Hooking into Hermes cron is a separate integration task on the host side
- **Real LLM prompt quality evaluation** — tripwire tests lock the prompt structure but the behavior quality (precision/recall of extraction and detection) requires an evaluation harness that's outside the plugin's scope
- **Load / throughput benchmarks** — correctness is tested; production throughput is not measured here
- **Vault sync, canvas generation, briefing service, TUI data source, daemon lifecycle, HTTP health endpoint, standalone scheduler, predictions, vec0 extension** — all explicitly out of scope (see the `plans/phase-2c-handoff.md` permanently deferred list)
