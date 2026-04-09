# Claudia Hybrid Memory

Local-first, trust-aware memory provider for the claudia-autonomous fork. SQLite + vector embeddings via Ollama. Extracts entities and commitments from conversations, merges duplicates, decays unverified facts, and lets the user verify, flag, contradict, or correct any memory via LLM tool calls. 100% local — no network beyond `localhost:11434`.

## Requirements

- **Ollama** running on `http://localhost:11434` (recommended)
  - Install: [ollama.com/download](https://ollama.com/download)
  - Pull the default models:
    ```bash
    ollama pull all-minilm:l6-v2    # embeddings (~23 MB)
    ollama pull qwen2.5:3b          # extraction + detection (~2 GB)
    ```
- Python 3.11+ (stdlib `sqlite3`) — already required by the fork
- That's it. No API keys, no cloud accounts, no external services.

**Ollama is optional at runtime.** If it's unreachable, the plugin runs in a degraded mode: hybrid search drops the vector component, entity extraction and commitment detection are skipped, and recall still works via FTS + importance + recency ranking. You get offline-safe memory without any configuration changes.

## Setup

```bash
claudia memory setup    # interactive picker — select "claudia"
```

Or manually:
```bash
claudia config set memory.provider claudia
```

No `.env` changes needed. Claudia has no API keys.

To verify the setup:
```bash
claudia doctor    # includes a Claudia Memory section
```

## Config

Config file: `$CLAUDIA_HOME/memory/claudia/config.json` (created on first `save_config` call).

| Key | Default | Description |
|-----|---------|-------------|
| `ollama_host` | `http://localhost:11434` | Ollama daemon URL. Env override: `CLAUDIA_OLLAMA_HOST` |
| `embedding_model` | `all-minilm:l6-v2` | Embedding model (384 dim, fast on CPU) |
| `extraction_model` | `qwen2.5:3b` | LLM for entity extraction |
| `detection_model` | `qwen2.5:3b` | LLM for commitment detection |
| `decay_half_life_days` | `30` | Unverified memory confidence half-life |
| `stale_flag_days` | `60` | Auto-flag pending memories after N days |
| `auto_merge_threshold` | `0.92` | Fuzzy entity merge threshold (higher = more conservative) |
| `retention_days` | `90` | Days before soft-deleted rows are purged |

All settings are non-secret. To use a larger model:

```bash
ollama pull llama3:8b
```

Then edit `$CLAUDIA_HOME/memory/claudia/config.json`:
```json
{
  "extraction_model": "llama3:8b",
  "detection_model": "llama3:8b"
}
```

## Tools

Claudia exposes 18 `memory.*` tools to the LLM. Usable directly via the model's tool-calling surface or dispatched through `manager.handle_tool_call()`.

### Read
| Tool | What it does |
|------|-------------|
| `memory.recall` | Hybrid semantic + lexical + importance search |
| `memory.about` | Exact entity lookup (case-insensitive) |
| `memory.search_entities` | Fuzzy substring search with `*` wildcard |
| `memory.commitments` | List commitments by status (`open`/`completed`/`all`) |
| `memory.recent` | Chronological view by `accessed_at` or `created_at` |
| `memory.trace` | Walk a memory's correction chain (provenance audit) |
| `memory.status` | Counts, current offline mode, session id |
| `memory.metrics` | Internal counters (extractions, detections, tool calls, errors) |

### Write
| Tool | What it does |
|------|-------------|
| `memory.remember` | Store a single fact |
| `memory.correct_memory` | Replace a memory with a corrected version (creates `corrected_from` link, marks old as contradicts) |
| `memory.relate` | Create a directed relationship between two entities |

### State mutation (trust signals)
| Tool | What it does |
|------|-------------|
| `memory.verify_memory` | Mark as verified (full recall weight) |
| `memory.flag_memory` | Mark as flagged (0.5× recall weight) |
| `memory.contradicts_memory` | Mark as contradicting another (0.3× weight) |
| `memory.commitment_complete` | Mark a commitment as done |
| `memory.commitment_drop` | Mark a commitment as abandoned |

### Delete
| Tool | What it does |
|------|-------------|
| `memory.forget_memory` | Soft-delete a specific memory |
| `memory.forget_entity` | Soft-delete an entity by name |

## Maintenance

Three public methods for scheduled maintenance (no CLI wrapping yet — call from your host agent or cron script):

```python
provider.consolidate()    # fuzzy entity dedup + commitment FK resolution
provider.verify()         # confidence decay + stale flagging
provider.purge_old_soft_deletes()   # permanent removal past retention window
```

Typical cadence: `verify` daily, `consolidate` weekly, `purge` monthly. All three are idempotent — running them twice in a row is a no-op the second time.

## Trust model

Claudia tracks where every fact came from and how much to trust it. Every memory has:

- **`origin`**: `user_stated`, `extracted`, `inferred`, or `corrected`
- **`confidence`**: 0.0–1.0, decays over time for extracted/inferred memories
- **`verification`**: `pending`, `verified`, `flagged`, or `contradicts`
- **`source_type`** + **`source_ref`**: provenance chain back to the originating turn or document

The final recall score is:

```
score = (vector + importance + recency + fts + rehearsal) × confidence × verification_multiplier
```

Verification multipliers:
- `verified` → 1.0
- `pending` → 1.0 (default, not a penalty)
- `flagged` → 0.5
- `contradicts` → 0.3

User corrections create a new `origin='corrected'` row linked via `corrected_from` and mark the old row as `contradicts`. The old row stays in the DB for audit — `memory.trace(id)` walks the full chain.

## Quick-start walkthrough

After setup, talk to the agent as you normally would. Claudia runs in the background:

1. Say *"I met Sarah Chen today — she's CTO at Acme Labs"*
2. Claudia extracts `Sarah Chen` (person) and `Acme Labs` (organization) via the LLM
3. Say *"I'll send her the proposal by Friday"*
4. Claudia detects the commitment, parses the deadline, stores it with source_ref=this session
5. Ask *"What did I promise Sarah?"* — Claudia's `memory.commitments` + `memory.about` tools surface the pending commitment
6. When you send the proposal, say *"I sent Sarah the proposal"* — the LLM calls `memory.commitment_complete`
7. Weeks later, Claudia's `verify()` pass starts decaying the unverified "Sarah is CTO" memory. Tell the LLM *"Yes, Sarah is still CTO"* and it calls `memory.verify_memory` to lock in full trust weight
8. If Sarah switches roles, say *"Actually, Sarah is now VP Engineering"*. The LLM calls `memory.correct_memory` — new fact at confidence 1.0, old fact marked as contradicts, full audit trail via `memory.trace`

All of this happens **locally**. Your conversation never leaves your machine.

## Troubleshooting

**`claudia doctor` says "Ollama not reachable"**
- Start Ollama: `ollama serve` (or open the Ollama desktop app)
- Verify: `curl http://localhost:11434/api/tags` should return JSON
- Check the `ollama_host` setting in `$CLAUDIA_HOME/memory/claudia/config.json`

**`claudia memory setup` doesn't show claudia**
- The plugin directory must be at `plugins/memory/claudia/` inside the fork
- Must contain both `__init__.py` and `plugin.yaml`
- Run `python -c "from plugins.memory import discover_memory_providers; print(discover_memory_providers())"` to debug

**Extraction / detection aren't running**
- Check `memory.metrics` — if `cognitive.extractions_run` stays at 0 but `sync_turn.calls` increases, Ollama is probably down
- `cognitive.extraction_errors` being nonzero means the LLM call itself failed — check Ollama logs
- On a very tight token budget (remaining_tokens < 2000) the plugin skips cognitive work by design (see `cognitive.runs` vs `sync_turn.skipped`)

**The database is growing unbounded**
- Run `provider.purge_old_soft_deletes()` to drop soft-deleted rows older than `retention_days` (default 90 days)
- Lower the threshold if you need tighter retention: `claudia memory setup` or edit `config.json`

**A fact keeps coming back wrong**
- Use `memory.correct_memory(id, new_content)` — the new version takes effect immediately and the old version is demoted to `contradicts` in ranking
- Or `memory.flag_memory(id)` if you're not sure what the right answer is yet

## Data locations

| What | Where |
|------|-------|
| SQLite database | `$CLAUDIA_HOME/memory/claudia/claudia.db` |
| Config | `$CLAUDIA_HOME/memory/claudia/config.json` |
| Embeddings | In the DB (BLOB column, little-endian float32) |

The database is plain SQLite — you can inspect it with `sqlite3 $CLAUDIA_HOME/memory/claudia/claudia.db` or any SQLite browser.

## Design notes

- **Single serialized writer.** All writes go through one `WriterQueue` thread. Readers share a bounded pool of connections in WAL mode. No write-write conflicts, no lock contention under typical load.
- **Dedicated cognitive executor.** LLM calls run on a separate single-worker thread so `sync_turn` stays non-blocking even when extraction takes 1–5 seconds.
- **Last-token safety for people.** Fuzzy entity merging refuses to merge two people whose last names differ, even if the SequenceMatcher score is high. Prevents "Joel Salinas" and "Joel Hart" from collapsing.
- **Idempotent maintenance.** `consolidate`, `verify`, and `purge` all use absolute-target math rather than relative decay, so running them twice in a row is a no-op the second time.
- **Protected origins.** `user_stated` and `corrected` memories are exempt from confidence decay. Explicit human assertions never drift.
- **Profile isolation.** Every query filters on `profile`. Multi-user gateway sessions share the same plugin instance without cross-contamination. Cross-profile tool calls return "unknown id" errors rather than revealing that the id exists elsewhere.

## Reference

- Design doc: `docs/decisions/memory-provider-design.md`
- Phase 2B→2C handoff: `plans/phase-2c-handoff.md`
- Source: `plugins/memory/claudia/`
- Tests: `tests/plugins/memory/claudia/` (~900 tests)

## License

MIT.
