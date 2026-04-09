# Getting started with Claudia Hybrid Memory

A 15-minute walkthrough from zero to a working trust-aware memory system, with Ollama running locally, the plugin activated in your claudia-autonomous session, and your first few memories stored and recalled.

If you want the full reference, read the plugin's [README.md](../README.md) instead. This doc is the narrative version.

## What you'll end up with

- Ollama running on your machine with two small models pulled
- Claudia-autonomous configured to use the `claudia` memory provider
- A SQLite database under `~/.claudia/memory/claudia/claudia.db` that grows as you talk to the agent
- A verified understanding of the 18 `memory.*` tools the LLM can call
- A clean baseline to start iterating on your own prompts, entities, and commitments

## Prerequisites

- macOS, Linux, or Windows with WSL2
- Python 3.11 or newer (the fork ships `pyproject.toml` with this floor)
- ~3 GB of free disk space (Ollama + the two default models)
- Internet for the initial model pull (everything after is offline)

## Step 1 — Install Ollama (~2 minutes)

**macOS** (recommended install method — gives you the menubar daemon):

```bash
brew install ollama
# or download the .dmg from https://ollama.com/download
open -a Ollama
```

**Linux** (one-line installer):

```bash
curl -fsSL https://ollama.com/install.sh | sh
sudo systemctl enable --now ollama
```

**Windows / WSL2**: grab the installer from [ollama.com/download](https://ollama.com/download).

Verify Ollama is reachable:

```bash
curl http://localhost:11434/api/tags
```

You should get back a JSON response listing any models you've already pulled. On a fresh install it'll be an empty array — that's fine.

## Step 2 — Pull the default models (~3 minutes)

Claudia uses two small models by default. They're intentionally chosen for speed on CPU-only machines; you can swap them for larger ones later.

```bash
# Embedding model — 23 MB, produces 384-dim vectors
ollama pull all-minilm:l6-v2

# LLM for entity extraction and commitment detection — 2 GB
ollama pull qwen2.5:3b
```

If your connection is slow, you can run these in parallel. Pulls resume on retry.

## Step 3 — Clone claudia-autonomous and install (~3 minutes)

```bash
git clone https://github.com/kbanc85/claudia-autonomous.git
cd claudia-autonomous
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
```

(If you don't have `uv`, `pip install -e ".[all,dev]"` works just as well — it's just slower.)

Verify the install:

```bash
python -m pytest tests/plugins/memory/claudia/ -q
```

You should see ~770 passing tests in ~10 seconds. If any fail, file an issue before continuing.

## Step 4 — Activate Claudia as your memory provider (~1 minute)

```bash
claudia memory setup
```

An interactive picker appears showing every discovered memory provider. Use arrow keys to highlight **claudia** and press Enter.

You'll see a confirmation block followed by a tailored post-setup hint. If Ollama is running with both models pulled, it'll look like this:

```
  ✓ Memory provider: claudia
  ✓ Activation saved to config.yaml

  Claudia is installed and ready.

  ✓ Ollama reachable with both default models.

  Try the demo:
    python -m plugins.memory.claudia.demo

  Read the full docs:
    plugins/memory/claudia/README.md

  Start a new session to activate.
```

If Ollama isn't running yet, the post-setup hint tells you exactly what to do next — it's not a guess.

Verify the activation:

```bash
claudia memory status
```

You should see `Provider: claudia` and `Plugin: installed ✓ Status: available ✓`.

## Step 5 — Run the offline demo (~1 minute)

Before talking to the live agent, run the scripted demo to see every feature of the plugin in isolation:

```bash
python -m plugins.memory.claudia.demo
```

The demo uses fake extractors and detectors so it runs deterministically without touching Ollama. It walks through 12 steps with color-coded output:

1. Spin up a fresh provider in a temp directory
2. Store a manual fact via `memory.remember`
3. Simulate two `sync_turn` calls (Sarah Chen → Acme Labs, then commitment detection)
4. Look up the entity via `memory.about`
5. List commitments
6. Semantic search via `memory.recall`
7. User correction via `memory.correct_memory`
8. Walk the audit trail via `memory.trace`
9. Run `provider.consolidate()`
10. Run `provider.verify()`
11. Print internal metrics
12. Final `memory.status` snapshot

Everything runs in under a second. The temp directory is cleaned up automatically. This is the fastest way to see what Claudia actually does.

## Step 6 — Start a real session (~2 minutes)

```bash
claudia
```

Talk to the agent the way you normally would. Try something like:

> "I met Sarah Chen at Acme Labs today — she's their CTO. I'll send her the proposal by Friday."

Claudia's memory runs in the background. After one turn:

- The memory row is inserted (embedding generated on the caller thread)
- The extractor identifies `Sarah Chen` (person) and `Acme Labs` (organization)
- The commitment detector picks out "send the proposal to Sarah" with a parsed Friday deadline
- All three writes land in the DB via the writer queue

In the next turn, try:

> "What did I promise Sarah?"

The LLM should call `memory.commitments` (or `memory.recall` + `memory.about`) and surface the pending commitment with the deadline. If it doesn't, try being more specific: *"Use memory.commitments to find what's open."*

Tip: the 18 tools are all exposed via the model's tool-calling surface. Well-instructed models pick them up automatically, but prompting the LLM with *"use the memory tools"* helps when the agent is being shy.

## Step 7 — Run maintenance (~30 seconds)

Claudia has three maintenance operations. You can run them manually from a Python shell or wire them into a cron job.

From a Python shell inside the fork:

```python
from plugins.memory.claudia import ClaudiaMemoryProvider

p = ClaudiaMemoryProvider()
p.initialize(
    session_id="manual-maintenance",
    claudia_home="~/.claudia",
    platform="cli",
)

# Run them in the recommended order
verify_result = p.verify(timeout=30)
print(f"Decayed: {verify_result.decayed_count}, flagged: {verify_result.flagged_stale_count}")

cons_result = p.consolidate(timeout=30)
print(f"Merged: {cons_result.entities_merged}, linked: {cons_result.commitments_linked}")

# Destructive — only run after you're comfortable with retention
purge_result = p.purge_old_soft_deletes(timeout=30)
print(f"Purged: {purge_result.total()} rows")

p.shutdown()
```

Recommended cadence: `verify` daily, `consolidate` weekly, `purge_old_soft_deletes` monthly. All three are idempotent.

## Step 8 — Inspect the database (~1 minute)

The SQLite database is plain vanilla — you can open it with any SQLite browser:

```bash
sqlite3 ~/.claudia/memory/claudia/claudia.db
```

Some queries to get a feel for what's stored:

```sql
-- How many memories total?
SELECT COUNT(*) FROM memories WHERE deleted_at IS NULL;

-- Top 10 most-recent memories
SELECT content, origin, verification, confidence
FROM memories
WHERE deleted_at IS NULL
ORDER BY accessed_at DESC
LIMIT 10;

-- All people Claudia knows about
SELECT name, importance, access_count
FROM entities
WHERE kind = 'person' AND deleted_at IS NULL;

-- Open commitments
SELECT content, deadline, status
FROM commitments
WHERE status = 'open' AND deleted_at IS NULL;

-- Correction history: memories that supersede other memories
SELECT a.id AS new_id, a.content AS new_content,
       b.id AS old_id, b.content AS old_content, b.verification AS old_status
FROM memories a
JOIN memories b ON a.corrected_from = b.id;
```

## Step 9 — Run `claudia doctor` to confirm everything

```bash
claudia doctor
```

Look for the `◆ Claudia Memory` section. It should show:

- `✓ Claudia plugin discoverable`
- `✓ Claudia database   ~/.claudia/memory/claudia/claudia.db (N KB)`
- `ℹ Claudia config file not set` (using defaults, fine) — or `✓ Claudia config   N key(s) set` if you've run `save_config`
- `✓ Ollama reachable   N model(s) available`
- (no warnings about missing default models)

If any of those are failing or warning, the doctor message tells you exactly what to run.

## Step 10 — Customize (~as long as you want)

The 8 config fields are all in `$CLAUDIA_HOME/memory/claudia/config.json`:

```json
{
  "ollama_host": "http://localhost:11434",
  "embedding_model": "all-minilm:l6-v2",
  "extraction_model": "qwen2.5:3b",
  "detection_model": "qwen2.5:3b",
  "decay_half_life_days": 30,
  "stale_flag_days": 60,
  "auto_merge_threshold": 0.92,
  "retention_days": 90
}
```

Edit and restart your session for changes to take effect. Common tweaks:

- **Bigger LLM for better extraction** — `ollama pull llama3:8b` and set `extraction_model` / `detection_model` to `llama3:8b`
- **More aggressive decay** — drop `decay_half_life_days` to 14 if you want unverified facts to fade faster
- **Tighter dedup** — raise `auto_merge_threshold` to 0.95 if consolidation is merging too eagerly
- **Longer retention** — raise `retention_days` to 365 if you never want to lose history automatically

## What next?

- Read the [plugin README](../README.md) for the reference documentation on every tool and config field
- Read the [CHANGELOG](../CHANGELOG.md) to see what's in v1.0 and what's explicitly deferred
- Look at [`demo.py`](../demo.py) as a template for writing your own integration script
- Browse the tests in `tests/plugins/memory/claudia/` — they're the most honest documentation of every edge case and invariant the plugin preserves

Troubleshooting is in the [README's troubleshooting section](../README.md#troubleshooting). If something's wrong that isn't covered there, file an issue with the output of `claudia doctor` and the last few lines of your session log.

Welcome to Claudia.
