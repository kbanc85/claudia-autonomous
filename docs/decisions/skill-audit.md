# Skill audit (Phase 3.1 + 3.3 output)

**Author**: Phase 3 analytical session, 2026-04-09 (parallel-track with Phase 2A)
**Scope**: The 12 core MVP skills from the Claudia Autonomous roadmap v3, mapped to Hermes tool dependencies and ported-skill requirements

## Skills matrix

Columns:
- **Skill** — the 12 core skills named in the roadmap
- **Claude Code dep** — does it rely on Claude Code-specific file/bash access?
- **Hermes equivalent** — which existing Hermes tool fills the need
- **Claudia memory need** — what the new Phase 2A provider must expose
- **Model compat** — expected difficulty across frontier vs local models
- **Port decision** — keep / rework / defer

| # | Skill | Claude Code dep | Hermes equivalent | Claudia memory need | Model compat | Port decision |
|---|---|---|---|---|---|---|
| 1 | `claudia-draft` | None — prose composition only | None needed beyond normal chat | Entity lookup (who am I drafting to), relationship context (tone, history) | Easy on frontier; local models may struggle with tone matching | **Keep** — highest-value skill, port first |
| 2 | `morning-brief` | None — produces structured text | Cron delivery hooks (Phase 4.3) | `recall()` on commitments + cooling relationships, `about()` on active projects | Easy on frontier; medium on local (structured output generation) | **Keep** — drives cron integration in Phase 4 |
| 3 | `inbox-check` | None | Gmail MCP integration (external) or email gateway adapter | Entity lookup per sender, importance boost for known-important contacts | Medium — depends on external Gmail MCP reliability | **Keep** — gateway-triggered |
| 4 | `research` / `cross-reference-research` | None | `exa-py`, `firecrawl-py`, `web_tools.py` (already in pyproject deps) | `remember()` each source with provenance for later recall | Easy (all models handle web research prompts) | **Keep** — highest-leverage productivity skill |
| 5 | `capture-meeting` | File upload for transcripts | `file_tools.read_file_tool`, transcription via STT/TTS tools | `remember()` extracted entities, commitments, and meeting summary with provenance to source file | Medium — structured extraction varies by model | **Keep** — feeds 2B commitment lifecycle |
| 6 | `draft-reply` | None | Email adapter (Phase 5 gateway) | Thread context + sender relationship history | Easy on frontier; harder on local (tone matching) | **Keep** — second most-used skill |
| 7 | `commit-commands` | Bash execution | `terminal_tool` for git commands, `file_tools` for diffs | Minimal — skill is procedural | Easy on all models (well-understood git flow) | **Keep** — no memory dependency, can ship first |
| 8 | `meeting-prep` | None | Calendar MCP (optional), memory heavy lifting | `recall()` on person, relationship history, prior meeting captures, open commitments with this person | Easy on frontier; medium on local (context assembly) | **Keep** — showcases the chief-of-staff positioning |
| 9 | `new-person` | None | None beyond memory | `remember()` with entity CRUD to create person record + initial relationship edge | Easy — simple structured extraction | **Keep** — low-risk, ship early |
| 10 | `send-followup` | None | Email adapter (Phase 5 gateway) | Meeting capture recall + person relationship context | Easy on frontier; medium on local | **Keep** — completes the meeting intelligence loop |
| 11 | `what-am-i-missing` | None | None beyond memory and pattern detection | Full recall across commitments, relationships, and patterns table (Phase 2B) | Medium — requires long-context reasoning over memory dumps | **Keep** — unique value, differentiates Claudia from generic agents |
| 12 | `weekly-review` | None | None beyond memory + aggregation | Bulk recall over the week's activity + commitment resolution stats | Medium — structured output over bulk data | **Keep** — drives engagement and continuous use |

## Dependency summary

### Hermes tools all 12 skills depend on

- **Memory provider plugin (Phase 2A)** — all 12 skills require `recall()` and/or `remember()` at some point. The Phase 2A.2 provider must ship before Phase 3.4 porting begins.
- **Terminal backend** — only `commit-commands` needs bash execution. All others are prose-only.
- **File reading** — only `capture-meeting` needs to read uploaded transcript files.

### External Hermes infrastructure the skills rely on

- **Gateway message format** (Phase 5.2) — 6 skills interact with messaging platforms (morning-brief delivery, inbox-check, draft-reply, send-followup, capture-meeting file upload, weekly-review delivery)
- **Cron scheduler** (Phase 4.3) — 4 skills trigger on schedules (morning-brief daily, commitment-check daily, weekly-review weekly, relationship-health weekly)
- **Web tools** — 1 skill (`research`) uses `exa-py` / `firecrawl-py`
- **Calendar integration** — 1 skill (`meeting-prep`) would benefit from calendar MCP but can work without

### Claude Code specific features that DON'T carry over

Looking at each skill's Claudia v1 source in `template-v2/.claude/skills/`, the main Claude Code-specific features used are:
- `Task` tool (agent delegation) → Hermes has `tools/delegate_tool.py` which is a close equivalent, though with different ergonomics
- `TodoWrite` tool (task tracking) → no Hermes equivalent; skills that use it need to be reworked to use Claudia memory for task state instead
- MCP tools beyond what Hermes exposes → the core 12 skills don't require anything Hermes doesn't already have

## Conflicts with existing Hermes tools

Searched `tools/` for names or behaviours that would shadow Claudia skills. Findings:

- **`tools/memory_tool.py`** — the built-in `remember` / `recall` tool exists and is always active. Claudia's provider exposes additional methods (`about`, `search_entities`, `trace`) that don't conflict; the base `remember`/`recall` get wrapped to Claudia's backend via the ABC.
- **`tools/web_tools.py`** — already implements `web_search` via Exa and Firecrawl. `research` skill should use this, not a new implementation.
- **`tools/file_tools.py`** — `capture-meeting` uses the existing `read_file_tool` without conflict.
- **`tools/terminal_tool.py`** — `commit-commands` uses existing shell execution without conflict.

**No conflicts found.** All 12 skills can be ported as pure markdown skill files using existing Hermes tools.

## Port priorities

Based on dependency order (low first = can ship earliest):

**Priority 1 (can ship during Phase 2A in parallel; no memory dependency):**
- `commit-commands`
- `new-person` (only uses entity CRUD, ships as soon as 2A.2d lands)

**Priority 2 (needs Phase 2A.2 provider for recall):**
- `claudia-draft`
- `research`

**Priority 3 (needs Phase 2A + 2B commitment lifecycle):**
- `morning-brief`
- `meeting-prep`
- `what-am-i-missing`
- `weekly-review`

**Priority 4 (needs Phase 5 gateway adapters for delivery):**
- `inbox-check`
- `draft-reply`
- `send-followup`

**Priority 5 (needs file upload infrastructure):**
- `capture-meeting`

Suggested Phase 3.4 porting order: P1 → P2 → P3 → P4 → P5. This gives Claudia a functional skill at every stage rather than requiring all-or-nothing ship.

## Rollback candidates

Per the roadmap: if Phase 3 hits problems, keep only the 5 highest-usage skills: `claudia-draft`, `morning-brief`, `inbox-check`, `research`, `meeting-prep`. Archive the rest. These 5 span draft + daily rhythm + info lookup + meeting workflow, which is the minimum viable chief-of-staff product.

## Notes for Phase 3.2 (model compatibility test script)

The automated test should, for each skill in the 12-core list:

1. Load the skill's prompt template
2. Send to 3+ model endpoints:
   - **Frontier**: Claude Sonnet (via OpenRouter), GPT-4.1 (via OpenRouter)
   - **Local**: Llama 3.3 70B (via Ollama), Gemma 4 (via Ollama)
3. Capture output
4. Score:
   - **Malformed JSON / tool call format failures** — critical, disqualifies the model-skill pair
   - **Missed instructions** — reduces score
   - **Hallucinated tool calls** (tools that don't exist) — critical
   - **Tone matching** — qualitative, scored 1-5
5. Write compatibility report to `../data/skill-compat-reports/<timestamp>.json`

Test script location: `tests/skill_compatibility.py` with `@pytest.mark.frontier` and `@pytest.mark.local` markers (added to pyproject.toml in Phase 0.4). Run explicitly via `pytest -m frontier` or `pytest -m local`.

**Script is not yet written.** Writing the actual script requires live API credentials and sample prompts, which belong in an implementation session rather than this analytical pass.

## Reference

- Phase 2A.1 memory provider design → `memory-provider-design.md`
- Claudia v1 skill sources → outer claudia repo `template-v2/.claude/skills/`
- 12 core skills list → roadmap section "Claudia codebase: what you're infusing"
