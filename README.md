<p align="center">
  <img src="assets/claudia.png" alt="Claudia" width="134" height="135">
</p>

<h1 align="center">Claudia</h1>

<p align="center">
  <strong>Terminal-based AI chief of staff. Remembers relationships, tracks commitments, helps you think strategically.</strong>
</p>

<p align="center">
  <a href="https://github.com/kbanc85/claudia-autonomous/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://github.com/kbanc85/claudia-autonomous/releases/tag/v1.0.0"><img src="https://img.shields.io/badge/Memory%20Plugin-v1.0.0-brightgreen?style=for-the-badge" alt="Memory plugin v1.0.0"></a>
  <a href="https://github.com/kbanc85/claudia-autonomous"><img src="https://img.shields.io/badge/Agent-Pre--beta-orange?style=for-the-badge" alt="Agent pre-beta"></a>
  <a href="https://github.com/kbanc85/claudia-autonomous/blob/main/THIRD-PARTY.md"><img src="https://img.shields.io/badge/Forked%20from-Hermes%20v0.7.0-blueviolet?style=for-the-badge" alt="Forked from Hermes Agent"></a>
</p>

Claudia runs 24/7 on a laptop, VPS, Mac Mini, or cloud VM. She tracks your commitments, remembers your relationships, notices patterns across conversations, and never takes external action without your explicit approval.

Talk to her from the terminal, from Telegram, Discord, Slack, WhatsApp, Signal, or email — all from a single gateway process. Switch between Claude Sonnet, GPT-4.1, Llama 3.3, or a local Ollama model with one command. No lock-in to any provider.

> **Status: pre-beta.** Claudia Autonomous is a fork of [Hermes Agent v0.7.0](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.4.3), in active development. The v0.1.0-beta release will go out to a small group once the chief-of-staff memory, persona, and proactive intelligence layers ship. See [the implementation roadmap](https://github.com/kbanc85/claudia/blob/main/autonomous/roadmap/claudia-autonomous-roadmap-v3.md) for what's planned.

## What Claudia does

| | |
|---|---|
| **Thinks in trends, not moments** | Notices recurring themes, surfaces them gently. "This is the third time you've committed to something without checking your calendar." |
| **Tracks commitments** | When you say "I'll send that by Friday", she remembers. Surfaces approaching and overdue items in daily briefs. |
| **Manages relationships** | Maintains context per person: communication preferences, history, current concerns. Flags cooling contacts before they go dark. |
| **Drafts, never sends** | Every external action (email, message, purchase, calendar invite) requires your explicit confirmation. Drafts are shown in full before anything ships. |
| **Lives where you do** | Terminal, Telegram, Discord, Slack, WhatsApp, Signal, email. One gateway process for all of them. |
| **Runs anywhere** | Local on your laptop. Dockerized on a VPS. Serverless on Modal or Daytona. Persistent SSH on a Mac Mini. |
| **Any model, no lock-in** | Switch with `claudia model` between Claude Sonnet, GPT-4.1, Llama 3.3, Ollama, or custom endpoints. Your persona, memory, and skills are portable. |
| **Proactive, not reactive** | Morning briefs, commitment checks, relationship health reports delivered on schedule via the cron subsystem. |

## Install

### macOS, Linux, WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.sh | bash
```

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.ps1 | iex
```

Runs in Windows PowerShell 5.1 (built in) or PowerShell 7+. No execution policy change needed: the `iwr | iex` pattern runs the installer as an in-memory expression, which bypasses the on-disk execution policy check the way Scoop and uv do.

### What the installer does

Both installers check prerequisites (git, python 3.11+, Ollama), install anything missing (Homebrew / `apt` / `winget` depending on your OS), clone the repo, create a venv via `uv`, install the package in editable mode, put the `claudia` CLI on your `PATH`, and pull the two default Ollama models for the hybrid memory plugin (`all-minilm:l6-v2` ~23 MB for embeddings, `qwen2.5:3b` ~2 GB for entity extraction and commitment detection).

Default install location:

| OS | Path |
|---|---|
| macOS, Linux, WSL2 | `~/claudia-autonomous` |
| Windows | `%USERPROFILE%\claudia-autonomous` |

### After the installer finishes

**macOS / Linux / WSL2:**

```bash
source ~/.zshrc            # or ~/.bashrc, so ~/.local/bin is on PATH
claudia memory setup       # pick 'claudia' for the trust-aware memory provider
claudia doctor             # verify everything is wired up
claudia                    # start chatting
```

**Windows:**

```powershell
# Close this PowerShell window and open a new one so the updated PATH takes effect
claudia memory setup       # pick 'claudia' for the trust-aware memory provider
claudia doctor             # verify everything is wired up
claudia                    # start chatting
```

### Environment overrides

Both installers honor the same environment variables. On Windows, set them in the same PowerShell session BEFORE running the `iwr | iex` line:

| Variable | Default (Unix / Windows) | Purpose |
|---|---|---|
| `INSTALL_DIR` | `~/claudia-autonomous` / `%USERPROFILE%\claudia-autonomous` | Where to clone the repo |
| `BRANCH` | `main` | Branch or tag to check out (use a release tag for stability) |
| `SKIP_OLLAMA` | unset | Skip Ollama install and model pull |
| `SKIP_MODELS` | unset | Skip model pull only (Ollama is still installed) |
| `SKIP_SETUP` | unset | Skip running the setup script (clone only) |

Windows example with overrides:

```powershell
$env:INSTALL_DIR = 'D:\dev\claudia'
$env:BRANCH      = 'v1.0.0'
iwr -useb https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.ps1 | iex
```

### Dev setup (clone manually)

```bash
# macOS / Linux / WSL2
git clone https://github.com/kbanc85/claudia-autonomous.git
cd claudia-autonomous
./setup-claudia.sh
```

```powershell
# Windows
git clone https://github.com/kbanc85/claudia-autonomous.git
cd claudia-autonomous
.\setup-claudia.ps1
```

## Getting started

```bash
claudia              # Interactive CLI — start a conversation
claudia model        # Choose your LLM provider and model
claudia tools        # Configure which tools are enabled
claudia gateway      # Start the messaging gateway (Telegram, Discord, etc.)
claudia setup        # Run the full setup wizard
claudia migrate      # Migrate from Hermes or OpenClaw (Phase 6)
claudia doctor       # Diagnose configuration and dependencies
claudia update       # Update to the latest version
```

## CLI vs messaging

Claudia has two entry points: start the terminal UI with `claudia`, or run the gateway and talk to her from a messaging platform. Slash commands work in both.

| Action | CLI | Messaging |
|---|---|---|
| Start chatting | `claudia` | Run `claudia gateway start`, then message the bot |
| Fresh conversation | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model` | `/model` |
| Check usage | `/usage` | `/usage` |
| Browse skills | `/skills` | `/skills` |
| Interrupt | `Ctrl+C` | `/stop` |

## Architecture

Claudia builds on [Hermes Agent v0.7.0](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.4.3) (MIT-licensed) and adds:

- **Chief-of-staff persona** — a rich identity injected via `SOUL.md` that shapes every response toward strategic thinking, proactive commitment tracking, and relational awareness
- **Hybrid memory** ([`plugins/memory/claudia/`](plugins/memory/claudia/)) — **shipped**. SQLite + vector embeddings with trust-aware ranking (`confidence × verification multiplier` on top of `50% vector + 25% importance + 10% recency + 15% FTS` + rehearsal). Local-first (Ollama on localhost), graceful offline fallback, 18 LLM tools including commitment tracking, entity/relationship management, user corrections with full audit trails, and internal metrics. Run `claudia memory setup` and pick "claudia", or `python -m plugins.memory.claudia.demo` for a scripted walkthrough. See the plugin [README](plugins/memory/claudia/README.md) for details.
- **Commitment lifecycle** — shipped as part of the memory plugin. Hybrid pattern + LLM detection on user turns, status tracking (open/completed/overdue/dropped), natural-language deadline parsing, LLM-facing mutation tools (`memory.commitment_complete`, `memory.commitment_drop`, `memory.commitments` listing).
- **Relationship graphs** — shipped. Entity upsert via extraction, explicit linking via `memory.relate`, fuzzy consolidation with last-token safety for people, relationship CRUD with health scoring.
- **Proactive daily behaviour** (Phase 4) — morning briefs, commitment checks, relationship health reports delivered via cron

The execution backends (local, Docker, SSH, Daytona, Singularity, Modal), the messaging gateway layer, the skill system, and the ~40 tools are all inherited from Hermes v0.7.0 and remain fully compatible.

## Safety model

Claudia's defining trait is that **she never takes external actions without your explicit approval in the current conversation**.

- Drafts are shown in full (recipients, content, timing) before any send
- Each significant action gets confirmed individually; blanket permissions do not override individual confirmations
- Dangerous shell commands (`rm -rf`, `chmod 777`, `curl | sh`, process kills, etc.) trigger a pattern-based detection layer in addition to the approval gate
- Self-termination is blocked: the agent cannot kill its own process or the gateway
- Secrets embedded in URLs are stripped or blocked before `browser_navigate` sends them
- Log output passes through regex-based redaction (OpenAI/Anthropic/GitHub/Slack/Google key formats)
- Docker backend runs with `--cap-drop ALL`, `--security-opt no-new-privileges`, and PID limits
- DM pairing uses cryptographic 8-char codes, rate limiting, and attempt lockout (OWASP / NIST SP 800-63-4 compliant)

A full security baseline audit is at [`docs/decisions/security-baseline.md`](docs/decisions/security-baseline.md).

## Contributing

Claudia is pre-beta and in active development. The implementation roadmap lives at [`autonomous/roadmap/claudia-autonomous-roadmap-v3.md`](https://github.com/kbanc85/claudia/blob/main/autonomous/roadmap/claudia-autonomous-roadmap-v3.md) in the tracking hub repo. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR process.

Dev setup:

```bash
git clone https://github.com/kbanc85/claudia-autonomous.git
cd claudia-autonomous
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

## License

MIT. See [LICENSE](LICENSE).

Claudia Autonomous is a fork of [Hermes Agent v0.7.0](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.4.3) by Nous Research (MIT). Attribution is preserved in [THIRD-PARTY.md](THIRD-PARTY.md) and the original LICENSE header.

Built by [Kamil Banc](https://github.com/kbanc85).
