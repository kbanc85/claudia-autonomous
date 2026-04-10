<p align="center">
  <img src="assets/claudia.png" alt="Claudia" width="134" height="135">
</p>

<h1 align="center">Claudia Autonomous</h1>

<h3 align="center">A thinking partner who runs 24/7 and tracks relationships, not just tasks.</h3>

<p align="center">
Remembers your people. Catches your commitments. Learns how you work.<br>
Available in your terminal, on Telegram, Discord, Slack, WhatsApp, Signal, or email.
</p>

<p align="center">
  <a href="https://github.com/kbanc85/claudia-autonomous/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT"></a>
  <a href="https://github.com/kbanc85/claudia-autonomous/releases/tag/v1.0.0"><img src="https://img.shields.io/badge/memory%20plugin-v1.0.0-brightgreen?style=flat-square" alt="Memory plugin v1.0.0"></a>
  <a href="https://github.com/kbanc85/claudia-autonomous"><img src="https://img.shields.io/badge/agent-pre--beta-orange?style=flat-square" alt="Agent pre-beta"></a>
  <a href="https://github.com/kbanc85/claudia-autonomous/blob/main/THIRD-PARTY.md"><img src="https://img.shields.io/badge/forked%20from-Hermes%20v0.7.0-blueviolet?style=flat-square" alt="Forked from Hermes Agent"></a>
</p>

<p align="center">
  <a href="#install"><strong>Install</strong></a> ·
  <a href="#what-claudia-does">What she does</a> ·
  <a href="#whats-different-about-this-version">What's different</a> ·
  <a href="#safety">Safety</a> ·
  <a href="https://claudia.aiadopters.club">Website</a>
</p>

---

## Install

### macOS, Linux, WSL2

```bash
curl -fsSL https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.sh | bash
```

### Windows (PowerShell)

```powershell
iwr -useb https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.ps1 | iex
```

Works on Windows PowerShell 5.1 (built in) or PowerShell 7+. No execution policy change needed.

### After the install finishes

Close and re-open your terminal (so the updated `PATH` takes effect), then:

```bash
claudia memory setup   # pick 'claudia' in the picker
claudia doctor         # verify everything is wired up
claudia                # start chatting
```

That's it. Full install options and troubleshooting are in the [install guide below](#install-details).

---

## What Claudia does

<table>
<tr>
<td width="33%" align="center">
<h3>Catches Commitments</h3>
<p>Say "I'll send that by Friday" and she tracks it. On Friday morning, she reminds you. Past due? She escalates.</p>
</td>
<td width="33%" align="center">
<h3>Knows Your People</h3>
<p>Every person she meets gets a living profile: communication preferences, history, current concerns. Flags cooling contacts before they go dark.</p>
</td>
<td width="33%" align="center">
<h3>Spots Patterns You Miss</h3>
<p>Overcommitting again? A key relationship going cold? The same mistake twice? She sees it forming and speaks up, gently.</p>
</td>
</tr>
<tr>
<td width="33%" align="center">
<h3>Shows Her Sources</h3>
<p>Every fact traces to its source. Ask "how do you know that?" and she shows the receipt. Contradictions surface instead of getting hidden.</p>
</td>
<td width="33%" align="center">
<h3>Drafts, Never Sends</h3>
<p>Every external action (email, message, calendar invite, purchase) requires your explicit confirmation. Drafts are shown in full before anything ships.</p>
</td>
<td width="33%" align="center">
<h3>Lives Where You Do</h3>
<p>Terminal, Telegram, Discord, Slack, WhatsApp, Signal, email. One gateway process for all of them. Your persona follows you.</p>
</td>
</tr>
<tr>
<td width="33%" align="center">
<h3>Runs Anywhere</h3>
<p>Local on your laptop. Dockerized on a VPS. Serverless on Modal or Daytona. Persistent over SSH on a Mac Mini. Same Claudia, any host.</p>
</td>
<td width="33%" align="center">
<h3>Any Model, No Lock-in</h3>
<p>Switch between Claude Sonnet, GPT-4.1, Llama 3.3, local Ollama, or custom endpoints with <code>claudia model</code>. Your memory and skills are portable.</p>
</td>
<td width="33%" align="center">
<h3>Remembers Overnight</h3>
<p>Background maintenance decays old memories, merges near-duplicates, flags stale facts. Each morning she knows a little more than yesterday.</p>
</td>
</tr>
</table>

---

## What's different about this version

There are two Claudias. This is the autonomous one.

**[`kbanc85/claudia`](https://github.com/kbanc85/claudia)** is the desktop + Obsidian variant. Install via `npx get-claudia`. Her memory syncs to an Obsidian vault organized with PARA. Best for solo users who want a second brain in plain markdown.

**`kbanc85/claudia-autonomous`** (this repo) is the server variant. Install via the one-liner above. She runs as a persistent process with a messaging gateway, answers from Telegram and Slack and email, and supports any model from any provider. Best for people who want her reachable from anywhere, not just the terminal they're sitting at.

Both share the same identity, principles, and memory model. You can run both side by side.

---

## Install details

### What the installer does

Both installers check prerequisites (git, python 3.11+, Ollama), install anything missing (Homebrew on macOS, the official script on Linux, `winget` on Windows), clone the repo, create a venv with `uv`, install the package in editable mode, put the `claudia` CLI on your PATH, and pull the two default Ollama models for the hybrid memory plugin (`all-minilm:l6-v2` ~23 MB for embeddings, `qwen2.5:3b` ~2 GB for entity extraction and commitment detection).

Total time: about 5 to 10 minutes on a decent connection, mostly the `qwen2.5:3b` download.

### Default install location

| OS | Path |
|---|---|
| macOS, Linux, WSL2 | `~/claudia-autonomous` |
| Windows | `%USERPROFILE%\claudia-autonomous` |

### Environment overrides

Both installers honor the same environment variables. Set them BEFORE running the install command:

| Variable | Default | Purpose |
|---|---|---|
| `INSTALL_DIR` | `~/claudia-autonomous` (or `%USERPROFILE%\claudia-autonomous` on Windows) | Where to clone the repo |
| `BRANCH` | `main` | Branch or tag to check out |
| `SKIP_OLLAMA` | unset | Skip Ollama install and model pull |
| `SKIP_MODELS` | unset | Skip model pull only (Ollama is still installed) |
| `SKIP_SETUP` | unset | Skip running the setup script (clone only) |

Example, Windows with a custom path and pinned version:

```powershell
$env:INSTALL_DIR = 'D:\dev\claudia'
$env:BRANCH      = 'v1.0.0'
iwr -useb https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.ps1 | iex
```

### Dev setup (clone manually)

```bash
# macOS, Linux, WSL2
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

---

## Commands

```bash
claudia                     # Start a conversation
claudia model               # Choose your LLM provider and model
claudia memory setup        # Activate Claudia's hybrid memory provider
claudia doctor              # Diagnose configuration and dependencies
claudia tools               # Configure which tools are enabled
claudia gateway start       # Run the messaging gateway (Telegram, Discord, etc.)
claudia update              # Update to the latest version
```

Slash commands inside a session: `/new`, `/model`, `/usage`, `/skills`, `/stop`. They work in the terminal and in any connected messaging platform.

---

## How she's built

Claudia Autonomous is a fork of [Hermes Agent v0.7.0](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.4.3) (MIT) with the chief-of-staff layers added on top:

- **Hybrid memory** ([`plugins/memory/claudia/`](plugins/memory/claudia/)) — **shipped in v1.0**. SQLite + vector embeddings with trust-aware ranking: `(50% vector + 25% importance + 10% recency + 15% FTS + rehearsal) × confidence × verification multiplier`. Local-first via Ollama, graceful offline fallback, 18 LLM tools including commitment tracking, entity/relationship management, user corrections with full audit trails, and internal metrics. Run `claudia memory setup` to activate, or [read the plugin README](plugins/memory/claudia/README.md) for the full reference.
- **Commitment lifecycle** — shipped with the memory plugin. Hybrid pattern + LLM detection on user turns, status tracking, natural-language deadline parsing, mutation tools for completion and drop.
- **Relationship graphs** — shipped. Entity upsert via extraction, explicit linking, fuzzy consolidation with last-token safety for people, relationship CRUD with health scoring.
- **Chief-of-staff persona** — a rich identity injected via `SOUL.md` that shapes every response toward strategic thinking, proactive commitment tracking, and relational awareness.
- **Proactive daily behaviour** (in progress) — morning briefs, commitment checks, relationship health reports delivered on schedule via the cron subsystem.

The execution backends (local, Docker, SSH, Daytona, Singularity, Modal), the messaging gateway layer, the skill system, and the ~40 tools are inherited from Hermes v0.7.0 and remain fully compatible. Attribution is preserved in [THIRD-PARTY.md](THIRD-PARTY.md).

> **Status: pre-beta.** The v0.1.0-beta release goes out to a small group once the persona and proactive intelligence layers ship. The hybrid memory plugin is already v1.0 and production-ready. See [the implementation roadmap](https://github.com/kbanc85/claudia/blob/main/autonomous/roadmap/claudia-autonomous-roadmap-v3.md) in the tracking hub repo for what's planned.

---

## Safety

Claudia's defining trait is that **she never takes external actions without your explicit approval in the current conversation.**

- Drafts are shown in full (recipients, content, timing) before any send
- Each significant action is confirmed individually. Blanket permissions do not override individual confirmations
- Dangerous shell commands (`rm -rf`, `chmod 777`, `curl | sh`, process kills, etc.) trigger a pattern-based detection layer on top of the approval gate
- Self-termination is blocked: she cannot kill her own process or the gateway
- Secrets embedded in URLs are stripped or blocked before `browser_navigate` sends them
- Log output passes through regex-based redaction for OpenAI, Anthropic, GitHub, Slack, and Google key formats
- Docker backend runs with `--cap-drop ALL`, `--security-opt no-new-privileges`, and PID limits
- DM pairing uses cryptographic 8-char codes, rate limiting, and attempt lockout (OWASP / NIST SP 800-63-4 compliant)

A full security baseline audit lives at [`docs/decisions/security-baseline.md`](docs/decisions/security-baseline.md).

---

## Privacy

Claudia is local-first. The hybrid memory plugin runs 100% on your machine via Ollama. No API keys, no cloud services, no telemetry. Your memories, your relationships, your commitments, and your conversations stay on your hardware.

You can still use Claude Sonnet, GPT-4.1, or any cloud model for the reasoning layer if you want, and then your turns go to that provider. That's your choice, and you swap providers with `claudia model`. The memory layer never leaves your machine either way.

---

## Contributing

Claudia is pre-beta and in active development. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and PR process, and the [implementation roadmap](https://github.com/kbanc85/claudia/blob/main/autonomous/roadmap/claudia-autonomous-roadmap-v3.md) in the tracking hub repo for what's planned.

Running the test suite:

```bash
python -m pytest tests/ -q
```

The memory plugin tests alone are ~920 tests that run in about 10 seconds.

---

## License

MIT. See [LICENSE](LICENSE).

Claudia Autonomous is a fork of [Hermes Agent v0.7.0](https://github.com/NousResearch/hermes-agent/releases/tag/v2026.4.3) by Nous Research (MIT). Attribution is preserved in [THIRD-PARTY.md](THIRD-PARTY.md) and the original LICENSE header.

Built by [Kamil Banc](https://github.com/kbanc85).
