<p align="center">
  <img src="assets/banner.png" alt="Claudia" width="100%">
</p>

# Claudia ☤

<p align="center">
  <a href="https://claudia-autonomous.example.com/docs/"><img src="https://img.shields.io/badge/Docs-claudia--agent.example.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/kbanc85/claudia-autonomous/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://example.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Kamil Banc"></a>
</p>

**The self-improving AI agent built by [Kamil Banc](https://example.com).** It's the only agent with a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from Telegram while it works on a cloud VM.

Use any model you want — [Nous Portal](https://portal.example.com), [OpenRouter](https://openrouter.ai) (200+ models), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), OpenAI, or your own endpoint. Switch with `claudia model` — no code changes, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Six terminal backends — local, Docker, SSH, Daytona, Singularity, and Modal. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, Atropos RL environments, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

```bash
curl -fsSL https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/scripts/install.sh | bash
```

Works on Linux, macOS, and WSL2. The installer handles everything — Python, Node.js, dependencies, and the `claudia` command. No prerequisites except git.

> **Windows:** Native Windows is not supported. Please install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and run the command above.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
claudia              # start chatting!
```

---

## Getting Started

```bash
claudia              # Interactive CLI — start a conversation
claudia model        # Choose your LLM provider and model
claudia tools        # Configure which tools are enabled
claudia config set   # Set individual config values
claudia gateway      # Start the messaging gateway (Telegram, Discord, etc.)
claudia setup        # Run the full setup wizard (configures everything at once)
claudia migrate migrate # Migrate from OpenClaw (if coming from OpenClaw)
claudia update       # Update to the latest version
claudia doctor       # Diagnose any issues
```

📖 **[Full documentation →](https://claudia-autonomous.example.com/docs/)**

## CLI vs Messaging Quick Reference

Claudia has two entry points: start the terminal UI with `claudia`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action | CLI | Messaging platforms |
|---------|-----|---------------------|
| Start chatting | `claudia` | Run `claudia gateway setup` + `claudia gateway start`, then send the bot a message |
| Start fresh conversation | `/new` or `/reset` | `/new` or `/reset` |
| Change model | `/model [provider:model]` | `/model [provider:model]` |
| Set a personality | `/personality [name]` | `/personality [name]` |
| Retry or undo the last turn | `/retry`, `/undo` | `/retry`, `/undo` |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]` |
| Browse skills | `/skills` or `/<skill-name>` | `/skills` or `/<skill-name>` |
| Interrupt current work | `Ctrl+C` or send a new message | `/stop` or send a new message |
| Platform-specific status | `/platforms` | `/status`, `/sethome` |

For the full command lists, see the [CLI guide](https://claudia-autonomous.example.com/docs/user-guide/cli) and the [Messaging Gateway guide](https://claudia-autonomous.example.com/docs/user-guide/messaging).

---

## Documentation

All documentation lives at **[claudia-autonomous.example.com/docs](https://claudia-autonomous.example.com/docs/)**:

| Section | What's Covered |
|---------|---------------|
| [Quickstart](https://claudia-autonomous.example.com/docs/getting-started/quickstart) | Install → setup → first conversation in 2 minutes |
| [CLI Usage](https://claudia-autonomous.example.com/docs/user-guide/cli) | Commands, keybindings, personalities, sessions |
| [Configuration](https://claudia-autonomous.example.com/docs/user-guide/configuration) | Config file, providers, models, all options |
| [Messaging Gateway](https://claudia-autonomous.example.com/docs/user-guide/messaging) | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](https://claudia-autonomous.example.com/docs/user-guide/security) | Command approval, DM pairing, container isolation |
| [Tools & Toolsets](https://claudia-autonomous.example.com/docs/user-guide/features/tools) | 40+ tools, toolset system, terminal backends |
| [Skills System](https://claudia-autonomous.example.com/docs/user-guide/features/skills) | Procedural memory, Skills Hub, creating skills |
| [Memory](https://claudia-autonomous.example.com/docs/user-guide/features/memory) | Persistent memory, user profiles, best practices |
| [MCP Integration](https://claudia-autonomous.example.com/docs/user-guide/features/mcp) | Connect any MCP server for extended capabilities |
| [Cron Scheduling](https://claudia-autonomous.example.com/docs/user-guide/features/cron) | Scheduled tasks with platform delivery |
| [Context Files](https://claudia-autonomous.example.com/docs/user-guide/features/context-files) | Project context that shapes every conversation |
| [Architecture](https://claudia-autonomous.example.com/docs/developer-guide/architecture) | Project structure, agent loop, key classes |
| [Contributing](https://claudia-autonomous.example.com/docs/developer-guide/contributing) | Development setup, PR process, code style |
| [CLI Reference](https://claudia-autonomous.example.com/docs/reference/cli-commands) | All commands and flags |
| [Environment Variables](https://claudia-autonomous.example.com/docs/reference/environment-variables) | Complete env var reference |

---

## Migrating from OpenClaw

If you're coming from OpenClaw, Claudia can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`claudia setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
claudia migrate migrate              # Interactive migration (full preset)
claudia migrate migrate --dry-run    # Preview what would be migrated
claudia migrate migrate --preset user-data   # Migrate without secrets
claudia migrate migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:
- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.claudia/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `claudia migrate migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

We welcome contributions! See the [Contributing Guide](https://claudia-autonomous.example.com/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for contributors:

```bash
git clone https://github.com/kbanc85/claudia-autonomous.git
cd claudia-autonomous
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all,dev]"
python -m pytest tests/ -q
```

> **RL Training (optional):** To work on the RL/Tinker-Atropos integration:
> ```bash
> git submodule update --init tinker-atropos
> uv pip install -e "./tinker-atropos"
> ```

---

## Community

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/kbanc85/claudia-autonomous/issues)
- 💡 [Discussions](https://github.com/kbanc85/claudia-autonomous/discussions)

---

## License

MIT — see [LICENSE](LICENSE).

Built by [Kamil Banc](https://example.com).
