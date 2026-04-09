---
name: claudia-autonomous
description: Complete guide to using and extending Claudia — CLI usage, setup, configuration, spawning additional agents, gateway platforms, skills, voice, tools, profiles, and a concise contributor reference. Load this skill when helping users configure Claudia, troubleshoot issues, spawn agent instances, or make code contributions.
version: 2.0.0
author: Claudia + Teknium
license: MIT
metadata:
  claudia:
    tags: [claudia, setup, configuration, multi-agent, spawning, cli, gateway, development]
    homepage: https://github.com/NousResearch/claudia-autonomous
    related_skills: [claude-code, codex, opencode]
---

# Claudia

Claudia is an open-source AI agent framework by Nous Research that runs in your terminal, messaging platforms, and IDEs. It belongs to the same category as Claude Code (Anthropic), Codex (OpenAI), and OpenClaw — autonomous coding and task-execution agents that use tool calling to interact with your system. Claudia works with any LLM provider (OpenRouter, Anthropic, OpenAI, DeepSeek, local models, and 15+ others) and runs on Linux, macOS, and WSL.

What makes Claudia different:

- **Self-improving through skills** — Claudia learns from experience by saving reusable procedures as skills. When it solves a complex problem, discovers a workflow, or gets corrected, it can persist that knowledge as a skill document that loads into future sessions. Skills accumulate over time, making the agent better at your specific tasks and environment.
- **Persistent memory across sessions** — remembers who you are, your preferences, environment details, and lessons learned. Pluggable memory backends (built-in, Honcho, Mem0, and more) let you choose how memory works.
- **Multi-platform gateway** — the same agent runs on Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Email, and 8+ other platforms with full tool access, not just chat.
- **Provider-agnostic** — swap models and providers mid-workflow without changing anything else. Credential pools rotate across multiple API keys automatically.
- **Profiles** — run multiple independent Claudia instances with isolated configs, sessions, skills, and memory.
- **Extensible** — plugins, MCP servers, custom tools, webhook triggers, cron scheduling, and the full Python ecosystem.

People use Claudia for software development, research, system administration, data analysis, content creation, home automation, and anything else that benefits from an AI agent with persistent context and full system access.

**This skill helps you work with Claudia effectively** — setting it up, configuring features, spawning additional agent instances, troubleshooting issues, finding the right commands and settings, and understanding how the system works when you need to extend or contribute to it.

**Docs:** https://claudia-autonomous.nousresearch.com/docs/

## Quick Start

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/NousResearch/claudia-autonomous/main/scripts/install.sh | bash

# Interactive chat (default)
claudia

# Single query
claudia chat -q "What is the capital of France?"

# Setup wizard
claudia setup

# Change model/provider
claudia model

# Check health
claudia doctor
```

---

## CLI Reference

### Global Flags

```
claudia [flags] [command]

  --version, -V             Show version
  --resume, -r SESSION      Resume session by ID or title
  --continue, -c [NAME]     Resume by name, or most recent session
  --worktree, -w            Isolated git worktree mode (parallel agents)
  --skills, -s SKILL        Preload skills (comma-separate or repeat)
  --profile, -p NAME        Use a named profile
  --yolo                    Skip dangerous command approval
  --pass-session-id         Include session ID in system prompt
```

No subcommand defaults to `chat`.

### Chat

```
claudia chat [flags]
  -q, --query TEXT          Single query, non-interactive
  -m, --model MODEL         Model (e.g. anthropic/claude-sonnet-4)
  -t, --toolsets LIST       Comma-separated toolsets
  --provider PROVIDER       Force provider (openrouter, anthropic, nous, etc.)
  -v, --verbose             Verbose output
  -Q, --quiet               Suppress banner, spinner, tool previews
  --checkpoints             Enable filesystem checkpoints (/rollback)
  --source TAG              Session source tag (default: cli)
```

### Configuration

```
claudia setup [section]      Interactive wizard (model|terminal|gateway|tools|agent)
claudia model                Interactive model/provider picker
claudia config               View current config
claudia config edit          Open config.yaml in $EDITOR
claudia config set KEY VAL   Set a config value
claudia config path          Print config.yaml path
claudia config env-path      Print .env path
claudia config check         Check for missing/outdated config
claudia config migrate       Update config with new options
claudia login [--provider P] OAuth login (nous, openai-codex)
claudia logout               Clear stored auth
claudia doctor [--fix]       Check dependencies and config
claudia status [--all]       Show component status
```

### Tools & Skills

```
claudia tools                Interactive tool enable/disable (curses UI)
claudia tools list           Show all tools and status
claudia tools enable NAME    Enable a toolset
claudia tools disable NAME   Disable a toolset

claudia skills list          List installed skills
claudia skills search QUERY  Search the skills hub
claudia skills install ID    Install a skill
claudia skills inspect ID    Preview without installing
claudia skills config        Enable/disable skills per platform
claudia skills check         Check for updates
claudia skills update        Update outdated skills
claudia skills uninstall N   Remove a hub skill
claudia skills publish PATH  Publish to registry
claudia skills browse        Browse all available skills
claudia skills tap add REPO  Add a GitHub repo as skill source
```

### MCP Servers

```
claudia mcp serve            Run Claudia as an MCP server
claudia mcp add NAME         Add an MCP server (--url or --command)
claudia mcp remove NAME      Remove an MCP server
claudia mcp list             List configured servers
claudia mcp test NAME        Test connection
claudia mcp configure NAME   Toggle tool selection
```

### Gateway (Messaging Platforms)

```
claudia gateway run          Start gateway foreground
claudia gateway install      Install as background service
claudia gateway start/stop   Control the service
claudia gateway restart      Restart the service
claudia gateway status       Check status
claudia gateway setup        Configure platforms
```

Supported platforms: Telegram, Discord, Slack, WhatsApp, Signal, Email, SMS, Matrix, Mattermost, Home Assistant, DingTalk, Feishu, WeCom, API Server, Webhooks, Open WebUI.

Platform docs: https://claudia-autonomous.nousresearch.com/docs/user-guide/messaging/

### Sessions

```
claudia sessions list        List recent sessions
claudia sessions browse      Interactive picker
claudia sessions export OUT  Export to JSONL
claudia sessions rename ID T Rename a session
claudia sessions delete ID   Delete a session
claudia sessions prune       Clean up old sessions (--older-than N days)
claudia sessions stats       Session store statistics
```

### Cron Jobs

```
claudia cron list            List jobs (--all for disabled)
claudia cron create SCHED    Create: '30m', 'every 2h', '0 9 * * *'
claudia cron edit ID         Edit schedule, prompt, delivery
claudia cron pause/resume ID Control job state
claudia cron run ID          Trigger on next tick
claudia cron remove ID       Delete a job
claudia cron status          Scheduler status
```

### Webhooks

```
claudia webhook subscribe N  Create route at /webhooks/<name>
claudia webhook list         List subscriptions
claudia webhook remove NAME  Remove a subscription
claudia webhook test NAME    Send a test POST
```

### Profiles

```
claudia profile list         List all profiles
claudia profile create NAME  Create (--clone, --clone-all, --clone-from)
claudia profile use NAME     Set sticky default
claudia profile delete NAME  Delete a profile
claudia profile show NAME    Show details
claudia profile alias NAME   Manage wrapper scripts
claudia profile rename A B   Rename a profile
claudia profile export NAME  Export to tar.gz
claudia profile import FILE  Import from archive
```

### Credential Pools

```
claudia auth add             Interactive credential wizard
claudia auth list [PROVIDER] List pooled credentials
claudia auth remove P INDEX  Remove by provider + index
claudia auth reset PROVIDER  Clear exhaustion status
```

### Other

```
claudia insights [--days N]  Usage analytics
claudia update               Update to latest version
claudia pairing list/approve/revoke  DM authorization
claudia plugins list/install/remove  Plugin management
claudia honcho setup/status  Honcho memory integration
claudia memory setup/status/off  Memory provider config
claudia completion bash|zsh  Shell completions
claudia acp                  ACP server (IDE integration)
claudia migrate migrate         Migrate from OpenClaw
claudia uninstall            Uninstall Claudia
```

---

## Slash Commands (In-Session)

Type these during an interactive chat session.

### Session Control
```
/new (/reset)        Fresh session
/clear               Clear screen + new session (CLI)
/retry               Resend last message
/undo                Remove last exchange
/title [name]        Name the session
/compress            Manually compress context
/stop                Kill background processes
/rollback [N]        Restore filesystem checkpoint
/background <prompt> Run prompt in background
/queue <prompt>      Queue for next turn
/resume [name]       Resume a named session
```

### Configuration
```
/config              Show config (CLI)
/model [name]        Show or change model
/provider            Show provider info
/prompt [text]       View/set system prompt (CLI)
/personality [name]  Set personality
/reasoning [level]   Set reasoning (none|low|medium|high|xhigh|show|hide)
/verbose             Cycle: off → new → all → verbose
/voice [on|off|tts]  Voice mode
/yolo                Toggle approval bypass
/skin [name]         Change theme (CLI)
/statusbar           Toggle status bar (CLI)
```

### Tools & Skills
```
/tools               Manage tools (CLI)
/toolsets            List toolsets (CLI)
/skills              Search/install skills (CLI)
/skill <name>        Load a skill into session
/cron                Manage cron jobs (CLI)
/reload-mcp          Reload MCP servers
/plugins             List plugins (CLI)
```

### Info
```
/help                Show commands
/commands [page]     Browse all commands (gateway)
/usage               Token usage
/insights [days]     Usage analytics
/status              Session info (gateway)
/profile             Active profile info
```

### Exit
```
/quit (/exit, /q)    Exit CLI
```

---

## Key Paths & Config

```
~/.claudia/config.yaml       Main configuration
~/.claudia/.env              API keys and secrets
~/.claudia/skills/           Installed skills
~/.claudia/sessions/         Session transcripts
~/.claudia/logs/             Gateway and error logs
~/.claudia/auth.json         OAuth tokens and credential pools
~/.claudia/claudia-autonomous/     Source code (if git-installed)
```

Profiles use `~/.claudia/profiles/<name>/` with the same layout.

### Config Sections

Edit with `claudia config edit` or `claudia config set section.key value`.

| Section | Key options |
|---------|-------------|
| `model` | `default`, `provider`, `base_url`, `api_key`, `context_length` |
| `agent` | `max_turns` (90), `tool_use_enforcement` |
| `terminal` | `backend` (local/docker/ssh/modal), `cwd`, `timeout` (180) |
| `compression` | `enabled`, `threshold` (0.50), `target_ratio` (0.20) |
| `display` | `skin`, `tool_progress`, `show_reasoning`, `show_cost` |
| `stt` | `enabled`, `provider` (local/groq/openai) |
| `tts` | `provider` (edge/elevenlabs/openai/kokoro/fish) |
| `memory` | `memory_enabled`, `user_profile_enabled`, `provider` |
| `security` | `tirith_enabled`, `website_blocklist` |
| `delegation` | `model`, `provider`, `max_iterations` (50) |
| `smart_model_routing` | `enabled`, `cheap_model` |
| `checkpoints` | `enabled`, `max_snapshots` (50) |

Full config reference: https://claudia-autonomous.nousresearch.com/docs/user-guide/configuration

### Providers

18 providers supported. Set via `claudia model` or `claudia setup`.

| Provider | Auth | Key env var |
|----------|------|-------------|
| OpenRouter | API key | `OPENROUTER_API_KEY` |
| Anthropic | API key | `ANTHROPIC_API_KEY` |
| Nous Portal | OAuth | `claudia login --provider nous` |
| OpenAI Codex | OAuth | `claudia login --provider openai-codex` |
| GitHub Copilot | Token | `COPILOT_GITHUB_TOKEN` |
| DeepSeek | API key | `DEEPSEEK_API_KEY` |
| Hugging Face | Token | `HF_TOKEN` |
| Z.AI / GLM | API key | `GLM_API_KEY` |
| MiniMax | API key | `MINIMAX_API_KEY` |
| Kimi / Moonshot | API key | `KIMI_API_KEY` |
| Alibaba / DashScope | API key | `DASHSCOPE_API_KEY` |
| Kilo Code | API key | `KILOCODE_API_KEY` |
| Custom endpoint | Config | `model.base_url` + `model.api_key` in config.yaml |

Plus: AI Gateway, OpenCode Zen, OpenCode Go, MiniMax CN, GitHub Copilot ACP.

Full provider docs: https://claudia-autonomous.nousresearch.com/docs/integrations/providers

### Toolsets

Enable/disable via `claudia tools` (interactive) or `claudia tools enable/disable NAME`.

| Toolset | What it provides |
|---------|-----------------|
| `web` | Web search and content extraction |
| `browser` | Browser automation (Browserbase, Camofox, or local Chromium) |
| `terminal` | Shell commands and process management |
| `file` | File read/write/search/patch |
| `code_execution` | Sandboxed Python execution |
| `vision` | Image analysis |
| `image_gen` | AI image generation |
| `tts` | Text-to-speech |
| `skills` | Skill browsing and management |
| `memory` | Persistent cross-session memory |
| `session_search` | Search past conversations |
| `delegation` | Subagent task delegation |
| `cronjob` | Scheduled task management |
| `clarify` | Ask user clarifying questions |
| `moa` | Mixture of Agents (off by default) |
| `homeassistant` | Smart home control (off by default) |

Tool changes take effect on `/reset` (new session). They do NOT apply mid-conversation to preserve prompt caching.

---

## Voice & Transcription

### STT (Voice → Text)

Voice messages from messaging platforms are auto-transcribed.

Provider priority (auto-detected):
1. **Local faster-whisper** — free, no API key: `pip install faster-whisper`
2. **Groq Whisper** — free tier: set `GROQ_API_KEY`
3. **OpenAI Whisper** — paid: set `VOICE_TOOLS_OPENAI_KEY`

Config:
```yaml
stt:
  enabled: true
  provider: local        # local, groq, openai
  local:
    model: base          # tiny, base, small, medium, large-v3
```

### TTS (Text → Voice)

| Provider | Env var | Free? |
|----------|---------|-------|
| Edge TTS | None | Yes (default) |
| ElevenLabs | `ELEVENLABS_API_KEY` | Free tier |
| OpenAI | `VOICE_TOOLS_OPENAI_KEY` | Paid |
| Kokoro (local) | None | Free |
| Fish Audio | `FISH_AUDIO_API_KEY` | Free tier |

Voice commands: `/voice on` (voice-to-voice), `/voice tts` (always voice), `/voice off`.

---

## Spawning Additional Claudia Instances

Run additional Claudia processes as fully independent subprocesses — separate sessions, tools, and environments.

### When to Use This vs delegate_task

| | `delegate_task` | Spawning `claudia` process |
|-|-----------------|--------------------------|
| Isolation | Separate conversation, shared process | Fully independent process |
| Duration | Minutes (bounded by parent loop) | Hours/days |
| Tool access | Subset of parent's tools | Full tool access |
| Interactive | No | Yes (PTY mode) |
| Use case | Quick parallel subtasks | Long autonomous missions |

### One-Shot Mode

```
terminal(command="claudia chat -q 'Research GRPO papers and write summary to ~/research/grpo.md'", timeout=300)

# Background for long tasks:
terminal(command="claudia chat -q 'Set up CI/CD for ~/myapp'", background=true)
```

### Interactive PTY Mode (via tmux)

Claudia uses prompt_toolkit, which requires a real terminal. Use tmux for interactive spawning:

```
# Start
terminal(command="tmux new-session -d -s agent1 -x 120 -y 40 'claudia'", timeout=10)

# Wait for startup, then send a message
terminal(command="sleep 8 && tmux send-keys -t agent1 'Build a FastAPI auth service' Enter", timeout=15)

# Read output
terminal(command="sleep 20 && tmux capture-pane -t agent1 -p", timeout=5)

# Send follow-up
terminal(command="tmux send-keys -t agent1 'Add rate limiting middleware' Enter", timeout=5)

# Exit
terminal(command="tmux send-keys -t agent1 '/exit' Enter && sleep 2 && tmux kill-session -t agent1", timeout=10)
```

### Multi-Agent Coordination

```
# Agent A: backend
terminal(command="tmux new-session -d -s backend -x 120 -y 40 'claudia -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t backend 'Build REST API for user management' Enter", timeout=15)

# Agent B: frontend
terminal(command="tmux new-session -d -s frontend -x 120 -y 40 'claudia -w'", timeout=10)
terminal(command="sleep 8 && tmux send-keys -t frontend 'Build React dashboard for user management' Enter", timeout=15)

# Check progress, relay context between them
terminal(command="tmux capture-pane -t backend -p | tail -30", timeout=5)
terminal(command="tmux send-keys -t frontend 'Here is the API schema from the backend agent: ...' Enter", timeout=5)
```

### Session Resume

```
# Resume most recent session
terminal(command="tmux new-session -d -s resumed 'claudia --continue'", timeout=10)

# Resume specific session
terminal(command="tmux new-session -d -s resumed 'claudia --resume 20260225_143052_a1b2c3'", timeout=10)
```

### Tips

- **Prefer `delegate_task` for quick subtasks** — less overhead than spawning a full process
- **Use `-w` (worktree mode)** when spawning agents that edit code — prevents git conflicts
- **Set timeouts** for one-shot mode — complex tasks can take 5-10 minutes
- **Use `claudia chat -q` for fire-and-forget** — no PTY needed
- **Use tmux for interactive sessions** — raw PTY mode has `\r` vs `\n` issues with prompt_toolkit
- **For scheduled tasks**, use the `cronjob` tool instead of spawning — handles delivery and retry

---

## Troubleshooting

### Voice not working
1. Check `stt.enabled: true` in config.yaml
2. Verify provider: `pip install faster-whisper` or set API key
3. Restart gateway: `/restart`

### Tool not available
1. `claudia tools` — check if toolset is enabled for your platform
2. Some tools need env vars (check `.env`)
3. `/reset` after enabling tools

### Model/provider issues
1. `claudia doctor` — check config and dependencies
2. `claudia login` — re-authenticate OAuth providers
3. Check `.env` has the right API key

### Changes not taking effect
- **Tools/skills:** `/reset` starts a new session with updated toolset
- **Config changes:** `/restart` reloads gateway config
- **Code changes:** Restart the CLI or gateway process

### Skills not showing
1. `claudia skills list` — verify installed
2. `claudia skills config` — check platform enablement
3. Load explicitly: `/skill name` or `claudia -s name`

### Gateway issues
Check logs first:
```bash
grep -i "failed to send\|error" ~/.claudia/logs/gateway.log | tail -20
```

---

## Where to Find Things

| Looking for... | Location |
|----------------|----------|
| Config options | `claudia config edit` or [Configuration docs](https://claudia-autonomous.nousresearch.com/docs/user-guide/configuration) |
| Available tools | `claudia tools list` or [Tools reference](https://claudia-autonomous.nousresearch.com/docs/reference/tools-reference) |
| Slash commands | `/help` in session or [Slash commands reference](https://claudia-autonomous.nousresearch.com/docs/reference/slash-commands) |
| Skills catalog | `claudia skills browse` or [Skills catalog](https://claudia-autonomous.nousresearch.com/docs/reference/skills-catalog) |
| Provider setup | `claudia model` or [Providers guide](https://claudia-autonomous.nousresearch.com/docs/integrations/providers) |
| Platform setup | `claudia gateway setup` or [Messaging docs](https://claudia-autonomous.nousresearch.com/docs/user-guide/messaging/) |
| MCP servers | `claudia mcp list` or [MCP guide](https://claudia-autonomous.nousresearch.com/docs/user-guide/features/mcp) |
| Profiles | `claudia profile list` or [Profiles docs](https://claudia-autonomous.nousresearch.com/docs/user-guide/profiles) |
| Cron jobs | `claudia cron list` or [Cron docs](https://claudia-autonomous.nousresearch.com/docs/user-guide/features/cron) |
| Memory | `claudia memory status` or [Memory docs](https://claudia-autonomous.nousresearch.com/docs/user-guide/features/memory) |
| Env variables | `claudia config env-path` or [Env vars reference](https://claudia-autonomous.nousresearch.com/docs/reference/environment-variables) |
| CLI commands | `claudia --help` or [CLI reference](https://claudia-autonomous.nousresearch.com/docs/reference/cli-commands) |
| Gateway logs | `~/.claudia/logs/gateway.log` |
| Session files | `~/.claudia/sessions/` or `claudia sessions browse` |
| Source code | `~/.claudia/claudia-autonomous/` |

---

## Contributor Quick Reference

For occasional contributors and PR authors. Full developer docs: https://claudia-autonomous.nousresearch.com/docs/developer-guide/

### Project Layout

```
claudia-autonomous/
├── run_agent.py          # AIAgent — core conversation loop
├── model_tools.py        # Tool discovery and dispatch
├── toolsets.py           # Toolset definitions
├── cli.py                # Interactive CLI (ClaudiaCLI)
├── claudia_state.py       # SQLite session store
├── agent/                # Prompt builder, compression, display, adapters
├── claudia_cli/           # CLI subcommands, config, setup, commands
│   ├── commands.py       # Slash command registry (CommandDef)
│   ├── config.py         # DEFAULT_CONFIG, env var definitions
│   └── main.py           # CLI entry point and argparse
├── tools/                # One file per tool
│   └── registry.py       # Central tool registry
├── gateway/              # Messaging gateway
│   └── platforms/        # Platform adapters (telegram, discord, etc.)
├── cron/                 # Job scheduler
├── tests/                # ~3000 pytest tests
└── website/              # Docusaurus docs site
```

Config: `~/.claudia/config.yaml` (settings), `~/.claudia/.env` (API keys).

### Adding a Tool (3 files)

**1. Create `tools/your_tool.py`:**
```python
import json, os
from tools.registry import registry

def check_requirements() -> bool:
    return bool(os.getenv("EXAMPLE_API_KEY"))

def example_tool(param: str, task_id: str = None) -> str:
    return json.dumps({"success": True, "data": "..."})

registry.register(
    name="example_tool",
    toolset="example",
    schema={"name": "example_tool", "description": "...", "parameters": {...}},
    handler=lambda args, **kw: example_tool(
        param=args.get("param", ""), task_id=kw.get("task_id")),
    check_fn=check_requirements,
    requires_env=["EXAMPLE_API_KEY"],
)
```

**2. Add import** in `model_tools.py` → `_discover_tools()` list.

**3. Add to `toolsets.py`** → `_CLAUDIA_CORE_TOOLS` list.

All handlers must return JSON strings. Use `get_claudia_home()` for paths, never hardcode `~/.claudia`.

### Adding a Slash Command

1. Add `CommandDef` to `COMMAND_REGISTRY` in `claudia_cli/commands.py`
2. Add handler in `cli.py` → `process_command()`
3. (Optional) Add gateway handler in `gateway/run.py`

All consumers (help text, autocomplete, Telegram menu, Slack mapping) derive from the central registry automatically.

### Agent Loop (High Level)

```
run_conversation():
  1. Build system prompt
  2. Loop while iterations < max:
     a. Call LLM (OpenAI-format messages + tool schemas)
     b. If tool_calls → dispatch each via handle_function_call() → append results → continue
     c. If text response → return
  3. Context compression triggers automatically near token limit
```

### Testing

```bash
source venv/bin/activate  # or .venv/bin/activate
python -m pytest tests/ -o 'addopts=' -q   # Full suite
python -m pytest tests/tools/ -q            # Specific area
```

- Tests auto-redirect `CLAUDIA_HOME` to temp dirs — never touch real `~/.claudia/`
- Run full suite before pushing any change
- Use `-o 'addopts='` to clear any baked-in pytest flags

### Commit Conventions

```
type: concise subject line

Optional body.
```

Types: `fix:`, `feat:`, `refactor:`, `docs:`, `chore:`

### Key Rules

- **Never break prompt caching** — don't change context, tools, or system prompt mid-conversation
- **Message role alternation** — never two assistant or two user messages in a row
- Use `get_claudia_home()` from `claudia_constants` for all paths (profile-safe)
- Config values go in `config.yaml`, secrets go in `.env`
- New tools need a `check_fn` so they only appear when requirements are met
