# Third-Party Attribution

Claudia Autonomous is a fork of **Hermes Agent v0.7.0**, released by **Nous Research** on 2026-04-03 under the MIT License.

## Upstream source

- **Repository**: https://github.com/NousResearch/hermes-agent
- **Fork commit**: `abf1e98f6253f6984479fe03d1098173a9b065a7`
- **Upstream tag**: `v2026.4.3`
- **Upstream release name**: "Hermes Agent v0.7.0"
- **License**: MIT License, Copyright (c) 2025 Nous Research

The original LICENSE file is preserved verbatim at the repo root. No license terms are altered or removed.

## What Claudia inherits from Hermes

Hermes v0.7.0 provides the entire runtime substrate that Claudia Autonomous builds on:

- The ReAct agent loop and 90-iteration conversation budget (`run_agent.py`)
- The pluggable memory provider interface (PR #4623, shipped in v0.7.0)
- ~40 built-in tools: file operations, browser automation, terminal execution, vision, TTS, STT, web search, MCP integration, and more
- Six execution backends: local, Docker, SSH, Daytona, Singularity, Modal
- Six messaging gateways: Telegram, Discord, Slack, WhatsApp, Signal, Email
- The cron scheduler with job storage and platform delivery
- The skill system compatible with [agentskills.io](https://agentskills.io) standard
- Gateway hardening (race conditions, approval routing, flood control)
- Credential pool rotation for same-provider failover
- The Camofox anti-detection browser
- Comprehensive security surfaces (dangerous command approval, pairing protocol, Tirith content scanning, secret redaction, URL exfiltration blocking)

Claudia's additions are documented in the [implementation roadmap](https://github.com/kbanc85/claudia/blob/main/autonomous/roadmap/claudia-autonomous-roadmap-v3.md).

## Relationship to upstream

Claudia Autonomous is a **permanent fork**, not a wrapper. There is no plan to contribute changes back upstream. The fork maintains its own release history, its own persona, its own memory architecture, and its own branding.

Upstream security patches will be **cherry-picked** on a case-by-case basis (tracked in `docs/decisions/upstream-cherry-picks.md` as of Phase 8). The fork does not rebase onto later Hermes releases.

## Other third-party dependencies

All runtime Python dependencies are declared in `pyproject.toml` with pinned version ranges. Each dependency carries its own license, retained in the installed wheel's metadata.

Notable transitive sources:

- **Honcho** ([plastic-labs/honcho](https://github.com/plastic-labs/honcho)) — dialectic user modelling. Apache 2.0.
- **Atropos** and **Tinker** — Nous Research and Thinking Machines research packages, only installed via the optional `[rl]` extra, not part of the default install.
- **atroposlib** — dependency of the `[rl]` extra, sourced from `NousResearch/atropos`.

The `[rl]` extra is kept for experimental RL training workflows but is not required by any part of Claudia's chief-of-staff product surface.

## Questions about attribution

If you notice missing attribution for any third-party code, file system, protocol, or idea, please open an issue at https://github.com/kbanc85/claudia-autonomous/issues. Attribution is a priority and misses will be corrected promptly.
