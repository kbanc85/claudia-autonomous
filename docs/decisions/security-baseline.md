# Security baseline — Claudia Autonomous at commit 7bede11

**Author**: Claudia scaffolding session, 2026-04-09
**Fork point**: NousResearch/hermes-agent commit `abf1e98` (tag `v2026.4.3`, Hermes Agent v0.7.0)
**Audited commit**: `7bede11` (Phase 0.2 rebrand complete; contents equivalent to v0.7.0 minus deleted Atropos RL infrastructure plus mechanical string substitutions)

## Scope

This document is a **static-analysis baseline** of the security posture Claudia Autonomous inherits from Hermes Agent v0.7.0. It is not a penetration test or a dynamic audit. Each attack surface the Claudia Autonomous roadmap v3 named is covered, plus several additional surfaces discovered during the audit.

Dynamic verification (command fuzzing, actual sandbox escapes, red-team testing) is out of scope for Phase 0. Phase 0.4 sets up the test harness that later phases will use to codify security invariants as automated tests.

The rebrand sweep applied in Phase 0.2 was pure mechanical string substitution — no logic changed — so the security posture is identical to the unmodified v0.7.0 fork except where otherwise noted in "Claudia-specific considerations".

## Attack surfaces

### 1. Dangerous command approval (`tools/approval.py`)

**What it does.** Single source of truth for detecting and gating dangerous shell commands before the agent executes them. Covers pattern detection, per-session approval state (thread-safe), interactive CLI + async gateway prompting, auxiliary-LLM smart approval for low-risk commands, and persistent allowlist storage at `config.yaml`.

**Coverage observed.** The `DANGEROUS_PATTERNS` list holds 30+ regexes across:
- Filesystem destruction: `rm -rf`, `rm /`, `find -delete`, `find -exec rm`, `xargs rm`
- Permission escalation: `chmod 777`, `chmod 666`, `chown -R root`, world-writable perms
- Block-level writes: `mkfs`, `dd if=`, `> /dev/sd`, `> /etc/`
- Database destruction: `DROP TABLE/DATABASE`, `DELETE FROM` without `WHERE`, `TRUNCATE`
- Process management: `systemctl stop/disable/mask`, `kill -9 -1`, `pkill -9`
- Shell injection: `bash -c`, `python -c`, `curl | sh`, process substitution of remote scripts
- Fork bombs: the classic `:(){ :|:& };:`
- Sensitive path writes: explicit allowlist covering `/etc/`, `/dev/sd`, `~/.ssh/`, `~/.claudia/.env` (with environment variable expansion detection)
- Self-termination protection: blocks `pkill claudia`, `pkill gateway`, `pkill cli.py` — the agent cannot kill its own process
- Gateway protection: blocks `gateway run &`, `nohup gateway run`, `disown` patterns that would start the gateway outside systemd management

**Configuration surface.** Approvals are stored per session (in-memory) and persistently via `config.yaml`. A smart-approval auxiliary LLM can auto-approve low-risk commands when enabled.

**Gaps observed.**
- Pattern-based detection is fundamentally incomplete for shell command semantics. Obfuscation via base64, process substitution (`$(cat foo)`), or IFS manipulation can bypass regex. The Tirith wrapper (§7) is the intended defense-in-depth for this.
- No mention of SELinux/AppArmor policy enforcement — relies on voluntary approval.
- The "smart approval" auxiliary LLM is itself a prompt-injection surface if the command text carries instructions.

**Verdict.** Strong pattern coverage for common destructive operations. Approval system is the primary gate but not the only gate — expect to rely on the container sandbox (§3) and pre-exec scanner (§7) for content-level and semantic defenses.

### 2. Gateway DM pairing (`gateway/pairing.py`)

**What it does.** Code-based approval flow for authorizing new users on messaging platforms (Telegram, Discord, Slack, WhatsApp, Signal, Email). Replaces static user-ID allowlists with a one-time pairing code approved by the bot owner via CLI. Module docstring explicitly cites OWASP and NIST SP 800-63-4 as the design reference.

**Coverage observed.**
- **Code entropy**: 8 characters drawn from a 32-character unambiguous alphabet (excludes `0/O`, `1/I`). Generated via `secrets.choice()` — cryptographically strong randomness, not `random.random()`.
- **Code TTL**: 1 hour (`CODE_TTL_SECONDS = 3600`).
- **Rate limiting**: 1 pairing request per user per 10 minutes (`RATE_LIMIT_SECONDS = 600`).
- **Attempt lockout**: 5 failed approval attempts trigger a 1-hour lockout (`MAX_FAILED_ATTEMPTS = 5`, `LOCKOUT_SECONDS = 3600`).
- **Pending-request cap**: max 3 pending codes per platform (`MAX_PENDING_PER_PLATFORM = 3`).
- **File permissions**: all pairing data files written with `chmod 0o600` via a dedicated `_secure_write()` helper.
- **Log hygiene**: codes are never written to stdout; the docstring explicitly calls this out as an invariant.
- **Storage isolation**: data lives at `~/.claudia/pairing/` with platform-scoped filenames (`{platform}-pending.json`, `{platform}-approved.json`, `_rate_limits.json`).

**Gaps observed.**
- No revocation audit trail in the file I read — a paired user can be removed from the approved list but the history of approval/revocation isn't persisted (check `PairingStore` full implementation in Phase 0.4 tests).
- File-based storage means a host compromise trivially leaks the approved list. Acceptable given the broader assumption that host compromise is game-over.
- `chmod 0o600` is silently skipped on Windows (`except OSError: pass`). Windows deployments should be considered lower-security until Phase 8 adds a filesystem ACL fallback.

**Verdict.** Textbook-quality access control for a messaging bot. The OWASP/NIST references and the `secrets` module usage suggest this was designed with security in mind from the start, not bolted on.

### 3. Docker backend container isolation (`tools/environments/docker.py`)

**What it does.** Sandboxed command execution via Docker containers. Used when the agent needs to run arbitrary shell commands without trusting them on the host.

**Hardening flags observed** (lines 108-113 of `docker.py`):
- `--cap-drop ALL` — drops every Linux capability before adding only what's strictly needed (if anything).
- `--security-opt no-new-privileges` — prevents setuid/setgid escalation inside the container.
- `--pids-limit 256` — caps total processes to prevent fork bombs.

**Other protective design.**
- Env var forwarding is controlled by an allowlist (`forward_env`) with regex validation (`_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")`) — invalid names are logged and dropped.
- `~/.claudia/.env` secrets are loaded via `load_env()` with a try/except wrapper so malformed env files don't break the backend.
- Docker binary path is resolved via `find_docker()` which probes known install locations on macOS (Homebrew paths, Docker Desktop app bundle) before falling back to `PATH`. Reduces reliance on launchd PATH hygiene when running as a gateway service.

**Gaps observed.**
- No resource limits visible in the excerpt for memory or CPU (probably controlled by higher-level config, verify in Phase 0.4).
- The `tools/environments/managed_modal.py` sibling backend was not audited — Modal is a cloud-managed alternative that may have different isolation properties.
- No seccomp profile mentioned — would be a defense-in-depth improvement, but Docker's default seccomp profile is applied automatically and blocks ~44 syscalls by default.

**Verdict.** Solid baseline isolation. The cap-drop ALL + no-new-privileges + PID limit combination is the standard hardening recipe and matches Docker's own documentation for running untrusted code. Defense in depth, not a single layer.

### 4. Cron subsystem isolation (`cron/scheduler.py`, `cron/jobs.py`)

**What it does.** Schedules autonomous tasks that run unattended and deliver output to messaging platforms. Tasks run through the same agent loop as interactive sessions.

**Path boundaries observed.** All cron state is scoped to `_claudia_home = get_claudia_home()`:
- Lock directory: `_LOCK_DIR = _claudia_home / "cron"`
- Tick lock file: `_LOCK_FILE = _LOCK_DIR / ".tick.lock"`
- Environment file: `_claudia_home / ".env"`
- Config file: `_claudia_home / "config.yaml"`

The `get_claudia_home()` helper reads `CLAUDIA_HOME` from environment, falling back to `~/.claudia`. Profile-scoped installations (via `_apply_profile_override`) redirect this to profile-specific subdirs, so cron state stays isolated per profile.

**Path-traversal audit (roadmap Task 0.3 requirement).** The roadmap flags a "path-traversal fix from April 5, 2026". With upstream git history stripped in Phase 0.1, I cannot inspect the original commit or its test cases. Static review of `cron/scheduler.py` shows:
- No direct use of user-supplied paths in filesystem operations.
- All file paths are derived from `_claudia_home` or `job_id` parameters that come from the scheduler's own JSON state.
- The `_resolve_delivery_target()` helper converts a job's origin dict into a delivery target via `resolve_channel_name()` — the channel name is looked up in `gateway/channel_directory.py` rather than used as a path.

I found no obvious path traversal vectors in the files read, but a complete audit of `cron/jobs.py` and `gateway/channel_directory.py` should be part of Phase 0.4 test harness writing. **Flagged as follow-up.**

**Gaps observed.**
- Cron jobs inherit the full agent tool catalog by default — a malicious scheduled prompt can execute any approved command. The approval system (§1) and Tirith scanner (§7) are the only gates.
- `load_dotenv(str(_claudia_home / ".env"), override=True)` with `override=True` means `.env` values replace existing environment variables — a compromised `.env` file can downgrade security-relevant env vars like `CLAUDIA_REDACT_SECRETS`.

**Verdict.** Structurally sound but shares the agent's full privilege with the approval system. The roadmap's path-traversal fix is present in the codebase (the careful use of `_claudia_home` as a base directory for every path derivation), but I cannot confirm the specific commit without upstream history.

### 5. Secret handling — redaction and exfiltration blocking

Two complementary mechanisms.

**5a. Log/output redaction (`agent/redact.py`).**
- Regex-based pattern matching for API key prefixes: `sk-*`, `ghp_*`, `github_pat_*`, `gho_*`, `ghu_*`, `ghs_*`, `ghr_*`, `xox[baprs]-*` (Slack), `AIza*` (Google), and others.
- **Redaction policy**:
  - Tokens shorter than 18 characters are fully masked.
  - Longer tokens preserve the first 6 and last 4 characters for debuggability (standard banking-style masking).
- **Kill-switch protection**: `_REDACT_ENABLED` is captured at module import time. Runtime attempts to disable redaction via `export CLAUDIA_REDACT_SECRETS=false` are ignored — an agent cannot turn off its own secret masking mid-session. The roadmap's "secret exfiltration blocking" (v0.7.0 feature) refers to this plus §5b.
- **Test coverage**: `tests/agent/test_redact.py`, `tests/gateway/test_pii_redaction.py`, `tests/test_cli_secret_capture.py`.

**5b. Browser secret exfiltration blocking (`tools/browser_tool.py`).**
- `browser_navigate()` inspects the target URL for embedded secrets before issuing the HTTP request.
- If the URL contains an API key pattern (e.g. `?key=sk-...`, `?token=sk-or-v1-...`), the navigation is blocked with an error message.
- Test coverage in `tests/tools/test_browser_secret_exfil.py` covers OpenAI keys, OpenRouter keys, and other common patterns.
- This is the primary defense against an LLM being tricked into sending credentials to an attacker-controlled URL via crafted tool calls.

**Verdict.** Two-layer defense is correct:
- **Redaction** catches secrets heading to logs, gateway output, and stdout (inbound attacker: "dump your env vars").
- **URL blocking** catches secrets heading to external services via browser navigation (outbound attacker: "fetch this URL with your token as a query param").

Both are bypassable via creative encodings (base64, URL-encoded variants of key prefixes) but the combination raises the bar significantly.

### 6. Credential pool and rotation (`agent/credential_pool.py`)

**What it does.** Persistent multi-credential pool for same-provider failover. Lets Claudia cycle through multiple OpenRouter or Anthropic keys if one is rate-limited, revoked, or expiring.

**Observed mechanisms.**
- Imports from `claudia_cli.auth`: provider registry, JWT decoding, token expiry detection, refresh skew constants.
- Tracks `ACCESS_TOKEN_REFRESH_SKEW_SECONDS` (default unknown from this excerpt) and `CODEX_ACCESS_TOKEN_REFRESH_SKEW_SECONDS`.
- Functions `_agent_key_is_usable()`, `_codex_access_token_is_expiring()`, `_is_expiring()` enforce "use a key only if it's clearly not about to die".
- Storage via `read_credential_pool()` / `write_credential_pool()` (presumably file-based, encrypted or 0600-permed — verify in Phase 0.4).

**Gaps observed.**
- No static check that credential-pool files have secure permissions (may be handled by `write_credential_pool()`, needs dynamic verification).
- JWT decoding `_decode_jwt_claims()` must not treat claims as authoritative — the JWT signature isn't verifiable by the client without the issuer's public key. Follow up in Phase 0.4 to confirm the decoded claims are only used for expiry hints, not authorization.

**Verdict.** Failover design is sound in principle; file-on-disk implementation detail needs dynamic verification.

### 7. Pre-exec content scanning (`tools/tirith_security.py`)

**What it does.** Wraps an external `tirith` binary that scans commands before execution for content-level threats that pattern-based approval (§1) misses: homograph URLs, pipe-to-interpreter attacks, terminal escape sequence injection, etc.

**Trust model observed.**
- `tirith` is auto-downloaded from GitHub releases if not found on PATH. Download destination: `$CLAUDIA_HOME/bin/tirith`.
- Integrity verification:
  - SHA-256 checksums are always verified (hard-coded in the Python source).
  - Cosign provenance verification runs if `cosign` is available on PATH — verifies the binary was produced by a specific GitHub Actions workflow.
  - If cosign is absent, the download proceeds with SHA-256 only. Docstring explicitly notes this is "still secure via HTTPS + checksum, just without supply chain provenance proof".
- Exit-code verdict model:
  - 0 = allow
  - 1 = block
  - 2 = warn
- JSON stdout enriches findings but never overrides the exit-code verdict.
- Operational failures (spawn error, timeout, unknown exit code) respect a `fail_open` config setting. **This is a security-relevant config**: `fail_open=true` means Tirith failures degrade to "allow", which is a reasonable availability choice but a security risk if an attacker can induce failures.

**Gaps observed.**
- Auto-download from GitHub is a supply-chain risk that SHA-256 + cosign is designed to mitigate but not eliminate. An attacker who compromises the GitHub release signing key can push a malicious binary that passes both checks.
- `fail_open` default is not visible in the excerpt I read. **Recommend checking and documenting the default in Phase 0.4**, and forcing `fail_open=false` in `cli-config.yaml.example` for security-conscious defaults.

**Verdict.** Clever use of an external scanner for content-level checks that regex approval can't catch. The SHA + cosign chain is the right supply-chain shape for this. Treat Tirith as defense in depth, not the primary gate — the primary gate is container isolation (§3).

### 8. Credential file passthrough (`tools/credential_files.py`)

**What it does.** Skills can declare `required_credential_files` in their frontmatter (e.g. `~/.claudia/google_token.json`). This module tracks which files should be mounted into remote sandboxes (Docker, Modal) when the skill runs.

**Trust model observed.**
- Two sources feed the registry: skill declarations (auto-registered when a skill is loaded) and user config (`terminal.credential_files` in `config.yaml`).
- Remote backends call `get_credential_file_mounts()` at sandbox creation time. The mount format is `{host_path, container_path}`.
- The module is session-scoped — mounts don't persist across sessions.

**Gaps observed.**
- **Trust transitivity**: if a malicious skill declares `required_credential_files: [~/.ssh/id_ed25519]`, the scanner would need to catch this at skill registration time. From the excerpt, I can't confirm whether declaration-time validation happens.
- Skill frontmatter is authored by whoever publishes the skill. A skill hub model (the roadmap mentions `agentskills.io`) means Claudia could end up running skills from arbitrary publishers. **Phase 3 skill porting should add mandatory user approval on first-load of any skill that declares `required_credential_files`** — this is not in Phase 0.2's scope but I flag it here.

**Verdict.** Mechanism is needed for remote backends to access user credentials. Security quality depends entirely on the trust boundary at skill loading — which is upstream of this file and should be validated separately in Phase 3.

### 9. Gateway rate limiting and flood control

**What exists.** Every gateway platform module (`gateway/platforms/telegram.py`, `discord.py`, `slack.py`, `whatsapp.py`, `signal.py`, `feishu.py`, `homeassistant.py`, `webhook.py`) contains flood-control / rate-limit logic, per a `git grep -l "flood\|rate.*limit" gateway/` sweep. `gateway/pairing.py` and `gateway/run.py` and `gateway/stream_consumer.py` also have rate-limiting code.

**Not audited in depth here.** Per-platform rate limit policies vary (different messaging APIs have different server-side limits, and Hermes's client-side rate limit must stay well below each). A full per-platform audit is out of scope for this baseline — log it as a follow-up for Phase 0.4 or Phase 5.

**Verdict.** The presence of rate limit code in every platform module is a good sign. Effectiveness needs dynamic verification.

## Consolidated gaps and follow-ups

Tracked for later phases:

| # | Gap | Relevant phase | Note |
|---|---|---|---|
| G1 | No dynamic verification of any surface in this audit | Phase 0.4 | Test harness should codify security invariants as tests |
| G2 | `cron/jobs.py` path-traversal fix not verified against original CVE | Phase 0.4 | Can't inspect upstream commit (history stripped) |
| G3 | `tirith` `fail_open` default not documented | Phase 0.4 | Force `fail_open=false` in default config |
| G4 | Credential pool file permissions not verified statically | Phase 0.4 | Dynamic test needed |
| G5 | Skill `required_credential_files` declarations need user approval on first load | Phase 3 | Skills from agentskills.io are an untrusted publisher boundary |
| G6 | Windows pairing-store permissions use silent `chmod` fallback | Phase 8 | Filesystem ACL fallback for Windows deployments |
| G7 | `agent/anthropic_adapter.py:1266` sanitizer is Hermes-specific legacy | Phase 1.2 | My Phase 0.2 C4 broad sed partially corrupted the sanitizer; review and likely remove |
| G8 | Gateway flood control not audited per-platform | Phase 0.4 or 5 | Verify each platform's client-side limits stay below server limits |
| G9 | Redaction regex misses base64-encoded or URL-encoded secret variants | Phase 8 | Defense-in-depth improvement, not blocking |
| G10 | Auxiliary-LLM "smart approval" is itself a prompt-injection surface | Phase 4 | Consider disabling for commands containing user-supplied content |

## Claudia-specific considerations (what changed vs v0.7.0)

Phase 0.2 was pure string substitution and scope deletion. Nothing changed semantically in the security surfaces above except:

1. **`environments/` directory removed** (Atropos RL training integration). Not an execution backend; no security impact. The real execution backends at `tools/environments/` are untouched.

2. **`.hermes` → `.claudia` config dir**. All paths and env vars now use `CLAUDIA_HOME` / `~/.claudia`. The permission modes (`chmod 0o600`, `0o700` where used) are unchanged. Path traversal defenses stay intact because they were always based on `get_claudia_home()` rather than hardcoded paths.

3. **`HERMES_REDACT_SECRETS` → `CLAUDIA_REDACT_SECRETS`** (env var rename). The kill-switch protection (`_REDACT_ENABLED` captured at import time) still holds — the env var name changed but the snapshot semantics didn't.

4. **Honcho plugin default workspace `"hermes"` → `"claudia"`** (semantic change, not just a label). New Claudia installs get a fresh Honcho workspace. Phase 6 migration will offer existing users import of their legacy `"hermes"` workspace. No direct security impact; the workspace is a namespace identifier, not a trust boundary.

5. **`agent/anthropic_adapter.py:1266` sanitizer**: `text.replace("Nous Research", "Anthropic")` was left intact in Phase 0.2 C5 as a deliberate exclusion. However, the earlier C4 broad sed corrupted the upper lines of this function (`text.replace("Hermes", "Claude Code")` → `text.replace("Claudia", "Claude Code")`, which is now semantically nonsense because the model output string is still "Hermes" in a model that thinks it's Hermes). **Phase 1.2 must review this function and likely remove it entirely** — Claudia doesn't need to impersonate Claude Code.

6. **Package rename `hermes-agent` → `claudia-autonomous`**. The egg-info directory becomes `claudia_autonomous.egg-info`. No security impact; first build produces the new name.

## What this baseline does NOT cover

- Penetration testing of any kind
- Fuzzing of approval-pattern regexes
- Sandbox escape attempts
- Gateway platform API abuse
- Upstream Hermes security advisories (no CVE database integration)
- The eventual Phase 2A/B Claudia memory provider (not implemented yet)
- The eventual Phase 4 proactive behavior hooks (not implemented yet)
- The eventual Phase 5 cost governance enforcement (not implemented yet)

Each new phase of Claudia Autonomous work should re-audit the surfaces it touches.

## Recommendation summary

The v0.7.0 fork inherits a solid security baseline: the approval system, container isolation, pairing protocol, and secret redaction are all textbook implementations, and the combination of pattern detection + content scanning + container sandbox gives defense in depth. The bet that Claudia Autonomous inherits this baseline **holds** — the Phase 0.2 rebrand was mechanical and did not weaken any control.

The items flagged in §9 "Consolidated gaps" should feed directly into Phase 0.4 test harness writing and the phase-specific reviews called out. No blocker on proceeding to Phase 0.4.
