#!/usr/bin/env bash
# ============================================================================
# Claudia Autonomous — one-line installer
# ============================================================================
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.sh | bash
#
# What it does:
#   1. Checks prerequisites (git, curl, python3 >= 3.11)
#   2. Installs Ollama if missing (macOS via Homebrew, Linux via official script)
#   3. Clones kbanc85/claudia-autonomous to $INSTALL_DIR
#   4. Runs setup-claudia.sh (which uses uv, installs deps, symlinks claudia)
#   5. Pulls the two default Ollama models for the Claudia Hybrid Memory plugin
#   6. Prints next steps
#
# Environment variable overrides:
#   INSTALL_DIR   Where to clone the repo  (default: ~/claudia-autonomous)
#   REPO_URL      Git URL of the repo      (default: https://github.com/kbanc85/claudia-autonomous.git)
#   BRANCH        Branch or tag to check out (default: main)
#   SKIP_OLLAMA   Skip Ollama install + model pull  (default: unset)
#   SKIP_SETUP    Skip running setup-claudia.sh   (default: unset)
#   SKIP_MODELS   Skip pulling Ollama models  (default: unset)
#
# Author: Kamil Banc (https://github.com/kbanc85)
# License: MIT
# ============================================================================

set -eu

# ----- Colors -----
if [ -t 1 ]; then
  BOLD="$(printf '\033[1m')"
  DIM="$(printf '\033[2m')"
  RED="$(printf '\033[31m')"
  GREEN="$(printf '\033[32m')"
  YELLOW="$(printf '\033[33m')"
  CYAN="$(printf '\033[36m')"
  RESET="$(printf '\033[0m')"
else
  BOLD="" DIM="" RED="" GREEN="" YELLOW="" CYAN="" RESET=""
fi

info()  { printf "%s==>%s %s\n" "$CYAN" "$RESET" "$*"; }
warn()  { printf "%s!%s   %s\n" "$YELLOW" "$RESET" "$*"; }
ok()    { printf "%s✓%s   %s\n" "$GREEN" "$RESET" "$*"; }
fail()  { printf "%s✗%s   %s\n" "$RED" "$RESET" "$*" >&2; exit 1; }
h1()    { printf "\n%s%s%s\n" "$BOLD" "$*" "$RESET"; }

# ----- Defaults -----
INSTALL_DIR="${INSTALL_DIR:-$HOME/claudia-autonomous}"
REPO_URL="${REPO_URL:-https://github.com/kbanc85/claudia-autonomous.git}"
BRANCH="${BRANCH:-main}"
SKIP_OLLAMA="${SKIP_OLLAMA:-}"
SKIP_SETUP="${SKIP_SETUP:-}"
SKIP_MODELS="${SKIP_MODELS:-}"

# ----- Header -----
printf "\n"
printf "%s%s╭─────────────────────────────────────────────╮%s\n" "$BOLD" "$CYAN" "$RESET"
printf "%s%s│  Claudia Autonomous — one-line installer    │%s\n" "$BOLD" "$CYAN" "$RESET"
printf "%s%s│  Trust-aware local AI memory + chief-of-staff│%s\n" "$BOLD" "$CYAN" "$RESET"
printf "%s%s╰─────────────────────────────────────────────╯%s\n" "$BOLD" "$CYAN" "$RESET"
printf "\n"
printf "%sTarget:%s %s\n" "$DIM" "$RESET" "$INSTALL_DIR"
printf "%sBranch:%s %s\n" "$DIM" "$RESET" "$BRANCH"
printf "\n"

# ----- Detect OS -----
h1 "Checking environment"
OS="$(uname -s)"
case "$OS" in
  Darwin) OS_KIND="macos" ;;
  Linux)  OS_KIND="linux" ;;
  *) fail "Unsupported OS: $OS. Claudia supports macOS and Linux." ;;
esac
ok "OS: $OS_KIND"

# ----- Check git -----
if ! command -v git >/dev/null 2>&1; then
  fail "git not found. Install git and re-run."
fi
ok "git: $(git --version | awk '{print $3}')"

# ----- Check curl -----
if ! command -v curl >/dev/null 2>&1; then
  fail "curl not found. Install curl and re-run."
fi
ok "curl: present"

# ----- Check python3 >= 3.11 -----
if ! command -v python3 >/dev/null 2>&1; then
  fail "python3 not found. Install Python 3.11 or newer and re-run."
fi

PY_VERSION="$(python3 -c 'import sys; print(".".join(str(v) for v in sys.version_info[:3]))')"
PY_OK="$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 11) else 0)')"
if [ "$PY_OK" != "1" ]; then
  warn "python3 is $PY_VERSION, but Claudia needs 3.11 or newer."
  warn "The setup script will try to install a newer Python via uv, so continuing."
else
  ok "python3: $PY_VERSION"
fi

# ----- Install / check Ollama -----
if [ -z "$SKIP_OLLAMA" ]; then
  h1 "Checking Ollama"
  if command -v ollama >/dev/null 2>&1; then
    ok "ollama: already installed"
  else
    info "Installing Ollama..."
    case "$OS_KIND" in
      macos)
        if command -v brew >/dev/null 2>&1; then
          brew install ollama || fail "brew install ollama failed"
          ok "Ollama installed via Homebrew"
        else
          warn "Homebrew not found. Please install Ollama manually:"
          warn "    https://ollama.com/download"
          warn ""
          warn "After installing Ollama, re-run this installer. Or re-run with:"
          warn "    SKIP_OLLAMA=1 curl -fsSL https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.sh | bash"
          exit 1
        fi
        ;;
      linux)
        curl -fsSL https://ollama.com/install.sh | sh || fail "Ollama install script failed"
        ok "Ollama installed"
        ;;
    esac
  fi
fi

# ----- Clone or update the repo -----
h1 "Fetching the code"
if [ -d "$INSTALL_DIR" ] && git -C "$INSTALL_DIR" rev-parse --git-dir >/dev/null 2>&1; then
  # .git can be a directory (normal clone) or a file (submodule pointer) —
  # git rev-parse handles both.
  info "Repo already exists at $INSTALL_DIR — updating..."
  git -C "$INSTALL_DIR" fetch --quiet origin "$BRANCH" || fail "git fetch failed"
  git -C "$INSTALL_DIR" checkout --quiet "$BRANCH" || fail "git checkout failed"
  if ! git -C "$INSTALL_DIR" pull --quiet --ff-only origin "$BRANCH" 2>/dev/null; then
    warn "Non-fast-forward — leaving local state alone."
    warn "To reset: rm -rf $INSTALL_DIR && re-run installer."
  fi
  ok "Updated to latest $BRANCH"
elif [ -e "$INSTALL_DIR" ]; then
  fail "$INSTALL_DIR exists but is not a git repo. Move it aside and re-run."
else
  info "Cloning $REPO_URL to $INSTALL_DIR..."
  mkdir -p "$(dirname "$INSTALL_DIR")"
  git clone --quiet --branch "$BRANCH" --recurse-submodules "$REPO_URL" "$INSTALL_DIR" || \
    fail "git clone failed"
  ok "Cloned"
fi

# ----- Run setup-claudia.sh -----
if [ -z "$SKIP_SETUP" ]; then
  h1 "Running setup-claudia.sh"
  if [ ! -x "$INSTALL_DIR/setup-claudia.sh" ]; then
    fail "setup-claudia.sh not found or not executable at $INSTALL_DIR/setup-claudia.sh"
  fi
  # setup-claudia.sh has up to 2 interactive prompts (ripgrep install, setup
  # wizard). Pipe a stream of "n" characters with NO newlines so that each
  # `read -n 1 -r` in setup-claudia.sh consumes one "n" and falls to the "no"
  # branch. Newlines here would be interpreted as delimiters and reset REPLY
  # to empty, which the script treats as "yes". The user can run
  # `claudia memory setup` and `claudia setup` interactively afterward.
  (cd "$INSTALL_DIR" && printf "nnnnnnnnnn" | ./setup-claudia.sh) || \
    fail "setup-claudia.sh failed. See output above."
  ok "setup-claudia.sh completed"
fi

# ----- Pull Ollama models for the memory plugin -----
if [ -z "$SKIP_OLLAMA" ] && [ -z "$SKIP_MODELS" ] && command -v ollama >/dev/null 2>&1; then
  h1 "Pulling Ollama models for Claudia Memory"
  # Check if the daemon is running
  if ! curl -fsS http://localhost:11434/api/tags >/dev/null 2>&1; then
    warn "Ollama daemon isn't running yet. Start it with:"
    case "$OS_KIND" in
      macos) warn "    open -a Ollama" ;;
      linux) warn "    sudo systemctl start ollama" ;;
    esac
    warn ""
    warn "Then pull the two default models manually:"
    warn "    ollama pull all-minilm:l6-v2"
    warn "    ollama pull qwen2.5:3b"
  else
    info "Pulling all-minilm:l6-v2 (embeddings, ~23 MB)..."
    ollama pull all-minilm:l6-v2 || warn "Failed to pull all-minilm:l6-v2"
    info "Pulling qwen2.5:3b (entity extraction + commitment detection, ~2 GB)..."
    ollama pull qwen2.5:3b || warn "Failed to pull qwen2.5:3b"
    ok "Models ready"
  fi
fi

# ----- Done -----
printf "\n"
printf "%s%s╭─────────────────────────────────────────────╮%s\n" "$BOLD" "$GREEN" "$RESET"
printf "%s%s│  ✓ Claudia installed.                       │%s\n" "$BOLD" "$GREEN" "$RESET"
printf "%s%s╰─────────────────────────────────────────────╯%s\n" "$BOLD" "$GREEN" "$RESET"
printf "\n"
printf "Next steps:\n"
printf "\n"
printf "  %s1.%s Reload your shell (so ~/.local/bin is on PATH):\n" "$BOLD" "$RESET"
printf "       source ~/.zshrc   %s# or ~/.bashrc%s\n" "$DIM" "$RESET"
printf "\n"
printf "  %s2.%s Activate Claudia's hybrid memory provider:\n" "$BOLD" "$RESET"
printf "       claudia memory setup   %s# pick 'claudia' in the picker%s\n" "$DIM" "$RESET"
printf "\n"
printf "  %s3.%s Verify everything is wired up:\n" "$BOLD" "$RESET"
printf "       claudia doctor\n"
printf "\n"
printf "  %s4.%s (Optional) Run the offline memory demo:\n" "$BOLD" "$RESET"
printf "       cd %s && python -m plugins.memory.claudia.demo\n" "$INSTALL_DIR"
printf "\n"
printf "  %s5.%s Start a session:\n" "$BOLD" "$RESET"
printf "       claudia\n"
printf "\n"
printf "%sDocs:%s https://github.com/kbanc85/claudia-autonomous\n" "$DIM" "$RESET"
printf "%sMemory plugin:%s %s/plugins/memory/claudia/README.md\n" "$DIM" "$RESET" "$INSTALL_DIR"
printf "\n"
