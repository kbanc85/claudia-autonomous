#!/bin/bash
# Docker entrypoint: bootstrap config files into the mounted volume, then run claudia.
set -e

CLAUDIA_HOME="/opt/data"
INSTALL_DIR="/opt/claudia"

# Create essential directory structure.  Cache and platform directories
# (cache/images, cache/audio, platforms/whatsapp, etc.) are created on
# demand by the application — don't pre-create them here so new installs
# get the consolidated layout from get_claudia_dir().
mkdir -p "$CLAUDIA_HOME"/{cron,sessions,logs,hooks,memories,skills}

# .env
if [ ! -f "$CLAUDIA_HOME/.env" ]; then
    cp "$INSTALL_DIR/.env.example" "$CLAUDIA_HOME/.env"
fi

# config.yaml
if [ ! -f "$CLAUDIA_HOME/config.yaml" ]; then
    cp "$INSTALL_DIR/cli-config.yaml.example" "$CLAUDIA_HOME/config.yaml"
fi

# SOUL.md
if [ ! -f "$CLAUDIA_HOME/SOUL.md" ]; then
    cp "$INSTALL_DIR/docker/SOUL.md" "$CLAUDIA_HOME/SOUL.md"
fi

# Sync bundled skills (manifest-based so user edits are preserved)
if [ -d "$INSTALL_DIR/skills" ]; then
    python3 "$INSTALL_DIR/tools/skills_sync.py"
fi

exec claudia "$@"
