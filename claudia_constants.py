"""Shared constants for Claudia.

Import-safe module with no dependencies — can be imported from anywhere
without risk of circular imports.
"""

import os
from pathlib import Path


def get_claudia_home() -> Path:
    """Return the Claudia home directory (default: ~/.claudia).

    Reads CLAUDIA_HOME env var, falls back to ~/.claudia.
    This is the single source of truth — all other copies should import this.
    """
    return Path(os.getenv("CLAUDIA_HOME", Path.home() / ".claudia"))


def get_optional_skills_dir(default: Path | None = None) -> Path:
    """Return the optional-skills directory, honoring package-manager wrappers.

    Packaged installs may ship ``optional-skills`` outside the Python package
    tree and expose it via ``CLAUDIA_OPTIONAL_SKILLS``.
    """
    override = os.getenv("CLAUDIA_OPTIONAL_SKILLS", "").strip()
    if override:
        return Path(override)
    if default is not None:
        return default
    return get_claudia_home() / "optional-skills"


def get_claudia_dir(new_subpath: str, old_name: str) -> Path:
    """Resolve a Claudia subdirectory with backward compatibility.

    New installs get the consolidated layout (e.g. ``cache/images``).
    Existing installs that already have the old path (e.g. ``image_cache``)
    keep using it — no migration required.

    Args:
        new_subpath: Preferred path relative to CLAUDIA_HOME (e.g. ``"cache/images"``).
        old_name: Legacy path relative to CLAUDIA_HOME (e.g. ``"image_cache"``).

    Returns:
        Absolute ``Path`` — old location if it exists on disk, otherwise the new one.
    """
    home = get_claudia_home()
    old_path = home / old_name
    if old_path.exists():
        return old_path
    return home / new_subpath


def display_claudia_home() -> str:
    """Return a user-friendly display string for the current CLAUDIA_HOME.

    Uses ``~/`` shorthand for readability::

        default:  ``~/.claudia``
        profile:  ``~/.claudia/profiles/coder``
        custom:   ``/opt/claudia-custom``

    Use this in **user-facing** print/log messages instead of hardcoding
    ``~/.claudia``.  For code that needs a real ``Path``, use
    :func:`get_claudia_home` instead.
    """
    home = get_claudia_home()
    try:
        return "~/" + str(home.relative_to(Path.home()))
    except ValueError:
        return str(home)


VALID_REASONING_EFFORTS = ("xhigh", "high", "medium", "low", "minimal")


def parse_reasoning_effort(effort: str) -> dict | None:
    """Parse a reasoning effort level into a config dict.

    Valid levels: "xhigh", "high", "medium", "low", "minimal", "none".
    Returns None when the input is empty or unrecognized (caller uses default).
    Returns {"enabled": False} for "none".
    Returns {"enabled": True, "effort": <level>} for valid effort levels.
    """
    if not effort or not effort.strip():
        return None
    effort = effort.strip().lower()
    if effort == "none":
        return {"enabled": False}
    if effort in VALID_REASONING_EFFORTS:
        return {"enabled": True, "effort": effort}
    return None


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS_URL = f"{OPENROUTER_BASE_URL}/models"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"

AI_GATEWAY_BASE_URL = "https://ai-gateway.vercel.sh/v1"
AI_GATEWAY_MODELS_URL = f"{AI_GATEWAY_BASE_URL}/models"
AI_GATEWAY_CHAT_URL = f"{AI_GATEWAY_BASE_URL}/chat/completions"

NOUS_API_BASE_URL = "https://inference-api.nousresearch.com/v1"
NOUS_API_CHAT_URL = f"{NOUS_API_BASE_URL}/chat/completions"
