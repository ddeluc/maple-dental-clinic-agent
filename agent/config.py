"""Load clinic-specific settings (prompts, greeting) from config/*.yaml."""

from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "maple_dental.yaml"

# Keys the agent can't run without. Checking them at startup gives a clear
# error now instead of a KeyError in the middle of a conversation.
REQUIRED_TEXT_KEYS = (
    "system_prompt",
    "greeting",
    "language_detection_prompt",
    "unclear_language_reply",
)
LANGUAGES = ("french", "english")


def load_clinic_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    # utf-8 must be explicit: on Windows the default encoding would garble
    # the French accents in the greeting.
    with path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"{path} must contain 'key: value' settings at the top level.")

    for key in REQUIRED_TEXT_KEYS:
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} needs a non-empty text value for '{key}'.")

    language_rules = config.get("language_rules")
    for language in LANGUAGES:
        if not isinstance(language_rules, dict) or not language_rules.get(language):
            raise ValueError(f"{path} needs 'language_rules' text for '{language}'.")

    # The tool definitions offer these names to the model as the only valid
    # locations, so the agent can't start without them.
    locations = config.get("locations")
    if not isinstance(locations, dict) or not locations:
        raise ValueError(f"{path} needs a 'locations' section.")

    return config


def location_names(config: dict[str, Any]) -> list[str]:
    return list(config["locations"])
