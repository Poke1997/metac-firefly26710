"""
Settings for the bot, all in one place.

Every value can be overridden with an environment variable of the same name
(on GitHub Actions: a repository variable under Settings -> Secrets and
variables -> Actions -> Variables). To move the bot to a newer model, change
the default here or set the variable; nothing else names a model.
"""

from __future__ import annotations

import os
from datetime import date, datetime, timezone


def _env(name: str, default: str) -> str:
    value = os.getenv(name, "").strip()
    return value or default


# Models (litellm names). FORECASTER_MODEL makes the submitted forecasts.
FORECASTER_MODEL = _env("FORECASTER_MODEL", "anthropic/claude-fable-5-2")
# Used instead of the forecaster once 70% of the cycle budget is spent.
ECONOMY_MODEL = _env("ECONOMY_MODEL", "anthropic/claude-opus-5-5")
# Research, summaries, output parsing, and the forecaster once 90% is spent.
HELPER_MODEL = _env("HELPER_MODEL", "anthropic/claude-sonnet-5-5")

# Prompt used by the third submitted forecast. "template" turns the experiment off.
CHALLENGER_VARIANT = _env("CHALLENGER_VARIANT", "base_rate_first")

# Shadow forecasts are made and logged but never submitted. They exist to
# compare cheaper setups against the forecaster on the same research.
# Format: comma-separated "kind:count"; kinds are economy, helper, forecaster_medium.
SHADOW_FORECASTS = _env("SHADOW_FORECASTS", "economy:2,helper:1,forecaster_medium:1")
# Shadows stop by themselves at the start of this UTC date. The default is the
# day the first API credit cycle ended; set a later date to run them again.
SHADOWS_UNTIL = _env("SHADOWS_UNTIL", "2026-11-27")

# Where a run writes its records before the workflow pushes them to the bot-data branch.
DATA_DIR = _env("BOT_DATA_DIR", ".bot_data")


def shadows_active(today: date | None = None) -> bool:
    today = today or datetime.now(timezone.utc).date()
    try:
        return today < date.fromisoformat(SHADOWS_UNTIL)
    except ValueError:
        return False


def shadow_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for part in SHADOW_FORECASTS.split(","):
        kind, _, number = part.strip().partition(":")
        if kind in ("economy", "helper", "forecaster_medium") and number.isdigit():
            counts[kind] = int(number)
    return counts
