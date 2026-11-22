"""
Budget guard for the bot.

The bot runs on a monthly Claude API credit that renews on a fixed day of the
month. This module keeps a small ledger of what the bot has spent in the
current credit cycle and picks a model tier from it, so the bot gets cheaper as
the credit runs down and stops before it would spend real money.

The ledger is a JSON file in `.budget/`. On GitHub Actions the workflow restores
and saves that folder through the Actions cache, because each run starts on a
fresh machine. The ledger only counts what this bot spends; other use of the
same API organization is not visible to it.

Settings (environment variables, all optional):
    BUDGET_USD          credit per cycle in USD (default 200)
    BUDGET_CYCLE_DAY    day of month the credit renews, in UTC (default 27)
    BUDGET_PER_RUN_USD  hard cap for one run in USD (default 25)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

LEDGER_PATH = Path(".budget") / "ledger.json"

# Share of the cycle budget already spent at which each tier starts.
ECONOMY_FROM = 0.70
MINIMAL_FROM = 0.90
STOP_FROM = 1.00


@dataclass(frozen=True)
class Tier:
    name: str
    forecaster_model: str | None
    predictions_per_report: int


TIERS = {
    "full": Tier("full", "anthropic/claude-fable-5-2", 3),
    "economy": Tier("economy", "anthropic/claude-opus-5-5", 3),
    "minimal": Tier("minimal", "anthropic/claude-sonnet-5-5", 1),
    "stop": Tier("stop", None, 0),
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    try:
        return float(raw) if raw else default
    except ValueError:
        logger.warning(f"{name}={raw!r} is not a number; using {default}")
        return default


def budget_usd() -> float:
    return _env_float("BUDGET_USD", 200.0)


def per_run_cap_usd() -> float:
    return _env_float("BUDGET_PER_RUN_USD", 25.0)


def cycle_day() -> int:
    return max(1, min(28, int(_env_float("BUDGET_CYCLE_DAY", 27))))


def cycle_start(today: date | None = None) -> date:
    """First day (UTC) of the credit cycle that contains `today`."""
    today = today or datetime.now(timezone.utc).date()
    day = cycle_day()
    if today.day >= day:
        return today.replace(day=day)
    year, month = (today.year, today.month - 1) if today.month > 1 else (today.year - 1, 12)
    return date(year, month, day)


def load_spent(today: date | None = None) -> float:
    """USD the bot has spent in the current cycle. A ledger from an older cycle counts as zero."""
    start = cycle_start(today).isoformat()
    try:
        data = json.loads(LEDGER_PATH.read_text())
    except FileNotFoundError:
        return 0.0
    except (OSError, ValueError) as error:
        logger.warning(f"Could not read budget ledger ({error}); treating spend as 0")
        return 0.0
    if data.get("cycle_start") != start:
        return 0.0
    return float(data.get("spent_usd", 0.0))


def record_spend(amount_usd: float, today: date | None = None) -> float:
    """Add `amount_usd` to the current cycle's ledger and return the new total."""
    start = cycle_start(today).isoformat()
    total = load_spent(today) + max(0.0, amount_usd)
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEDGER_PATH.write_text(
        json.dumps(
            {
                "cycle_start": start,
                "spent_usd": round(total, 4),
                "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
    )
    return total


def pick_tier(spent_usd: float) -> Tier:
    share = spent_usd / budget_usd() if budget_usd() > 0 else 1.0
    if share >= STOP_FROM:
        return TIERS["stop"]
    if share >= MINIMAL_FROM:
        return TIERS["minimal"]
    if share >= ECONOMY_FROM:
        return TIERS["economy"]
    return TIERS["full"]


def run_cap_usd(spent_usd: float) -> float:
    """Most this run may spend: the per-run cap, or what is left of the cycle budget if that is less."""
    remaining = max(0.0, budget_usd() - spent_usd)
    return max(0.01, min(per_run_cap_usd(), remaining))
