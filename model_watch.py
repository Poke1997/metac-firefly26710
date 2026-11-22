"""
Check whether Anthropic has released a model newer than the ones the bot uses.

Writes status/models.json in the data folder. It only reports; it never changes
the bot's models, because a new model should pass a test run before it makes
scored forecasts. To switch, set FORECASTER_MODEL (see bot_config.py).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

import bot_config

logger = logging.getLogger(__name__)


def _bare(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    response = requests.get(
        "https://api.anthropic.com/v1/models",
        params={"limit": 100},
        headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        },
        timeout=60,
    )
    response.raise_for_status()
    models = [
        {
            "id": item.get("id"),
            "display_name": item.get("display_name"),
            "created_at": item.get("created_at"),
        }
        for item in response.json().get("data", [])
    ]
    in_use = {
        "forecaster": _bare(bot_config.FORECASTER_MODEL),
        "economy": _bare(bot_config.ECONOMY_MODEL),
        "helper": _bare(bot_config.HELPER_MODEL),
    }
    created = {model["id"]: model["created_at"] or "" for model in models}
    forecaster_created = created.get(in_use["forecaster"], "")
    newer = [
        model
        for model in models
        if model["id"] not in in_use.values()
        and forecaster_created
        and (model["created_at"] or "") > forecaster_created
    ]
    missing = [name for name, model_id in in_use.items() if model_id not in created]

    payload = {
        "checked_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "in_use": in_use,
        "in_use_but_not_listed_by_api": missing,
        "released_after_forecaster": newer,
        "all_models": models,
    }
    folder = Path(bot_config.DATA_DIR) / "status"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "models.json").write_text(json.dumps(payload, indent=2))
    logger.info(f"In use: {in_use}")
    logger.info(f"Released after the forecaster: {[model['id'] for model in newer]}")
    if missing:
        logger.warning(f"Configured but not listed by the API: {missing}")


if __name__ == "__main__":
    main()
