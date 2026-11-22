"""
Records of what the bot did, written as files under bot_config.DATA_DIR.

The workflow pushes that folder to the `bot-data` branch after each run, which
is where reviews and offline scoring read from. One JSON line per question goes
to runs/YYYY-MM/<run>.jsonl; a small status file per run goes to status/.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bot_config

logger = logging.getLogger(__name__)

_RUN_STARTED = datetime.now(timezone.utc)
_RUN_ID = _RUN_STARTED.strftime("%Y%m%dT%H%M%SZ") + "_" + os.getenv("GITHUB_RUN_ID", "local")


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    return str(value)


def record_question(mode: str, question: Any, research: str, slot_results: list[dict[str, Any]]) -> None:
    """Append one question's research and every slot's forecast to this run's file. Never raises."""
    try:
        record = {
            "run_id": _RUN_ID,
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mode": mode,
            "question": {
                "post_id": getattr(question, "id_of_post", None),
                "question_id": getattr(question, "id_of_question", None),
                "url": getattr(question, "page_url", None),
                "title": getattr(question, "question_text", None),
                "type": type(question).__name__,
                "close_time": _to_jsonable(getattr(question, "close_time", None)),
                "scheduled_resolution_time": _to_jsonable(
                    getattr(question, "scheduled_resolution_time", None)
                ),
                "tournament_slugs": _to_jsonable(getattr(question, "tournament_slugs", None)),
            },
            "research": research,
            "slots": [_to_jsonable(result) for result in slot_results],
        }
        folder = Path(bot_config.DATA_DIR) / "runs" / _RUN_STARTED.strftime("%Y-%m")
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{_RUN_ID}_{mode}.jsonl", "a") as run_file:
            run_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as error:
        logger.error(f"Could not write the run record: {error}")


def record_status(mode: str, status: dict[str, Any]) -> None:
    """Write this run's status file. Never raises."""
    try:
        folder = Path(bot_config.DATA_DIR) / "status"
        folder.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": _RUN_ID,
            "mode": mode,
            "started_utc": _RUN_STARTED.isoformat(timespec="seconds"),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **_to_jsonable(status),
        }
        (folder / f"latest_{mode}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        if payload.get("questions_forecast") or payload.get("errors"):
            history = Path(bot_config.DATA_DIR) / "status" / "history"
            history.mkdir(parents=True, exist_ok=True)
            with open(history / f"{_RUN_STARTED.strftime('%Y-%m')}.jsonl", "a") as history_file:
                history_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception as error:
        logger.error(f"Could not write the run status: {error}")
