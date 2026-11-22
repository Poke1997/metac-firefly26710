"""
Save the bot's current standing on a tournament leaderboard.

Appends one row to track_record/<project>.csv and writes
track_record/<project>_latest.json in the data folder. Run daily, this builds a
dated history of rank and score that stays in the repository's `bot-data`
branch after the season ends.

Usage: python track_record.py <numeric project id> [<numeric project id> ...]
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from metaculus_bot_review.client import ReviewClient

import bot_config

logger = logging.getLogger(__name__)

FIELDS = [
    "time_utc",
    "project_id",
    "project_name",
    "username",
    "rank",
    "score",
    "questions_counted",
    "prize",
    "ranked_entries",
    "score_type",
    "finalized",
]


def snapshot(client: ReviewClient, project_id: int) -> None:
    board = client.get_project_leaderboard(project_id)
    entry = board.user_entry
    ranked = [item for item in board.entries if item.rank is not None and not item.excluded]
    row = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "project_id": board.project_id,
        "project_name": board.project_name,
        "username": entry.username if entry else None,
        "rank": entry.rank if entry else None,
        "score": entry.score if entry else None,
        "questions_counted": entry.contribution_count if entry else None,
        "prize": entry.prize if entry else None,
        "ranked_entries": len(ranked),
        "score_type": board.score_type,
        "finalized": board.finalized,
    }
    folder = Path(bot_config.DATA_DIR) / "track_record"
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"project_{project_id}.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)
    (folder / f"project_{project_id}_latest.json").write_text(json.dumps(row, indent=2))
    logger.info(f"Leaderboard snapshot: {row}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    client = ReviewClient()
    for raw_id in sys.argv[1:]:
        try:
            snapshot(client, int(raw_id))
        except Exception as error:
            logger.error(f"Could not snapshot project {raw_id}: {error}")


if __name__ == "__main__":
    main()
