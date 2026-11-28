#!/usr/bin/env bash
# Push the files a run left in the data folder to the `bot-data` branch.
#
# runs/ and status/history/ only ever grow; status/latest_*.json and the other
# status files are overwritten. A heartbeat is written at most every 6 hours so
# quiet runs (no open questions) do not create a commit each time.
#
# Usage: scripts/push_bot_data.sh [data_dir]     Needs GITHUB_TOKEN and GITHUB_REPOSITORY.
set -uo pipefail

DATA_DIR="${1:-.bot_data}"
BRANCH="bot-data"
REMOTE="${BOT_DATA_REMOTE:-https://x-access-token:${GITHUB_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}"
WORK="$(mktemp -d)"

for attempt in 1 2 3; do
  rm -rf "$WORK"
  if git clone --quiet --depth 1 --branch "$BRANCH" "$REMOTE" "$WORK" 2>/dev/null; then
    :
  else
    mkdir -p "$WORK"
    git -C "$WORK" init --quiet
    git -C "$WORK" checkout --quiet --orphan "$BRANCH"
    git -C "$WORK" remote add origin "$REMOTE"
    printf '# Bot data\n\nWritten by the bot: one record per forecast question under `runs/`, run status under `status/`, leaderboard snapshots under `track_record/`, weekly reviews under `reviews/` and `reports/`.\n' > "$WORK/README.md"
  fi

  has_news=false
  if [ -d "$DATA_DIR" ]; then
    while IFS= read -r -d '' file; do
      relative="${file#"$DATA_DIR"/}"
      mkdir -p "$WORK/$(dirname "$relative")"
      case "$relative" in
        status/history/*|track_record/*.csv)
          # Append-only files: add this run's lines to what the branch has.
          if [ -f "$WORK/$relative" ] && [[ "$relative" == *.csv ]]; then
            tail -n +2 "$file" >> "$WORK/$relative"
          else
            cat "$file" >> "$WORK/$relative"
          fi
          has_news=true ;;
        runs/*|reviews/*|reports/*|track_record/*|scores/*)
          cp "$file" "$WORK/$relative"; has_news=true ;;
        *)
          cp "$file" "$WORK/$relative" ;;
      esac
    done < <(find "$DATA_DIR" -type f -print0)
  fi

  # Heartbeat: refresh when there is news, or when the last one is over 6 hours old.
  now_epoch="$(date -u +%s)"
  last_epoch=0
  if [ -f "$WORK/status/heartbeat.json" ]; then
    last_epoch="$(sed -n 's/.*"epoch": *\([0-9]*\).*/\1/p' "$WORK/status/heartbeat.json" | head -1)"
    last_epoch="${last_epoch:-0}"
  fi
  if [ "$has_news" = true ] || [ $((now_epoch - last_epoch)) -gt 21600 ]; then
    mkdir -p "$WORK/status"
    printf '{\n  "last_run_utc": "%s",\n  "epoch": %s,\n  "workflow": "%s"\n}\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$now_epoch" "${GITHUB_WORKFLOW:-local}" > "$WORK/status/heartbeat.json"
  else
    echo "Nothing new and the heartbeat is fresh; not pushing."
    exit 0
  fi

  git -C "$WORK" add -A
  if git -C "$WORK" diff --cached --quiet; then
    echo "No changes to push."
    exit 0
  fi
  git -C "$WORK" -c user.name="firefly-bot" -c user.email="firefly-bot@users.noreply.github.com" \
    commit --quiet -m "Bot data: ${GITHUB_WORKFLOW:-local} run ${GITHUB_RUN_ID:-0}"
  if git -C "$WORK" push --quiet origin "$BRANCH"; then
    echo "Pushed bot data."
    exit 0
  fi
  echo "Push failed (attempt $attempt); retrying."
  sleep $((attempt * 5))
done

echo "Could not push bot data after 3 attempts." >&2
exit 0
