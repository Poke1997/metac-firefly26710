"""
Score every logged forecast slot against real outcomes, offline.

Reads the run records the bot saved (runs/YYYY-MM/*.jsonl in the bot-data
branch), asks Metaculus how each closed question resolved, and scores each
slot's own forecast. This is how an experiment is read out: the challenger
slot is compared with the two control slots on the same questions and the
same research. It makes no forecasts and uses no LLM tokens.

Score used (higher is better, 0 = no better than an uninformed forecast,
the same shape as Metaculus's baseline score):
  binary            100 * (ln(p) / ln(2) + 1)        p = probability given to what happened
  multiple choice   100 * (ln(p) / ln(K) + 1)        K = number of options
  numeric/date      100 * ln(p / baseline) / 2       p = probability given to the outcome's
                                                     bucket of the 201-point distribution
The tournament itself pays on peer score, which needs every other bot's
forecast. The difference between two of our own slots on the same question is
the same under both scores up to a constant factor, so comparisons hold.

Usage:
  python score_slots.py --data <folder holding runs/ and (optionally) scores/>
                        [--resolutions-file <json>]   # tests: skip Metaculus
Writes scores/resolutions.json, scores/slot_scores.csv, scores/summary.json and
scores/summary.md into bot_config.DATA_DIR.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import logging
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bot_config

logger = logging.getLogger(__name__)

MIN_QUESTIONS = 100  # an experiment is not read out below this many scored questions
SE_MULTIPLE = 2.0  # adopt only if the gain is more than this many standard errors
PROBABILITY_FLOOR = 1e-6
CANCELED = {"annulled", "ambiguous"}
OUT_OF_BOUNDS = {"above_upper_bound", "below_lower_bound"}
SUBMITTED_LINEUP = ["control_1", "control_2", "challenger"]
MIX_LINEUP = ["control_1", "shadow_economy_1", "shadow_economy_2"]


# ---------------------------------------------------------------- reading records


def load_records(data_dir: Path) -> list[dict[str, Any]]:
    """One record per question id (the earliest), from every run file."""
    by_question: dict[Any, dict[str, Any]] = {}
    for run_file in sorted((data_dir / "runs").glob("*/*.jsonl")):
        for line in run_file.read_text().splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping an unreadable line in {run_file.name}")
                continue
            question = record.get("question") or {}
            key = question.get("question_id") or question.get("post_id")
            if key is None:
                continue
            if key not in by_question or record.get("time_utc", "") < by_question[key].get("time_utc", ""):
                by_question[key] = record
    return list(by_question.values())


def _parse_time(text: Any) -> datetime | None:
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- resolutions


def fetch_resolutions(records: list[dict[str, Any]], known: dict[str, Any]) -> dict[str, Any]:
    """Ask Metaculus about closed questions that are not yet known to be resolved."""
    from forecasting_tools import MetaculusClient

    client = MetaculusClient()
    now = datetime.now(timezone.utc)
    for record in records:
        question = record["question"]
        key = str(question.get("question_id"))
        if known.get(key, {}).get("resolution") is not None:
            continue
        close_time = _parse_time(question.get("close_time"))
        if close_time and close_time > now:
            continue
        post_id = question.get("post_id")
        if post_id is None:
            continue
        try:
            fetched = client.get_question_by_post_id(int(post_id), group_question_mode="unpack_subquestions")
            candidates = fetched if isinstance(fetched, list) else [fetched]
            match = next((item for item in candidates if str(item.id_of_question) == key), None)
            if match is None and len(candidates) == 1:
                match = candidates[0]
            if match is None:
                continue
            known[key] = {
                "post_id": post_id,
                "resolution": match.resolution_string,
                "resolved_utc": match.actual_resolution_time.isoformat()
                if match.actual_resolution_time
                else None,
                "checked_utc": now.isoformat(timespec="seconds"),
            }
        except Exception as error:
            logger.warning(f"Could not fetch post {post_id}: {type(error).__name__}: {error}")
    return known


# ---------------------------------------------------------------- scoring one forecast


def _cdf_points(prediction: dict[str, Any]) -> tuple[list[float], list[float]]:
    """(values, cumulative probabilities) of the distribution Metaculus received."""
    from forecasting_tools.data_models.numeric_report import NumericDistribution

    cdf = NumericDistribution(**prediction).get_cdf()
    return [point.value for point in cdf], [point.percentile for point in cdf]


def _numeric_outcome(resolution: str) -> float | str:
    if resolution in OUT_OF_BOUNDS:
        return resolution
    try:
        return float(resolution)
    except ValueError:
        parsed = _parse_time(resolution)
        if parsed is None:
            raise ValueError(f"Cannot read numeric resolution {resolution!r}")
        return parsed.timestamp()


def score_numeric_cdf(
    values: list[float], cumulative: list[float], open_lower: bool, open_upper: bool, resolution: str
) -> float:
    outcome = _numeric_outcome(resolution)
    buckets = len(values) - 1
    open_bounds = int(open_lower) + int(open_upper)
    inbound_baseline = (1 - 0.05 * open_bounds) / buckets
    if outcome == "below_lower_bound" or (isinstance(outcome, float) and outcome < values[0]):
        probability, baseline = cumulative[0], 0.05
    elif outcome == "above_upper_bound" or (isinstance(outcome, float) and outcome > values[-1]):
        probability, baseline = 1 - cumulative[-1], 0.05
    else:
        assert isinstance(outcome, float)
        index = min(max(bisect.bisect_right(values, outcome) - 1, 0), buckets - 1)
        probability, baseline = cumulative[index + 1] - cumulative[index], inbound_baseline
    return 100 * math.log(max(probability, PROBABILITY_FLOOR) / baseline) / 2


def score_binary(probability_yes: float, resolution: str) -> float:
    probability = probability_yes if resolution == "yes" else 1 - probability_yes
    return 100 * (math.log(max(probability, PROBABILITY_FLOOR)) / math.log(2) + 1)


def score_options(options: dict[str, float], resolution: str) -> float:
    if resolution not in options:
        raise ValueError(f"Resolution {resolution!r} is not one of the options {list(options)}")
    probability = max(options[resolution], PROBABILITY_FLOOR)
    return 100 * (math.log(probability) / math.log(len(options)) + 1)


def _kind(prediction: Any) -> str:
    if isinstance(prediction, (int, float)) and not isinstance(prediction, bool):
        return "binary"
    if isinstance(prediction, dict) and "predicted_options" in prediction:
        return "options"
    if isinstance(prediction, dict) and "declared_percentiles" in prediction:
        return "numeric"
    return "unknown"


def _options(prediction: dict[str, Any]) -> dict[str, float]:
    return {item["option_name"]: float(item["probability"]) for item in prediction["predicted_options"]}


def score_prediction(prediction: Any, resolution: str) -> float:
    kind = _kind(prediction)
    if kind == "binary":
        if resolution not in ("yes", "no"):
            raise ValueError(f"Binary question resolved as {resolution!r}")
        return score_binary(float(prediction), resolution)
    if kind == "options":
        return score_options(_options(prediction), resolution)
    if kind == "numeric":
        values, cumulative = _cdf_points(prediction)
        return score_numeric_cdf(
            values, cumulative, prediction["open_lower_bound"], prediction["open_upper_bound"], resolution
        )
    raise ValueError("Unknown prediction format")


def score_lineup(predictions: list[Any], resolution: str) -> float:
    """Score the median of several slots, combined the way the bot combines them."""
    kinds = {_kind(prediction) for prediction in predictions}
    if len(kinds) != 1:
        raise ValueError("Mixed prediction formats")
    kind = kinds.pop()
    if kind == "binary":
        return score_binary(statistics.median(float(prediction) for prediction in predictions), resolution)
    if kind == "options":
        option_lists = [_options(prediction) for prediction in predictions]
        medians = {name: statistics.median(options[name] for options in option_lists) for name in option_lists[0]}
        total = sum(medians.values()) or 1.0
        return score_options({name: value / total for name, value in medians.items()}, resolution)
    if kind == "numeric":
        curves = [_cdf_points(prediction) for prediction in predictions]
        values = curves[0][0]
        cumulative = [statistics.median(curve[1][i] for curve in curves) for i in range(len(values))]
        first = predictions[0]
        return score_numeric_cdf(values, cumulative, first["open_lower_bound"], first["open_upper_bound"], resolution)
    raise ValueError("Unknown prediction format")


# ---------------------------------------------------------------- scoring everything


def score_records(records: list[dict[str, Any]], resolutions: dict[str, Any]) -> tuple[list[dict], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    counts = {"logged": len(records), "resolved": 0, "canceled": 0, "scored": 0, "waiting": 0}
    for record in records:
        question = record["question"]
        resolution = (resolutions.get(str(question.get("question_id"))) or {}).get("resolution")
        if resolution is None:
            counts["waiting"] += 1
            continue
        if resolution in CANCELED:
            counts["canceled"] += 1
            continue
        counts["resolved"] += 1
        base = {
            "question_id": question.get("question_id"),
            "post_id": question.get("post_id"),
            "mode": record.get("mode"),
            "type": question.get("type"),
            "forecast_utc": record.get("time_utc"),
            "resolution": resolution,
        }
        usable: dict[str, Any] = {}
        for slot in record.get("slots") or []:
            if not isinstance(slot, dict) or slot.get("error") or slot.get("prediction") is None:
                continue
            try:
                score = score_prediction(slot["prediction"], resolution)
            except Exception as error:
                logger.warning(f"Question {base['question_id']} slot {slot.get('slot')}: {error}")
                continue
            usable[slot["slot"]] = slot["prediction"]
            rows.append({**base, "slot": slot["slot"], "variant": slot.get("variant"), "model": slot.get("model"), "score": round(score, 3)})
        for name, members in (("lineup_submitted", SUBMITTED_LINEUP), ("lineup_1_forecaster_2_economy", MIX_LINEUP)):
            if all(member in usable for member in members):
                try:
                    score = score_lineup([usable[member] for member in members], resolution)
                    rows.append({**base, "slot": name, "variant": None, "model": None, "score": round(score, 3)})
                except Exception as error:
                    logger.warning(f"Question {base['question_id']} {name}: {error}")
        if usable:
            counts["scored"] += 1
    return rows, counts


def _paired(rows: list[dict], left: list[str], right: list[str], modes: set[str] | None) -> dict[str, Any] | None:
    """Mean of (average of `left` slots) minus (average of `right` slots), question by question."""
    by_question: dict[Any, dict[str, float]] = {}
    for row in rows:
        if modes is None or row["mode"] in modes:
            by_question.setdefault(row["question_id"], {})[row["slot"]] = row["score"]
    differences = [
        statistics.mean(scores[name] for name in left) - statistics.mean(scores[name] for name in right)
        for scores in by_question.values()
        if all(name in scores for name in left + right)
    ]
    if not differences:
        return None
    count = len(differences)
    mean = statistics.mean(differences)
    standard_error = statistics.stdev(differences) / math.sqrt(count) if count > 1 else None
    if standard_error is None:
        verdict = "too early (1 question)"
    else:
        if mean > SE_MULTIPLE * standard_error:
            verdict = "better"
        elif mean < -SE_MULTIPLE * standard_error:
            verdict = "worse"
        else:
            verdict = "no detectable difference"
        if count < MIN_QUESTIONS:
            verdict = f"provisional, {count} of {MIN_QUESTIONS} questions: {verdict}"
    return {
        "questions": count,
        "mean_difference": round(mean, 2),
        "standard_error": round(standard_error, 2) if standard_error is not None else None,
        "verdict": verdict,
    }


COMPARISONS = [
    ("challenger vs controls", ["challenger"], ["control_1", "control_2"]),
    ("control_1 vs control_2 (noise check)", ["control_1"], ["control_2"]),
    ("economy model vs controls", ["shadow_economy_1", "shadow_economy_2"], ["control_1", "control_2"]),
    ("helper model vs controls", ["shadow_helper_1"], ["control_1", "control_2"]),
    ("forecaster at medium effort vs controls", ["shadow_forecaster_medium_1"], ["control_1", "control_2"]),
    ("1 forecaster + 2 economy vs submitted line-up", ["lineup_1_forecaster_2_economy"], ["lineup_submitted"]),
]
SCOPES = [("tournament and MiniBench only", {"tournament"}), ("all logged questions", None)]


def summarise(rows: list[dict], counts: dict[str, int], records: list[dict]) -> dict[str, Any]:
    slot_scores: dict[str, list[float]] = {}
    for row in rows:
        slot_scores.setdefault(row["slot"], []).append(row["score"])
    variants = sorted(
        {slot.get("variant") for record in records for slot in record.get("slots") or [] if isinstance(slot, dict) and slot.get("slot") == "challenger" and slot.get("variant")}
    )
    return {
        "scored_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "counts": counts,
        "challenger_variants_in_data": variants,
        "rule": f"adopt only with at least {MIN_QUESTIONS} scored questions and a gain above {SE_MULTIPLE:g} standard errors",
        "slots": {
            name: {"questions": len(scores), "mean_score": round(statistics.mean(scores), 2)}
            for name, scores in sorted(slot_scores.items())
        },
        "comparisons": {
            scope_name: {
                label: result
                for label, left, right in COMPARISONS
                if (result := _paired(rows, left, right, modes)) is not None
            }
            for scope_name, modes in SCOPES
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# Slot scores",
        "",
        f"Scored {summary['scored_utc']}. Questions logged: {counts['logged']}; resolved and scored: {counts['scored']}; "
        f"annulled: {counts['canceled']}; not resolved yet: {counts['waiting']}.",
        f"Challenger variant(s) in the data: {', '.join(summary['challenger_variants_in_data']) or 'none'}.",
        f"Rule: {summary['rule']}.",
        "",
        "Score: 0 = no better than an uninformed forecast; higher is better (see score_slots.py).",
        "",
    ]
    if not summary["slots"]:
        lines.append("No logged question has resolved yet, so there is nothing to score.")
        return "\n".join(lines) + "\n"
    lines += ["| Slot | Questions | Mean score |", "|---|---|---|"]
    lines += [f"| {name} | {item['questions']} | {item['mean_score']} |" for name, item in summary["slots"].items()]
    for scope_name, comparisons in summary["comparisons"].items():
        lines += ["", f"## Comparisons: {scope_name}", ""]
        if not comparisons:
            lines.append("Nothing to compare yet.")
            continue
        lines += ["| Comparison | Questions | Mean difference | Standard error | Reading |", "|---|---|---|---|---|"]
        for label, item in comparisons.items():
            error = "-" if item["standard_error"] is None else item["standard_error"]
            lines.append(f"| {label} | {item['questions']} | {item['mean_difference']} | {error} | {item['verdict']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Score logged forecast slots against outcomes.")
    parser.add_argument("--data", required=True, help="Folder holding runs/ (a checkout of the bot-data branch)")
    parser.add_argument("--resolutions-file", help="Use this resolutions JSON and do not call Metaculus")
    arguments = parser.parse_args()

    data_dir = Path(arguments.data)
    records = load_records(data_dir)
    if arguments.resolutions_file:
        resolutions = json.loads(Path(arguments.resolutions_file).read_text())
    else:
        cache = data_dir / "scores" / "resolutions.json"
        known = json.loads(cache.read_text()) if cache.exists() else {}
        resolutions = fetch_resolutions(records, known)

    rows, counts = score_records(records, resolutions)
    summary = summarise(rows, counts, records)

    folder = Path(bot_config.DATA_DIR) / "scores"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "resolutions.json").write_text(json.dumps(resolutions, indent=1, sort_keys=True))
    fields = ["question_id", "post_id", "mode", "type", "forecast_utc", "resolution", "slot", "variant", "model", "score"]
    with open(folder / "slot_scores.csv", "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (folder / "summary.json").write_text(json.dumps(summary, indent=2))
    (folder / "summary.md").write_text(render_markdown(summary))
    logger.info(f"Scored {counts['scored']} questions; {counts['waiting']} not resolved yet.")


if __name__ == "__main__":
    main()
