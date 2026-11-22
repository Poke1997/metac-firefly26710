"""
Forecast slots and prompt variants.

Each question gets a list of slots. A slot is one forecast: which model makes
it, which prompt variant it uses, and whether it is submitted or only logged.

Submitted slots are control, control, challenger. The two controls use the
unchanged template prompt; the challenger uses the variant being tested. The
median of three always lies between the two controls or on one of them, so the
challenger cannot move the submitted forecast outside the controls' range. The
gap between the two controls also shows how much two runs of the same prompt
differ by chance.

Shadow slots are never submitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import bot_config
import budget

# Text that starts the instruction part of every template forecast prompt.
# Everything before it (question, criteria, research) is shared by all slots.
INSTRUCTIONS_MARKER = "Before answering you write:"

BASE_RATE_FIRST_PREFACE = """First, write an OUTSIDE VIEW section before you consider the specifics of the current situation:
- Name the reference class this question belongs to (comparable past events, or how this quantity has behaved over periods of the same length).
- State the base rate or historical distribution for that reference class, using figures from the research where it gives them.
- State the forecast you would make from the outside view alone.

Then write an INSIDE VIEW section using the steps below. For each piece of current evidence, say whether it moves you away from the outside-view forecast, in which direction and by roughly how much. Weak or ambiguous evidence should move you only a little.

"""


@dataclass(frozen=True)
class Slot:
    name: str
    submitted: bool
    llm_key: str  # key into the bot's slot LLMs: forecaster, economy, helper, forecaster_medium
    variant: str  # template or base_rate_first


def build_slots(tier: budget.Tier) -> list[Slot]:
    if tier.predictions_per_report <= 1:
        return [Slot("control_1", True, "forecaster", "template")]

    slots = [
        Slot("control_1", True, "forecaster", "template"),
        Slot("control_2", True, "forecaster", "template"),
        Slot("challenger", True, "forecaster", bot_config.CHALLENGER_VARIANT),
    ]
    if tier.name == "full" and bot_config.shadows_active():
        for kind, count in bot_config.shadow_counts().items():
            for number in range(1, count + 1):
                slots.append(Slot(f"shadow_{kind}_{number}", False, kind, "template"))
    return slots


def split_prompt(prompt: str) -> tuple[str, str]:
    """Split a template forecast prompt into (shared context, instructions)."""
    index = prompt.find(INSTRUCTIONS_MARKER)
    if index <= 0:
        return "", prompt
    return prompt[:index], prompt[index:]


def apply_variant(instructions: str, variant: str) -> str:
    if variant == "base_rate_first":
        return BASE_RATE_FIRST_PREFACE + instructions
    return instructions
