"""
Prompt construction for the four prompting conditions used in the paper.

Conditions (following paper terminology):
    MIN  — Minimal: replicates the human experiment instructions verbatim
    ALT  — Alternative-Aware: adds a one-shot chain-of-thought exemplar
    KMA  — Knowledge-and-Motives-Aware: adds explicit epistemic reasoning step
    COM  — Combined: integrates both ALT and KMA extensions
"""

task_description = (
    "In this experiment, you'll read a brief description of a situation involving two people. "
    "One of these people asks a question, and the second person answers it. "
    "Your task will be to answer some questions about the second person."
)

_example_block = """\
**Example situation:**
Jordan and Sam are planning a work meeting.
Jordan: "Do you know what time the meeting starts?"
Sam: "It starts at around 9."

**Example task:**
Based on what Sam says, how polite does Sam sound?
Use a 7-point Likert scale:
1 = not at all polite
7 = very polite

**Example reasoning (for illustration only):**
Sam gives an approximate answer rather than an exact time. In this context, an approximate \
answer can be appropriate and polite, since it provides useful information without unnecessary \
detail. Nothing in Sam's response suggests rudeness or disrespect.

**Example answer:**
6

**Now the actual task**
You will now see a new situation.
Please answer the question based only on the information given."""


def _task_block_minimal(speaker: str, attribute: str) -> str:
    return (
        f"Based on what {speaker} says, how {attribute} does {speaker} sound?\n\n"
        f"Use a 7-point Likert scale:\n"
        f"1 = not at all {attribute}\n"
        f"7 = very {attribute}\n\n"
        f"Answer with a single number between 1 and 7. Give only the number, no other text."
    )


def _task_block_kma(speaker: str, attribute: str) -> str:
    return (
        f"Based on what {speaker} says, how {attribute} does {speaker} sound?\n\n"
        f"Before you answer, please note:\n"
        f"The same utterance can arise from different speaker knowledge states and motivations. "
        f"You should therefore avoid assuming a single motive or level of knowledge unless the "
        f"context clearly supports it.\n\n"
        f"- Step 1: Briefly list two or three plausible reasons why the speaker might have chosen "
        f"this wording, considering both their possible knowledge state and their communicative goals.\n"
        f"- Step 2: Based on this uncertainty, provide a balanced social evaluation of the speaker.\n\n"
        f"Now it's your turn: How {attribute} does {speaker} sound?\n\n"
        f"Use a 7-point Likert scale:\n"
        f"1 = not at all {attribute}\n"
        f"7 = very {attribute}\n\n"
        f"Answer with a single number between 1 and 7. Give only the number, no other text."
    )


def _situation_block(scenario_items: dict, context: str, utterance: str) -> str:
    return f"{scenario_items[context]}\n{scenario_items[utterance]}"


# ── Public prompt builders ─────────────────────────────────────────────────────

def build_minimal_prompt(
    scenario_items: dict,
    context: str,
    utterance: str,
    attribute: str,
) -> str:
    """MIN condition: mirrors human experiment instructions."""
    speaker = scenario_items["speaker"]
    return (
        f"**Task description:**\n{task_description}\n\n"
        f"**Task situation:**\n{_situation_block(scenario_items, context, utterance)}\n\n"
        f"**Task:**\n{_task_block_minimal(speaker, attribute)}"
    ).strip()


def build_alt_prompt(
    scenario_items: dict,
    context: str,
    utterance: str,
    attribute: str,
) -> str:
    """ALT condition: adds one-shot chain-of-thought exemplar (alternative-awareness)."""
    speaker = scenario_items["speaker"]
    return (
        f"**Task description:**\n{task_description}\n\n"
        f"{_example_block}\n\n"
        f"**Task situation:**\n{_situation_block(scenario_items, context, utterance)}\n\n"
        f"**Task:**\n{_task_block_minimal(speaker, attribute)}"
    ).strip()


def build_kma_prompt(
    scenario_items: dict,
    context: str,
    utterance: str,
    attribute: str,
) -> str:
    """KMA condition: prompts explicit reasoning over speaker knowledge and motives."""
    speaker = scenario_items["speaker"]
    return (
        f"**Task description:**\n{task_description}\n\n"
        f"**Task situation:**\n{_situation_block(scenario_items, context, utterance)}\n\n"
        f"**Task:**\n{_task_block_kma(speaker, attribute)}"
    ).strip()


def build_combined_prompt(
    scenario_items: dict,
    context: str,
    utterance: str,
    attribute: str,
) -> str:
    """COM condition: integrates ALT exemplar and KMA epistemic reasoning."""
    speaker = scenario_items["speaker"]
    return (
        f"**Task description:**\n{task_description}\n\n"
        f"{_example_block}\n\n"
        f"**Task situation:**\n{_situation_block(scenario_items, context, utterance)}\n\n"
        f"**Task:**\n{_task_block_kma(speaker, attribute)}"
    ).strip()
