"""
Data collection script: queries LLMs across all experimental conditions.

Usage:
    Set PROMPT_TYPE to one of: MIN, ALT, KMA, COM
    Then run: python -m src.collection.main

Results are saved incrementally to data/llm_ratings/.
"""

import json
import os
import time

from src.collection.models import MODEL_CALLERS
from src.parsing import parse_likert_response
from src.io_utils import save_json
from src.config import (
    N_SAMPLES, SLEEP_BETWEEN_CALLS,
    MODELS, CONTEXTS, UTTERANCES, SCENARIOS, ATTRIBUTES,
)
from prompts.build_prompts import (
    build_minimal_prompt,
    build_alt_prompt,
    build_kma_prompt,
    build_combined_prompt,
)

# ── Configuration ─────────────────────────────────────────────────────────────

RUN_MODEL = True
PRINT_PROMPT = False
PROMPT_TYPE = "COM"  # One of: MIN, ALT, KMA, COM

PROMPT_MAP = {
    "MIN": ("data/llm_ratings/results_MIN.json", build_minimal_prompt),
    "ALT": ("data/llm_ratings/results_ALT.json", build_alt_prompt),
    "KMA": ("data/llm_ratings/results_KMA.json", build_kma_prompt),
    "COM": ("data/llm_ratings/results_COM.json", build_combined_prompt),
}

# ── Experiment runner ──────────────────────────────────────────────────────────

def run_experiment():
    if PROMPT_TYPE not in PROMPT_MAP:
        raise ValueError(f"Invalid PROMPT_TYPE '{PROMPT_TYPE}'. Choose from: {list(PROMPT_MAP)}")

    results_file, build_prompt_func = PROMPT_MAP[PROMPT_TYPE]

    results = []
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

    # Track completed samples to allow resuming interrupted runs
    max_sample_ids = {}
    for result in results:
        key = (result["model"], result["scenario"], result["context"],
               result["utterance"], result["attribute"])
        max_sample_ids[key] = max(max_sample_ids.get(key, -1), result["sample_id"])

    for model in MODELS:
        for scenario, scenario_items in SCENARIOS.items():
            for context in CONTEXTS:
                for utterance in UTTERANCES:
                    for attribute in ATTRIBUTES:

                        prompt = build_prompt_func(
                            scenario_items=scenario_items,
                            context=context,
                            utterance=utterance,
                            attribute=attribute,
                        )

                        if PRINT_PROMPT:
                            print(prompt)

                        current_key = (model, scenario, context, utterance, attribute)
                        start_sample_id = max_sample_ids.get(current_key, -1) + 1

                        for sample_id in range(start_sample_id, N_SAMPLES):
                            print(f"{model=} {scenario=} {context=} "
                                  f"{utterance=} {attribute=} {sample_id=}")

                            if RUN_MODEL:
                                raw = MODEL_CALLERS[model](prompt)
                                likert, valid = parse_likert_response(raw)

                                results.append({
                                    "model": model,
                                    "scenario": scenario,
                                    "context": context,
                                    "utterance": utterance,
                                    "attribute": attribute,
                                    "sample_id": sample_id,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "raw_response": raw,
                                    "likert": likert,
                                    "valid": valid,
                                })
                                save_json(results, results_file)
                                time.sleep(SLEEP_BETWEEN_CALLS)


if __name__ == "__main__":
    run_experiment()
