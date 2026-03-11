from src.exp_texts import *

# ── Experiment configuration ──────────────────────────────────────────────────

N_SAMPLES = 10
SLEEP_BETWEEN_CALLS = 0.1

MODELS = ["claude", "gpt", "gemini"]

CONTEXTS = ["low", "high"]
UTTERANCES = ["precise", "approx"]

ATTRIBUTES = [
    "competent",
    "likeable",
    "pedantic",
    "helpful",
    "knowledgeable",
    "well-prepared",
]

SCENARIOS = {
    "house": {
        "speaker": "Dylan",
        "low": HOUSE_LOW,
        "high": HOUSE_HIGH,
        "approx": HOUSE_APPROX,
        "precise": HOUSE_PRECISE,
    },
    "cinema": {
        "speaker": "Quinn",
        "low": CINEMA_LOW,
        "high": CINEMA_HIGH,
        "approx": CINEMA_APPROX,
        "precise": CINEMA_PRECISE,
    },
    "conference": {
        "speaker": "Logan",
        "low": CONFERENCE_LOW,
        "high": CONFERENCE_HIGH,
        "approx": CONFERENCE_APPROX,
        "precise": CONFERENCE_PRECISE,
    },
    "bicycle": {
        "speaker": "Jamie",
        "low": BICYCLE_LOW,
        "high": BICYCLE_HIGH,
        "approx": BICYCLE_APPROX,
        "precise": BICYCLE_PRECISE,
    },
    "office": {
        "speaker": "Carter",
        "low": OFFICE_LOW,
        "high": OFFICE_HIGH,
        "approx": OFFICE_APPROX,
        "precise": OFFICE_PRECISE,
    },
    "pasta": {
        "speaker": "Taylor",
        "low": PASTA_LOW,
        "high": PASTA_HIGH,
        "approx": PASTA_APPROX,
        "precise": PASTA_PRECISE,
    },
}
