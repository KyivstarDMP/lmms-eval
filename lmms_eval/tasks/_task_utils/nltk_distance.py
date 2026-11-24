"""NLTK 3.10.3 refuses distance inputs over 2000 characters; a page of OCR text is longer."""

import importlib
import sys


def allow_long_inputs() -> None:
    try:
        distance = importlib.import_module("nltk.metrics.distance")
    except ImportError:
        return

    if hasattr(distance, "MAX_DISTANCE_INPUT_LEN"):
        distance.MAX_DISTANCE_INPUT_LEN = sys.maxsize
