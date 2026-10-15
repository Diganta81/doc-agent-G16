"""Stage 9 — confidence calibration (calibrated-confidence NFR)"""
from __future__ import annotations
from ..contracts import *  # noqa
from . import metrics

def temperature_scale(logits, labels):
    """Fit temperature on val; return scaler. IMPLEMENT."""
    raise NotImplementedError("Calibration: temperature scaling")
def ece(confidences, correct) -> float:
    return metrics.ece(confidences, correct)

