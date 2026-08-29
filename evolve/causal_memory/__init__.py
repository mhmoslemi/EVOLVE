"""Causal option memory: matched-pair records, promotion, and retrieval."""

from .promotion import DRIFT_SIGMA, DRIFT_WINDOW, detect_drift, evaluate_promotion, stratify_drift
from .records import CausalMemoryError, add_contraindication, add_effect, memory_id_for, new_memory_record
from .retrieval import MemoryStore, MemoryStoreError

__all__ = [
    "DRIFT_SIGMA",
    "DRIFT_WINDOW",
    "CausalMemoryError",
    "MemoryStore",
    "MemoryStoreError",
    "add_contraindication",
    "add_effect",
    "detect_drift",
    "evaluate_promotion",
    "memory_id_for",
    "new_memory_record",
    "stratify_drift",
]
