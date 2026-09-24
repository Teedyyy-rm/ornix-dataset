"""Calibration metrics (spec §5.2).

False-accept / false-reject rates are computed against human ground truth.
REVIEW / UNKNOWN are reported in their own buckets and never hidden from the
funnel denominator.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def classification_metrics(y_true: List[bool], y_pred: List[bool]) -> Dict[str, Any]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4)}


def false_accept_rate(gold_is_noisy: List[bool], decisions: List[str]) -> float:
    """noisy clips ACCEPTed / total noisy clips."""
    noisy = [d for g, d in zip(gold_is_noisy, decisions) if g]
    if not noisy:
        return 0.0
    return sum(1 for d in noisy if d == "ACCEPT") / len(noisy)


def false_reject_rate(gold_is_noisy: List[bool], decisions: List[str]) -> float:
    """clean clips REJECTed / total clean clips (REJECT or REJECT_TECH)."""
    clean = [d for g, d in zip(gold_is_noisy, decisions) if not g]
    if not clean:
        return 0.0
    return sum(1 for d in clean if d in ("REJECT", "REJECT_TECH")) / len(clean)


def coverage(decisions: List[str]) -> Dict[str, Any]:
    from collections import Counter

    c = Counter(decisions)
    total = len(decisions)
    return {"total": total, "by_decision": dict(c),
            "accept_fraction": round(c.get("ACCEPT", 0) / total, 4) if total else 0.0,
            "review_fraction": round(c.get("REVIEW", 0) / total, 4) if total else 0.0}


def calibration_report(gold_is_noisy: List[bool], decisions: List[str],
                       per_class: Dict[str, Tuple[List[bool], List[bool]]] = None
                       ) -> Dict[str, Any]:
    report = {
        "false_accept_rate": round(false_accept_rate(gold_is_noisy, decisions), 4),
        "false_reject_rate": round(false_reject_rate(gold_is_noisy, decisions), 4),
        "coverage": coverage(decisions),
        "per_class": {},
        "note": "Thresholds are operator-signed after baseline; no fabricated targets.",
    }
    for label, (yt, yp) in (per_class or {}).items():
        report["per_class"][label] = classification_metrics(yt, yp)
    return report
