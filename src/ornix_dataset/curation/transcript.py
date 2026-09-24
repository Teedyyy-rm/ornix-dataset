"""Transcript QA (spec §3 TRANSCRIPT_MISMATCH gate, Phase 4/5).

We do NOT auto-generate the official transcript with ASR (out-of-scope v1). An
ASR hypothesis, when supplied, is used only to *flag* mismatches. A transcript is
ACCEPT-eligible only when explicitly verified (human or strong evidence).
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from ..contracts.enums import MeasurementStatus

_WS = re.compile(r"\s+")


def normalize_vi(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower().strip()
    text = re.sub(r"[^\w\sàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệ"
                  r"ìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]", " ", text)
    return _WS.sub(" ", text).strip()


def char_error_rate(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize_vi(reference), normalize_vi(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein distance (character level)
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def transcript_match_status(reference: Optional[str], hypothesis: Optional[str] = None,
                            verified: bool = False, cer_threshold: float = 0.35
                            ) -> MeasurementStatus:
    if reference is None or reference.strip() == "":
        return MeasurementStatus.NOT_APPLICABLE
    if verified:
        return MeasurementStatus.OK
    if hypothesis is None:
        return MeasurementStatus.UNKNOWN  # cannot confirm without evidence/human
    cer = char_error_rate(reference, hypothesis)
    if cer > cer_threshold:
        return MeasurementStatus.ERROR
    return MeasurementStatus.UNKNOWN  # flagged-consistent, still needs sign-off
