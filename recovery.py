"""Recovery classification — determines appropriate response to node failures."""
from __future__ import annotations

from typing import Literal


FailureClass = Literal["transient", "validation_error", "upstream_failure"]

TRANSIENT_KEYWORDS = frozenset({
    "503", "502", "504", "timeout", "connection",
    "bad gateway", "gateway timeout", "connectionerror",
    "httpstatuserror", "service unavailable", "rate limit",
    "ratelimiterror", "apitimeouterror", "apiconnectionerror",
    "throttl",
})

VALIDATION_KEYWORDS = frozenset({
    "malformed", "validationerror", "validation error",
    "json", "parse error", "schema", "invalid json",
    "decode error", "jsondecodeerror",
})


def classify_failure(error_text: str) -> FailureClass:
    lower = error_text.lower()

    for kw in TRANSIENT_KEYWORDS:
        if kw in lower:
            return "transient"

    for kw in VALIDATION_KEYWORDS:
        if kw in lower:
            return "validation_error"

    return "upstream_failure"
