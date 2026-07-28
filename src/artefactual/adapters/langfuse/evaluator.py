from __future__ import annotations

from typing import TYPE_CHECKING

from artefactual.scoring.base_detector import BaseDetector

if TYPE_CHECKING:
    from langfuse import Langfuse


class HallucinationEvaluator:
    def __init__(self, name: str, langfuse_client: Langfuse, detector: BaseDetector) -> None:
        self.name = name
        self.langfuse = langfuse_client
        self.detector = detector

    def score_trace(self, trace_id: str) -> float:
        trace = self.langfuse.api.trace.get(trace_id)
        value = float(self.detector.predict_proba(trace.output)[0, 1])
        self.langfuse.create_score(
            trace_id=trace_id,
            name=self.name,
            value=value,
            score_id=f"{trace_id}-{value}",  # idempotency key
        )
        return value
