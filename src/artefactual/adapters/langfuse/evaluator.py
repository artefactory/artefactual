"""Scoring of Langfuse traces with an artefactual detector.

`langfuse` is an optional dependency (`pip install artefactual[adapters]`) and is imported
only for type checking, so this module is importable without it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from artefactual.scoring.base_detector import BaseDetector

if TYPE_CHECKING:
    from langfuse import Langfuse


class HallucinationEvaluator:
    """Scores a Langfuse trace's output and writes the result back as a trace score.

    The trace `output` must be the completion response itself, carrying logprobs — the
    same object `BaseDetector.predict_proba` accepts.
    """

    def __init__(self, name: str, langfuse_client: Langfuse, detector: BaseDetector) -> None:
        """
        Args:
            name: Score name recorded in Langfuse, used as the metric label in its UI.
            langfuse_client: An authenticated `Langfuse` client.
            detector: A calibrated detector, e.g. `epr("mistralai/Ministral-8B-Instruct-2410")`.
        """
        self.name = name
        self.langfuse = langfuse_client
        self.detector = detector

    def score_trace(self, trace_id: str) -> float:
        """Fetch a trace, score its output, and attach the probability to it.

        The score id is derived from the trace id and the metric name, so a trace carries
        one score per metric however often it is re-scored. Keying on the value instead
        would mint a fresh id whenever the score moved — after a weights update or a
        detector swap — and accumulate rows rather than replacing them.

        Args:
            trace_id: Id of the trace to fetch and score.

        Returns:
            P(hallucination) for the trace's first sequence, as a plain `float` —
            Langfuse serialises the value to JSON, which numpy scalars do not survive.

        Raises:
            TypeError: If the trace output is not a completion response carrying logprobs.
        """
        trace = self.langfuse.api.trace.get(trace_id)
        value = float(self.detector.predict_proba(trace.output)[0, 1])
        self.langfuse.create_score(
            trace_id=trace_id,
            name=self.name,
            value=value,
            score_id=f"{trace_id}-{self.name}",  # idempotency key: one score per metric
        )
        return value
