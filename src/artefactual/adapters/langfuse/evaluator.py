from langfuse import Langfuse

from artefactual.preprocessing import parse_top_logprobs
from artefactual.scoring.hallucination_detector import HallucinationDetector


class HallucinationEvaluator:
    def __init__(self, name: str, langfuse_client: Langfuse, detector: HallucinationDetector) -> None:
        self.name = name
        self.langfuse = langfuse_client
        self.detector = detector

    def score_trace(self, trace_id: str) -> float:
        trace = self.langfuse.api.trace.get(trace_id)
        parsed_logprobs = parse_top_logprobs(trace.output)
        value = self.detector.predict_proba(parsed_logprobs)[0]
        self.langfuse.create_score(
            trace_id=trace_id,
            name=self.name,
            value=value,
            id=f"{trace_id}-{value}",  # idempotency key
        )
        return value
