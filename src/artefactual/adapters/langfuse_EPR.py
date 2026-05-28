from artefactual.preprocessing import parse_top_logprobs
from langfuse import Langfuse
from artefactual.scoring import EPR

class Artefactual_EPR:
    def __init__(self, name: str, langfuse_client: Langfuse):
        self.name = name
        self.langfuse = langfuse_client

    def score_trace(self, trace_id: str) -> float:
        trace = self.langfuse.api.trace.get(trace_id) 
        
        epr = EPR()
        parsed_logprobs = parse_top_logprobs(trace.output)  

        value = epr.compute(parsed_logprobs)[0]

        self.langfuse.create_score(
            trace_id=trace_id,
            name=self.name,
            value=value
        )
        return value

