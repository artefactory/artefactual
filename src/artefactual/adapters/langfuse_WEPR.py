from artefactual.preprocessing import parse_top_logprobs
from langfuse import Langfuse
from artefactual.scoring import WEPR

class Artefactual_WEPR:
    def __init__(self, name: str, langfuse_client: Langfuse, pretrained_model_name_or_path: str):
        self.name = name
        self.langfuse = langfuse_client
        self.wepr = WEPR(pretrained_model_name_or_path)

    def score_trace(self, trace_id: str) -> float:
        trace = self.langfuse.api.trace.get(trace_id)
        
        parsed_logprobs = parse_top_logprobs(trace.output)
        
        value = self.wepr.compute(parsed_logprobs)[0]
        
        self.langfuse.create_score(
            trace_id=trace_id,
            name=self.name,
            value=value
        )
        return value
