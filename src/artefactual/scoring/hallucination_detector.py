from sklearn.base import ClassifierMixin

from artefactual.scoring import EPR, WEPR


class HallucinationDetector(ClassifierMixin):
    def predict_proba(self, parsed_logprobs) -> list[float]:
        raise NotImplementedError


class EPRDetector(HallucinationDetector):
    def __init__(self):
        self.epr = EPR()

    def predict_proba(self, parsed_logprobs) -> list[float]:
        return self.epr.compute(parsed_logprobs)


class WEPRDetector(HallucinationDetector):
    def __init__(self, pretrained_model_name_or_path):
        self.wepr = WEPR(pretrained_model_name_or_path)

    def predict_proba(self, parsed_logprobs) -> list[float]:
        return self.wepr.compute(parsed_logprobs)
