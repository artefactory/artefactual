from collections.abc import Sequence

from vllm import RequestOutput

from artefactual.scoring.uncertainty import UncertaintyDetector


class WEPR(UncertaintyDetector):
    """Computes Weighted Entropy Production Rate (WEPR)."""

    def compute(self, outputs: Sequence[RequestOutput], weights: list[float], *, return_tokens: bool = False):
        """
        Computes a weighted average of per-token entropy sums.
        The weights could be based on token position, importance, etc.
        """
        # This method would also call self._entropy_contributions()
        # but would apply a weighting scheme before averaging.

        # Example logic:
        # seq_scores = []
        # for out in outputs:
        #     ...
        #     s_kj = self._entropy_contributions(token_logprobs)
        #     token_epr = s_kj.sum(axis=1)
        #
        #     # Apply weights
        #     if len(token_epr) != len(weights):
        #         raise ValueError("Length of weights must match number of tokens.")
        #
        #     weighted_seq_epr = np.average(token_epr, weights=weights)
        #     seq_scores.append(weighted_seq_epr)
        # ...
        pass
