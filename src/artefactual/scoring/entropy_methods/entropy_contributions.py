"""Per-rank entropy contributions, the quantity both EPR and WEPR reduce."""

import numpy as np
from beartype import beartype

# Hints below are bare `np.ndarray`, not `NDArray[np.floating]`: beartype cannot resolve any
# NDArray subscript against numpy >= 2.5 and raises at decoration time, making the package
# unimportable. Re-subscripting them reintroduces that failure.


class EntropyContributionsMixin:
    """Provides `entropy_contributions` to the transformers that reduce it."""

    @staticmethod
    @beartype
    def entropy_contributions(logprobs: np.ndarray) -> np.ndarray:
        """Entropy contribution `s_kj = -p_kj * ln(p_kj)` for each rank of each token.

        The contribution peaks at `p = 1/e`, so it is not monotonic in rank: a mid-ranked
        candidate contributes more than either a near-certain top rank or a negligible
        tail one.

        Ranks are sorted descending before the conversion, so callers need not supply them
        in order. NaN sorts to the end and propagates, keeping padded positions padded.

        In nats, matching the ECIR2026 release the paper's results were produced with.
        The paper writes the entropy in bits (Eq. 1 and 3); the two differ by a factor of
        ln 2, which a fitted coefficient absorbs, so the released convention is the one to
        follow -- it is what the published weights were trained against.

        Args:
            logprobs: Natural-log probabilities, with the rank axis last. Any number of
                leading axes is allowed.

        Returns:
            Contributions in nats, same shape, NaN in the same positions.
        """

        if logprobs.size == 0:
            return np.empty_like(logprobs)

        # Enforce descending rank order along the rank axis.
        logprobs = -np.sort(-logprobs, axis=-1)

        # Convert to probabilities (logprobs are in natural log, base e)
        probs = np.exp(logprobs)

        # s = -p * ln(p), with the log term already supplied as a natural log
        with np.errstate(divide="ignore", invalid="ignore"):
            return -probs * logprobs
