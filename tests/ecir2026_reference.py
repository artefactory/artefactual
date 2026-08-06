"""Frozen copy of the scoring arithmetic from the ECIR2026 release.

This is the code that produced the numbers in *"Learned Hallucination Detection in
Black-Box LLMs using Token-level Entropy Production Rate"*, and it is therefore the
authority on what EPR and WEPR are — ahead of the paper's equations where the two differ,
because the published coefficients were trained against this.

Taken from tag `ECIR2026` (commit `66d05f5`) of this repository:

* `src/artefactual/scoring/entropy_methods/entropy_contributions.py`
    → `compute_entropy_contributions`
* `src/artefactual/scoring/entropy_methods/epr.py` → `EPR._compute_impl`
* `src/artefactual/scoring/entropy_methods/wepr.py` → `WEPR._compute_impl`

Reproduced here rather than imported, because that release is not installable alongside
the current one and a test must not depend on git at runtime. The arithmetic is verbatim;
only the surrounding plumbing (parsing, calibration loading, sigmoid) is dropped, since
what is being pinned is the feature extraction.

Do not "fix" or tidy anything in this file. Its value is that it does not change: if it
drifts toward the implementation it is meant to check, it stops being evidence. Any
divergence found against it is a question about the current code, not about this.
"""

import numpy as np

# --- verbatim from entropy_contributions.py @ ECIR2026 ---------------------------------


def compute_entropy_contributions(logprobs, k):
    """s_kj for the top-k ranks of each token, zero-padded or truncated to exactly k.

    Note the natural log, and that missing ranks become 0 rather than NaN.
    """
    logprobs = np.asarray(logprobs, dtype=np.float64)
    if logprobs.size == 0:
        return np.empty((0, k), dtype=np.float32)

    # Convert to probabilities (logprobs are in natural log, base e)
    probs = np.exp(logprobs)

    # Calculate entropy contributions: s = -p * log(p) = -exp(logp) * logp
    with np.errstate(divide="ignore", invalid="ignore"):
        s = -probs * logprobs
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    # Pad or truncate to k elements along the K dimension (axis=1)
    num_tokens, num_logprobs = s.shape
    if num_logprobs == k:
        return s

    s_kj = np.zeros((num_tokens, k), dtype=np.float32)
    if num_logprobs < k:
        s_kj[:, :num_logprobs] = s
    else:  # num_logprobs > k
        s_kj[:, :] = s[:, :k]
    return s_kj


# --- verbatim from epr.py @ ECIR2026 ---------------------------------------------------


def sequence_epr(sequence_logprobs, k):
    """EPR of one sequence: sum over rank K, then mean over the sequence."""
    s_kj = compute_entropy_contributions(sequence_logprobs, k)

    # sum over rank K (Token EPR)
    token_epr = np.sum(s_kj, axis=1)

    # Mean over sequence (Sequence EPR)
    return float(np.mean(token_epr)) if token_epr.size > 0 else 0.0


# --- verbatim from wepr.py @ ECIR2026 --------------------------------------------------


def sequence_wepr(sequence_logprobs, k, mean_weights, max_weights, intercept):
    """WEPR of one sequence (Eq. 8), before the sigmoid."""
    s_kj = compute_entropy_contributions(sequence_logprobs, k)

    # Token-level WEPR (S_beta): weighted sum across K using mean_weights
    token_wepr = s_kj @ mean_weights + intercept

    # 1. Average of token scores S_beta
    mean_term = np.mean(token_wepr)

    # 2. Weighted sum of max contributions per rank
    max_contributions_per_rank = np.max(s_kj, axis=0)
    max_term = max_contributions_per_rank @ max_weights

    return float(mean_term + max_term)
