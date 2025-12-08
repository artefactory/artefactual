# Design Document — Hallucination Scoring Pipeline (Updated)

**Scope:** Minimal, provider-agnostic pipeline to parse LLM responses, build features, and produce hallucination scores.  
**Status:** Aligned with the current parsing API (`OpenAIParsedCompletionV1`, `parse_openai_completion`, `parsed_token_logprobs`, `parsed_top_logprobs_tensor`).

---

## Components

1) **Parsing (OpenAI-style first)**
   - Dataclass: `OpenAIParsedCompletionV1` with `from_response(response)`.
   - Token info: `TokenInfo(token, logprob, top_logprobs)`.
   - Parsing function: `parse_openai_completion(response) -> OpenAIParsedCompletionV1`.
   - Utility: `parsed_token_logprobs(parsed_list) -> np.ndarray` (flattened token logprobs, skipping `None`).
   - Utility: `parsed_top_logprobs_tensor(parsed_list, top_k, max_tokens=None) -> np.ndarray` (3D tensor padded/truncated to top_k per token).
   - Note: APIs may return fewer than the requested top_k logprobs per token; padding handles short rows and truncation handles long rows.
   - Behavior: if multiple `choices` exist, we log a warning and parse only the first (consistent with `n=1`).

2) **Feature building (to be implemented)**
   - Inputs: parsed completions (`OpenAIParsedCompletionV1`), or arrays from `parsed_token_logprobs` / `parsed_top_logprobs_tensor`.
   - Feature ideas:
     - Sequence-level: length, sum/mean/max logprob, entropy over token logprobs.
     - Token-level: per-token logprob, entropy (if top-k available), logprob margin to runner-up.
     - Aggregations: mean/std/min/max over tokens; coverage flags (missing logprobs).
   - API sketch (future):
     - `build_features(parsed: list[OpenAIParsedCompletionV1]) -> np.ndarray`
     - `build_token_features(parsed: list[OpenAIParsedCompletionV1]) -> np.ndarray`

3) **Hallucination estimator (to be implemented)**
   - Input: feature arrays (sequence-level or token-level).
   - Output:
     - Sequence score: probability the response is non-hallucinatory.
     - Token scores (optional): per-token risk score.
   - Desired interfaces:
     - `fit(X, y)`, `predict_proba(X)` for sequence-level scoring.
     - `transform(parsed)` or `predict_token_scores(parsed)` for token-level scores (mapping back to token positions).
   - Candidate models: logistic regression / isotonic calibration on top of simple features; keep backend to scikit-learn.

---

## Public API (current)

```python
from artefactual.parsers import (
    OpenAIParsedCompletionV1,
    parse_openai_completion,
    parsed_token_logprobs,
)

pc = OpenAIParsedCompletionV1.from_response(raw_response)
arr = parsed_token_logprobs([pc])  # flattened token logprobs for quick heuristics
```

The rest (feature builders, estimators) will layer on top of `OpenAIParsedCompletionV1` and these parser utilities.

---

## Hallucination scoring workflow (planned)

1) Collect raw LLM responses (OpenAI-style JSON, including logprobs).
2) Parse: `parsed = [OpenAIParsedCompletionV1.from_response(r) for r in responses]`.
3) Build features:
   - Sequence-level: aggregate stats from `parsed_token_logprobs(parsed)` and from token structures (e.g., min/mean/std logprob, entropy).
   - Token-level: per-token logprob/entropy and margins.
4) Train/use estimator:
   - Sequence: `proba = estimator.predict_proba(seq_features)[:, 1]` → hallucination confidence.
   - Token: `token_scores = token_estimator.transform(parsed)` → per-token scores.
5) Thresholding / reporting: choose thresholds for binary flags; optionally return token heatmaps.

---

## Notes & rationale

- Focus on OpenAI-compatible schemas first (OpenAI, vLLM, LiteLLM). Claude/others lacking logprobs will produce empty token info and should be handled upstream by feature builders (e.g., mark missing logprob coverage).
- Parsing warns and selects the first choice to stay aligned with common `n=1` usage. If multi-choice handling is needed, we can extend the API later.
- The estimator layer will remain lightweight and sklearn-based; no heavy dependencies or provider coupling. Only rely on numpy arrays built from parsed objects.

---

## Estimator design (detailed)

Goals: sklearn-friendly estimators that can be trained on sequence features and can optionally emit token-level scores for inspection/debugging.

- **APIs**
  - Sequence: `fit(X, y)`, `predict_proba(X)[:, 1]` returns hallucination probability (1 = hallucination).
  - Token: `transform(parsed|X_token)` -> `(N, L)` token risk scores aligned to tokens (padding handled).
  - Minimal extras: `predict_token_scores(parsed)` convenience wrapper around `transform`.
- **Inputs**
  - Sequence features: dense 2D arrays from feature builders (mean/min/max/std over token logprobs/entropy, coverage flags).
  - Token features: 3D arrays (N, L, K) of per-token contributions (e.g., entropic contributions). Padding convention: zeros in the last axis imply padding.
- **Models**
  - Start with linear models: `LogisticRegression` or `SGDClassifier` to stay fast and interpretable.
  - Optional calibration (Platt/Isotonic) via `CalibratedClassifierCV` if metrics demand better probability quality.
- **Token-scoring strategy**
  - Interpret linear model weights in blocks that correspond to aggregated stats (e.g., mean/max). This enables projecting per-token contributions back into scores via a weighted harmonic mean.
  - Requires knowing `K` (features per token) and block count `M` (number of aggregators in the feature union).
  - Token scores bounded in [0, 1]; padding positions set to a sentinel (e.g., -1) or zeroed after scoring.
- **Metrics**
  - Offline: ROC-AUC/PR-AUC for sequence; token-level calibration curves or per-token AUROC if labeled data exists.
  - Runtime: track coverage (percent of tokens with logprobs) and fall back to neutral scores when coverage is low.
- **Persistence**
  - Estimators are pickleable via sklearn. Store alongside feature-builder configuration to guarantee shape compatibility.

---

## Implementation plan (estimator)

1) **Define token-aware transformers**  
   - Add `EntropicContributionTransformer` (3D logprobs -> entropic contributions) and `SequenceStatAggregator` (mean/max/min/sum over tokens).
2) **Implement token-scoring estimators**  
   - Create `TokenScoringMixin` that reshapes linear weights into blocks and computes harmonic-mean token risks. Provide `TokenScoringLogisticRegression`/`TokenScoringSGD` that set `K` and optional `component_weights`.
3) **Build sequence pipeline**  
   - Compose `Pipeline([('entropic_S', EntropicContributionTransformer), ('features', FeatureUnion([... aggregators ...])), ('clf', TokenScoringLogisticRegression)])`.
   - Ensure `get_feature_names_out` works end-to-end for inspection.
4) **Expose public API**  
   - Add helpers `predict_token_scores(parsed)` that parse -> build token features -> call estimator `transform`.
   - Document expected shapes, padding conventions, and component weights.
5) **Testing and examples**  
   - Unit tests for shape checks, padding handling, and deterministic token scores.
   - Example script demonstrating GridSearchCV, feature inspection, and token heatmap visualization.
