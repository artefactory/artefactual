---
file_format: mystnb
jupytext:
  notebook_metadata_filter: mystnb,file_format
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
mystnb:
  execution_mode: 'off'
---

# Build the training set from an endpoint, then train

You start with an endpoint and nothing else. This notebook builds the whole training set
for the model that endpoint serves — it asks your questions, keeps the `top_logprobs`
behind each response, has a judge grade them — then fits a WEPR detector and saves it.

A detector is trained for one model, on answers that model produced plus a verdict on
each. The wider context is in the guide's *Training a detector*.

Four sections. The first costs no requests and is still the expensive one: a gold answer
per question is written by someone. The two that do spend requests write their results to
disk, so a refit never pays for them twice.

| Section | What it does | Cost |
|---|---|---|
| Bring the questions | Your questions, each with a gold answer | the work: writing the gold answers |
| Generate the responses | One request each, keeping `top_logprobs` | N requests |
| Judge the responses | One request each, against the gold answer | N requests |
| Fit and evaluate | Fit, evaluate, save | seconds |

Both written files are the **OpenAI Batch output shape** — one JSON object per line wrapping
a completion under `custom_id` — which is what the Batch API returns and what
`scripts/train_detector.py` reads, so nothing downstream needs a conversion.

## Prerequisites

| Variable | What it is |
|---|---|
| `OPENAI_BASE_URL` | Any OpenAI-compatible endpoint returning `top_logprobs` |
| `OPENAI_API_KEY` | Its key |
| `OPENAI_MODEL` | The model being scored — the detector belongs to it, and the id has to be one your endpoint serves |

The endpoint must return at least `K` ranks per token. `top_logprobs` is commonly capped at
20, so `K = 15` fits; an endpoint that caps lower is refused by name when the responses are
generated, rather than training on narrower data.

This notebook is not executed when the documentation is built, so the numbers you see are
the ones your own run produces.

```{code-cell} ipython3
# From a clone: `uv sync --group notebooks`.
#
# On Colab, uncomment to install the package and fetch the questions this notebook reads.
# !pip install -q 'artefactual[adapters]' jinja2
# !wget -q https://raw.githubusercontent.com/artefactory/artefactual/main/docs/examples/questions_sample.json
```

## Configuration

Every knob in one place. `MODEL` has no default: a model id only means something to the
endpoint serving it, and a wrong one fails on every generation request rather than here.

`K` is part of the feature definition, not a batch size — WEPR fits one coefficient per
rank, so the detector is only ever loaded at the value it was fitted at.

The two prompts are the paper's. The generation prompt asks for short answers on purpose:
the detector reads the distribution behind the response, so a model that pads with hedging
spends its tokens on text carrying nothing to score.

```{code-cell} ipython3
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jinja2 import Template

# No default: a model id only means something to the endpoint serving it, and a wrong one
# fails on every generation request rather than here.
MODEL = os.environ["OPENAI_MODEL"]

# Ranks kept per token. Part of the feature definition, not a batch size: WEPR fits one
# coefficient per rank, so a detector is only ever used at the k it was fitted at. Every
# published detector uses 15.
K = 15
# The API caps it at 20, and an endpoint asked for more rejects every generation request --
# which arrives as "no answers were generated" three cells later, pointing at the wrong
# thing.
assert 1 <= K <= 20, "top_logprobs must be between 1 and 20"

SEED = 42
WORKERS = 8

RESPONSES = Path("responses.jsonl")
JUDGMENTS = Path("judgments.jsonl")
# The generation prompt, the paper's (4.1.2). Short answers on purpose: the detector reads
# the token distribution behind the answer, so a model that pads with hedging spends its
# tokens on text that carries nothing to score.
GENERATE = Template("""You are a useful assistant that help finding short and precise answers for a given query or question.
            Please keep your output AS SHORT AND CONCISE AS POSSIBLE.
            Here is the query :
            {{ query }}
            """)

# The judging prompt, the paper's, in full. Rendered with jinja, as the original is: the
# reply format it demands is itself a JSON object, and jinja passes a bare `{` through
# where `str.format` would read it as a field, as it would a question containing one.
JUDGE = Template("""You are an expert evaluator tasked with determining if two answers convey compatible information. Your task is to make a binary True/False judgment on whether the answers are SEMANTICALLY COMPATIBLE.

Query:
{{ query }}

Expected Answer:
{{ expected_answer }}
{% if answer_aliases %}
Answer Aliases (Additional Correct Answers):
{% for alias in answer_aliases %}
- {{ alias }}
{% endfor %}
{% endif %}

Generated Answer:
{{ generated_answer }}

CRITICAL INSTRUCTIONS:
1. FIRST, perform a simple VERBATIM TEXT COMPARISON:
   - If the generated answer is IDENTICAL (exact same text) to EITHER the expected answer OR ANY of the answer aliases, your judgment MUST be TRUE
   - If not identical to any of them, proceed to semantic comparison

2. For SEMANTIC COMPARISON, use these MANDATORY RULES:
   - Judge "True" if the generated answer matches the SEMANTIC MEANING of EITHER the expected answer OR ANY of the answer aliases
   - Judge "True" WHENEVER the general meaning or core concept is the same as either the expected answer or any alias
   - Judge "True" if one answer is GENERAL and one is SPECIFIC about the same thing
   - Judge "True" if one answer names a CATEGORY (e.g., "missionaries") and the other provides SPECIFIC INSTANCES of that category (e.g., "Augustine was sent by Pope Gregory")
   - Judge "True" if one answer gives a BRIEF fact and the other ELABORATES with more details
   - Judge "True" if one answer is more detailed but does NOT contradict the other
   - Judge "False" ONLY if the answers directly CONTRADICT all of the expected answer and all aliases, or discuss ENTIRELY different topics

3. EXTREMELY IMPORTANT RULES ABOUT SPECIFICITY:
   - When one answer is general and one is specific → TRUE
   - When one uses a category term and one gives examples → TRUE
   - When one gives "who/what" and the other adds "when/where/how/why" → TRUE
   - When one gives a person's role and the other gives their name → TRUE
   - When one refers to a group and the other names individuals → TRUE

4. Always check if the specific answer is an INSTANCE or EXAMPLE of the general answer
   - If it is, the judgment MUST be TRUE regardless of how detailed the specific answer is

5. The query is provided ONLY for context - do NOT use it in your judgment

6. IMPORTANT: The generated answer should be considered TRUE if it matches EITHER the expected answer OR ANY of the answer aliases in meaning

FINAL CHECK BEFORE SUBMITTING:
- If the generated answer could reasonably be considered matching ANY of the expected answer or aliases → TRUE
- If after reading all answers, they feel like they're talking about the same basic concept → TRUE
- If you think "the generated answer is not contradicting the expected answer or any of its aliases" → TRUE

Your response MUST follow this format:
{
  "judgment": true/false,
  "explanation": "One clear sentence explaining why the answers are compatible or contradictory."
}""")
```

## Bring the questions

**This is the cell to replace.** `questions_sample.json` sits beside this notebook: a
hundred TriviaQA rows, there so the notebook runs end to end out of the box. The detector
you actually want is trained on the questions your users ask, because it learns how *your*
model behaves on the traffic it will meet. The paper measured that gap: its numbers drop
10-20 points when a TriviaQA-trained detector is pointed at WebQuestions.

Two fields per question, plus two optional ones: `question` is asked, `short_answer` is what
a correct response has to agree with, `answer_aliases` lists other wordings the judge should
also accept, and `question_id` names the row if the file already carries one.

Two properties decide whether a question set works. Responses must be **short enough for a
judge to grade**, and the model must get **enough of them wrong** that both classes appear.
A hundred is the working size; at twenty-five a capable model often gets nothing wrong, and
the run spends the requests before refusing to fit.

For a set at the paper's scale, `scripts/ecir/build_questions.sh` draws one from the Hub in
these same fields -- `./build_questions.sh triviaqa 500 > questions.json`, or `webquestions`
for the out-of-domain set -- and `QUESTIONS_FILE` below becomes `Path("questions.json")`.

`question_id` is what travels: it becomes `custom_id` on the responses and the verdicts, and
that is what every later join pairs on. A duplicate would pair an answer with another
question's gold answer -- a wrong label rather than an error -- so a file that names its own
rows is checked for collisions, and a file that does not gets ids from position, which
cannot collide.

```{code-cell} ipython3
QUESTIONS_FILE = Path("questions_sample.json")

questions = [
    {
        "question_id": entry.get("question_id") or f"q{position:03d}",
        "question": entry["question"],
        "short_answer": entry["short_answer"],
        "answer_aliases": entry.get("answer_aliases") or [],
    }
    for position, entry in enumerate(json.loads(QUESTIONS_FILE.read_text(encoding="utf-8")))
]

repeated = [qid for qid, seen in Counter(q["question_id"] for q in questions).items() if seen > 1]
assert not repeated, (
    f"{len(repeated)} question_id(s) appear more than once, e.g. {repeated[0]!r}. Every "
    f"later join pairs on this id, so a duplicate would grade one answer against another question."
)

incomplete = [q for q in questions if not q["question"] or not q["short_answer"]]
assert not incomplete, (
    f"{len(incomplete)} question(s) are missing `question` or `short_answer`, e.g. "
    f"{incomplete[0]}. Both are needed: one is asked, the other is what the judge grades against."
)

print(f"{len(questions)} questions")
print(json.dumps(questions[0], indent=2, ensure_ascii=False))
```

## Generate the responses, keeping the log-probabilities

One request per question, `WORKERS` at a time, with `logprobs=True` and `top_logprobs=K`.
That distribution is the entire input to the detector; the response text is only ever read
to judge it.

Each result is written as it arrives rather than after the pool finishes. `generate` returns
the API's failures instead of raising them, but the SDK does not wrap every transport
pathology — a 502 whose HTML body arrives under a JSON content type raises `JSONDecodeError`
inside the worker — and that would come out of `pool.map` and discard every response already
paid for.

The file is the OpenAI Batch output shape, so `scripts/train_detector.py` reads it without
knowing what produced it -- this notebook, a Batch job, or an offline runner.

The cell after it runs `LogProbParser` over what came back. That is the pipeline's own first
step, and it is where an endpoint that accepted `logprobs=True` and ignored it, or that
capped the ranks below `K`, is refused by name — before the judge spends another N requests
judging responses that cannot be trained on.

```{code-cell} ipython3
from openai import OpenAI, OpenAIError

from artefactual.preprocessing import BatchRequestOutput, BatchResponseData

# `max_retries` above the SDK's default of 2: this fires N requests at once, and a burst
# of 429s that exhausts the retries becomes a thinner dataset rather than an error.
client = OpenAI(max_retries=6)  # reads OPENAI_BASE_URL and OPENAI_API_KEY

# Which endpoint this is actually talking to. Unset, OPENAI_BASE_URL silently means
# api.openai.com, and a self-hosted run then fails N times with an authentication error.
print(f"endpoint: {client.base_url}")


def generate(question):
    try:
        return client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": GENERATE.render(query=question["question"])}],
            logprobs=True,
            top_logprobs=K,
            temperature=1.0,
            top_p=1.0,
            # The paper also samples with top_k=50. OpenAI's API has no such parameter, so
            # only a self-hosted server can be asked for it, through `extra_body`.
            max_completion_tokens=200,
        )
    except OpenAIError as error:
        # Every failure this call can produce -- transport, timeout, rate limit, a
        # rejected request -- is an OpenAIError, and one of them should not cost the
        # run. Anything else is a bug in the code above and should not be caught here.
        return error


with RESPONSES.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=WORKERS) as pool:
    # Written as each result arrives, not after the pool finishes. `generate` returns the
    # API's failures rather than raising them, but the SDK does not wrap every transport
    # pathology -- a 502 whose HTML body arrives under a JSON content type raises
    # JSONDecodeError from inside the worker -- and that would come out of `pool.map` and
    # discard every answer already paid for. This way the file holds what was generated up
    # to the failure, which is the whole argument for writing it out at all.
    generated = []
    for question, result in zip(questions, pool.map(generate, questions), strict=True):
        generated.append(result)
        failed = isinstance(result, Exception)
        # The envelope is `BatchRequestOutput`, the model `read_batch` validates with, so
        # the writer and the reader share one definition of it. The body stays the API's
        # own payload: it carries the token text and the log-probabilities, and narrowing
        # it to what the parser reads would throw the rest away.
        # `id` is left unset: the Batch API assigns it (a `batch_req_...` value), and a
        # line written outside a batch run has no such id to carry. Nothing joins on it --
        # `custom_id` is the key -- so an invented one would be provenance that is not true.
        line = BatchRequestOutput(
            custom_id=question["question_id"],
            response=None if failed else BatchResponseData(status_code=200, body=result.model_dump()),
            # The class name, not the provider's text: an authentication error quotes the
            # key it rejected, and this file is one you hand onward. The full message is
            # printed below, where it stays in the session.
            error={"message": type(result).__name__} if failed else None,
        )
        # `json.dumps` rather than `model_dump_json`, for `ensure_ascii`: every reader of
        # these files splits them with `splitlines()`, which breaks on U+2028, U+2029 and
        # U+0085 -- characters JSON does not require escaping and a model can emit.
        out.write(json.dumps(line.model_dump(), ensure_ascii=True) + "\n")
ok = [(q, r) for q, r in zip(questions, generated) if not isinstance(r, Exception)]
print(f"wrote {RESPONSES}, {len(ok)}/{len(generated)} generated")
for question, result in zip(questions, generated):
    if isinstance(result, Exception):
        print(f"  failed: {question['question_id']}: {result}")
```

```{code-cell} ipython3
from artefactual.preprocessing import LogProbParser

# Nothing came back at all -- almost always OPENAI_BASE_URL, the key, or a model name the
# endpoint does not serve. Checked before indexing, because the errors above say what
# happened and a bare IndexError here would not.
assert ok, (
    "no responses were generated. Check OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL, "
    "and read the per-request errors above: an endpoint that rejects `top_logprobs`, "
    "`temperature` or `max_completion_tokens` fails every request the same way."
)

# The pipeline's own first step, run here rather than at fit time. It owns the rank axis, so
# it is what refuses a response carrying no log-probabilities or fewer than K ranks on any
# token -- and it says which, by name. Running it now costs one pass and saves N judge
# requests spent on responses that cannot be trained on.
logprobs = LogProbParser(k=K).transform([completion for _, completion in ok])

logprobs.shape
```

## Judge the responses

A second request per response, at `temperature=0`, asking the model to compare what it said
against the gold answer and reply with `{"judgment": true|false, "explanation": "..."}`.

The reply is written to disk whole, so each verdict keeps the sentence explaining it.

`read_judgment` turns that reply into the label. Judges wrap the object in prose or a
Markdown fence often enough that a bare `json.loads` is unsafe, and a stringified `"false"`
must not be taken for a verdict — `bool("false")` is True, which would mark every wrong
response correct.

The class balance is printed with the count. All-correct means the questions were too easy
for this model, all-wrong usually means it is not answering in the short form the judge
expects.

```{code-cell} ipython3
def render_judge(question, completion):
    return JUDGE.render(
        query=question["question"],
        expected_answer=question["short_answer"],
        answer_aliases=question["answer_aliases"],
        generated_answer=completion.choices[0].message.content or "",
    )


print(render_judge(ok[0][0], ok[0][1])[:600])
```

```{code-cell} ipython3
from artefactual.preprocessing import read_judgment


def judge(pair):
    """The judge's reply, kept whole: the verdict is derived from it, not instead of it."""
    question, completion = pair
    try:
        return client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": render_judge(question, completion)}],
            temperature=0,
            max_completion_tokens=200,
        )
    except OpenAIError as error:
        # Every failure this call can produce -- transport, timeout, rate limit, a
        # rejected request -- is an OpenAIError, and one of them should not cost the
        # run. Anything else is a bug in the code above and should not be caught here.
        return error


with JUDGMENTS.open("w", encoding="utf-8") as out, ThreadPoolExecutor(max_workers=WORKERS) as pool:
    # Written as each reply arrives, for the same reason the generation cell is: a failure
    # the SDK does
    # not wrap comes out of `pool.map`, and the verdicts already paid for should survive it.
    # The verdict stays as the judge wrote it; nothing is distilled out.
    verdicts = []
    for (question, _), verdict in zip(ok, pool.map(judge, ok), strict=True):
        verdicts.append(verdict)
        failed = isinstance(verdict, Exception)
        line = BatchRequestOutput(
            custom_id=question["question_id"],
            response=None if failed else BatchResponseData(status_code=200, body=verdict.model_dump()),
            error={"message": type(verdict).__name__} if failed else None,
        )
        out.write(json.dumps(line.model_dump(), ensure_ascii=True) + "\n")

# `read_judgment` returns True when the judge said the response was CORRECT, the opposite of
# the class the detector predicts, so the label is its negation. It reads the fenced and
# prose-wrapped replies a model actually returns, and answers None for a reply -- or a failed
# request -- carrying no verdict at all.
judgments = [
    (question, completion, int(not verdict))
    for (question, completion), reply in zip(ok, verdicts, strict=True)
    if not isinstance(reply, Exception) and (verdict := read_judgment(reply)) is not None
]

assert judgments, (
    "no verdict could be read from any reply. The judge answers in JSON; a model that "
    "cannot hold that format needs a different one, or a larger max_completion_tokens."
)

dropped = len(verdicts) - len(judgments)
hallucinated = sum(flag for _, _, flag in judgments)
print(f"wrote {JUDGMENTS}, {len(verdicts)} replies" + (f", {dropped} failed or unreadable" if dropped else ""))
print(f"  {len(judgments)} labelled, {hallucinated} hallucinations ({hallucinated / len(judgments):.0%})")

# A reply cut off at `max_completion_tokens` is invalid JSON, so it lands in `dropped` with
# nothing saying why. The judge answers in JSON *and* explains itself, so this is the cap
# that runs out first.
truncated = sum(1 for v in verdicts if not isinstance(v, Exception) and v.choices[0].finish_reason == "length")
if truncated:
    print(f"  {truncated} reply(ies) hit max_completion_tokens; raise it and rerun this cell")
for question, completion, flag in judgments[:2]:
    said = (completion.choices[0].message.content or "").strip()
    print(f"  [{'hallucination' if flag else 'grounded'}] said {said[:40]!r} (gold: {question['short_answer']!r})")
```

## Fit and evaluate

This step reads the two files back rather than using what is still in memory, so a refit at
another `k` costs no requests. `read_batch` is the reader, so a file written by anything
else parses the same way.

`WEPR()` returns the unfitted pipeline, which takes the batch lines directly: the
parser opens the envelope itself, so there is no feature extraction to write.

**ROC-AUC** scores the ranking — whether hallucinations sort above grounded responses —
which is what matters if you triage by score. The **classification report** scores the
decisions at a 0.5 cut, where recall on the `hallucination` row is the fraction actually
caught. Only the AUC carries over to another threshold, so pick one from the held-out scores
rather than assuming 0.5.

```{code-cell} ipython3
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from artefactual.preprocessing import index_by_custom_id, read_batch, read_judgment, read_message
from artefactual.scoring import WEPR, BaseDetector

# Read back from disk rather than from the variables above: this is the path a fresh kernel
# takes, and the one anything else reading these files takes too. `read_batch` validates
# each line and refuses a repeated `custom_id`, which would pair a response with another
# question's verdict.
generated = index_by_custom_id(read_batch(RESPONSES))
labelled = [
    (generated[row.custom_id], int(not verdict))
    for row in read_batch(JUDGMENTS)
    if row.custom_id in generated and (verdict := read_judgment(row.completion)) is not None
]

responses = [row for row, _ in labelled]
y = np.array([label for _, label in labelled])
print(f"read {len(responses)} labelled responses back from {RESPONSES.name} and {JUDGMENTS.name}")

x_train, x_test, y_train, y_test = train_test_split(responses, y, test_size=0.25, stratify=y, random_state=SEED)
detector = WEPR(k=K).fit(x_train, y_train)
print(f"fitted on {len(y_train)}, holding out {len(y_test)}")

scores = detector.predict_proba(x_test)[:, 1]
print(f"\nROC-AUC: {roc_auc_score(y_test, scores):.2f}")
print(classification_report(y_test, scores >= 0.5, target_names=["grounded", "hallucination"], zero_division=0))
```

## Audit the labels the judge produced

A detector fitted on mislabelled rows inherits them. The judge says whether the answer
matches the gold one; the label is the negation of that, so a mismatch is `hallucination`,
the class the detector is fitted to predict. Both columns below are on that axis.

`cross_val_predict` gives every response a score from a fold that did not contain it. The
rows where the detector is most confident and disagrees with the label are the ones to read
first, printed with the question, the gold answer and the judge's own explanation.

```{code-cell} ipython3
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# What each verdict was made against, and the sentence the judge gave for it.
asked = {question["question_id"]: question for question in questions}
explanations = {
    row.custom_id: json.loads(read_message(row.completion) or "{}").get("explanation", "")
    for row in read_batch(JUDGMENTS)
    if read_message(row.completion)
}

folds = StratifiedKFold(5, shuffle=True, random_state=SEED)
# Out-of-fold: every score comes from a detector that never saw that response.
out_of_fold = cross_val_predict(WEPR(k=K), responses, y, cv=folds, method="predict_proba")[:, 1]

scored = zip(out_of_fold, y, responses, strict=True)
disagreements = sorted(scored, key=lambda row: abs(row[0] - row[1]), reverse=True)[:5]

print("the five rows the detector disagrees with the label about most:\n")
for score, label, row in disagreements:
    question = asked[row.custom_id]
    # `label` is the judge's verdict already negated, so both columns say hallucination.
    print(f"  labelled {'hallucination' if label else 'grounded'}, detector P(hallucination)={score:.3f}")
    print(f"    Q     {question['question']}")
    print(f"    said  {(read_message(row.completion) or '').strip()!r}")
    print(f"    gold  {question['short_answer']!r}")
    print(f"    why   {explanations.get(row.custom_id, '')!r}\n")
```

## Save it, and load it back

`.skops` rather than a pickle. `from_pretrained` takes a `.skops` file, a directory holding
`model.skops`, or **a Hugging Face repository id** — the shipped detectors are loaded that
way, and yours is too once you push its `model.skops` to a repository of your own.

`k` is part of the weights, not a runtime option: the coefficients were fitted at one rank
count and mean nothing at another, so loading at a different `k` raises rather than
mis-shaping the score.

```{code-cell} ipython3
path = detector.save_estimator("wepr-generated.skops")
reloaded = WEPR.from_pretrained(path, k=K)

# Held-out responses: the rows the fit above never saw.
for row, label in list(zip(x_test, y_test, strict=True))[:5]:
    said = (read_message(row.completion) or "")[:40]
    print(f"[{'hallucination' if label else 'grounded    '}] P={reloaded.predict_proba(row)[0, 1]:.3f}  {said!r}")
```

## Where to go next

- **Refit without regenerating.** The two files are the expensive part: a different `k`,
  `epr` instead of `wepr`, or relabelled verdicts all reuse them, and none of it costs
  another request.
- **Feed the CLI instead.** This run's two Batch files go straight into the
  `train_detector.py` script in
  [the repository](https://github.com/artefactory/artefactual/blob/main/scripts/train_detector.py),
  which also reports the bootstrap confidence intervals the fit above does not — worth
  having, because a holdout this size cannot pin a score down on its own.
- **More questions.** The generation is the cost and the fit is seconds, so point
  `QUESTIONS_FILE` at a longer set rather than economising on labels.
- **At thousands of questions**, stop making one request per response. Batch submission
  takes a JSONL of requests and returns the JSONL these files already are, at roughly half
  the price: OpenAI's [Batch API](https://platform.openai.com/docs/api-reference/batch) if
  your provider hosts one, or an offline batch runner against a self-hosted server. Only
  the two generating cells change -- what they write, and everything after them, stays as
  it is.
- **Your own questions.** Only the questions cell changes, and it is a list. Training data
  should resemble the traffic being scored: the paper's numbers drop 10-20 points when a
  TriviaQA-trained detector meets WebQuestions.
