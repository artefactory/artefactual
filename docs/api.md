# API reference

The public surface of `artefactual`, grouped by role: estimators, transformers, parsing and
utilities. Detectors are built from
`artefactual.EPR` and `artefactual.WEPR`, which score responses, and `artefactual.EPRClassifier`
and `artefactual.WEPRClassifier`, which score logprob arrays; `artefactual.PoolingDetector` takes any `Pooling` or pooling function; the other sections are the
parts they are made of and the helpers around them.

## Estimators

```{eval-rst}
.. currentmodule:: artefactual

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   EPR
   WEPR
   EPRClassifier
   WEPRClassifier
   scoring.PoolingDetector
```

## Transformers

```{eval-rst}
.. currentmodule:: artefactual

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   scoring.EntropyTransformer

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   scoring.epr_reduction
   scoring.wepr_reduction

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   scoring.Pooling
   scoring.EPRPooling
   scoring.WEPRPooling
   scoring.FunctionPooling

Subclass `scoring.Pooling` for a pooling of your own, or pass a plain function
`f(x, axis)`; either goes to `scoring.EntropyTransformer` or `scoring.PoolingDetector`.
```

## Parsing

`LogProbParser`, readers and validated models for the response shapes it accepts.

```{eval-rst}
.. currentmodule:: artefactual.preprocessing

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   LogProbParser

.. autosummary::
   :toctree: _autosummary
   :nosignatures:

   index_by_custom_id
   parse_sampled_token_logprobs
   parse_top_logprobs
   read_batch
   read_judgment
   read_message

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   BatchRequestOutput
   BatchResponseData
   ChatChoice
   ChatCompletion
   ChatMessage
   Judgment
   ResponsesPayload
```

## Utilities

Base classes and mixins to build on, and the Langfuse integration.

```{eval-rst}
.. currentmodule:: artefactual

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   scoring.BaseDetector
   scoring.BaseEntropyClassifier
   scoring.EntropyContributionsMixin
```

### Integrations

Needs the `adapters` extra: `pip install "artefactual[adapters]"`.

```{eval-rst}
.. currentmodule:: artefactual.adapters.langfuse.evaluator

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   HallucinationEvaluator
```

### Exceptions and warnings

```{eval-rst}
.. currentmodule:: artefactual.exceptions

.. autosummary::
   :toctree: _autosummary
   :template: autosummary/class.rst
   :nosignatures:

   ArtefactualError
   ArtefactualWarning
   EmptySequenceWarning
```
