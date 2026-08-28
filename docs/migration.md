# Migration from gp3bayes 0.5.0

`gp3bayespy` preserves the frozen 458-export R gp3bayes 0.5.0 public contract while adapting object-oriented and backend behavior to Python.

## Naming

All frozen exported function names are available unchanged from the package root:

```python
import gp3bayespy as gp

gp.audit_model_readiness(...)
gp.predict_model(...)
gp.create_pupil_contract(...)
```

R S3 result objects are represented by typed Python dataclasses or explicit pandas tables. Table accessors remain available where the R package exposed them.

## Backends

The Python fitting bridge uses governed Python backends and analytic reference paths where documented. Backend availability is explicit through `backend_capabilities()` and `validate_backend_environment()`. The package never silently switches to unrestricted formulas, likelihoods, priors, or arbitrary model code.

## Quantiles and numerical parity

Frozen R quantile conventions are preserved where they affect public semantics. Existing parity fixtures identify type-7 versus type-8 summaries. Identical random-number streams between R and Python are not required; stochastic parity is distributional/semantic.

## Governance

Diagnostics, predictive checks, sensitivity analyses, rankings, LOO, pupil analyses, and model comparisons remain evidence-generating tools. They do not automatically certify adequacy, convergence, robustness, causality, cognitive state, emotional state, a preferred model, or automatic participant/group exclusion.

## Pupil workflows

The pupil API requires explicit measurement provenance, units, time units, and baseline declarations. Advanced, binocular, Gaussian-process, ARMA, measurement-error, missingness, response-shape, predictive-calibration, and model-comparison functions are exposed through the same root namespace.

## Executable examples

See [`examples/index.md`](examples/index.md) for runnable Python workflows covering the major package families.
