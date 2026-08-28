# Backend Reliability, Parity and Object Schemas

> Python-facing port of `backend-reliability.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Two interchangeable implementation backends, one modeling contract

The approved full-MCMC interface remains `brms` with either `rstan` or
`cmdstanr`. Backend portability should preserve the model family, formula,
priors, estimand and sampling contract. It should **not** imply identical
random-number streams or identical posterior draws.

An optional compiler smoke test can be requested explicitly and is not run in
this vignette:

## Posterior-summary parity

Parity is evaluated relative to Monte Carlo uncertainty rather than exact draw
identity. The data-frame interface below makes the rule transparent and is also
useful for archived summary comparisons.

With real fits, the same function obtains posterior summaries from each fit:

## Object-schema compatibility

A stable release also needs to know when serialized object structure changes.
Schema capture records structure rather than values.

Freezing does not write anything unless a path is explicitly provided:

A schema match is a compatibility check only. It says nothing about numerical
identity, statistical adequacy, or scientific validity.

## Python API mapping

- `gp3bayespy.audit_backend_parity`
- `gp3bayespy.backend_capabilities`
- `gp3bayespy.capture_gp3bayes_schema`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.freeze_gp3bayes_schema`
- `gp3bayespy.read_gp3bayes_schema`
- `gp3bayespy.validate_backend_environment`
- `gp3bayespy.validate_gp3bayes_schema`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/backend_status.py`.
