# Fitting hierarchical pupil time-course models

> Python-facing port of `fitting-pupil-timecourse-models.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Approved family

The first direct pupil model family is Gaussian with an identity link. It
supports a governed temporal trajectory, optional condition-specific
trajectory, participant hierarchy, optional item hierarchy, declared numeric
nuisance covariates, and optional AR(1) dependence for sufficiently regular
within-trial sampling.

## Translation without fitting

Translation is restricted. The user does not provide an arbitrary formula,
family, Stan program, algorithm, or open-ended backend argument list.

## Prior-predictive gate

Prior-predictive execution is governed separately from posterior fitting. The
default call records the approved prior-only plan and does not compile Stan.

A researcher can set `execute = TRUE` with either approved backend during
manual analysis. The operation never changes priors automatically and its
evidence does not certify model adequacy.

## Full-MCMC backends

Real fitting is optional and requires `brms` plus one approved backend.

The wrappers preserve a common gp3bayes object shape. A fitted object does not
by itself establish convergence, adequacy, measurement validity, or a causal
interpretation.

## Python API mapping

- `gp3bayespy.check_pupil_prior_predictive`
- `gp3bayespy.create_pupil_contract`
- `gp3bayespy.fit_pupil_model_backend`
- `gp3bayespy.prepare_pupil_timecourse`
- `gp3bayespy.simulate_pupil_timecourse`
- `gp3bayespy.specify_pupil_timecourse_model`
- `gp3bayespy.translate_pupil_model_to_brms`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/pupil_workflow.py`.
