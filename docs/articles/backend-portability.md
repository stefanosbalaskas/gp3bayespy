# Backend Portability and Installation

> Python-facing port of `backend-portability.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

The advanced extension supports the two backends officially exposed by brms:
`rstan` and `cmdstanr`. The model family, formula, priors, and algorithm remain
restricted by gp3bayes. Only the implementation backend is selectable.

## Audit installed components

## CmdStanR setup

Install CmdStanR from the Stan R-universe repository:

Then check the C++ toolchain and install CmdStan:

The gp3bayes installer never installs or repairs CmdStan automatically unless
that explicit option is enabled.

## Full MCMC only

Variational inference, Pathfinder, Laplace approximation, arbitrary Stan code,
and arbitrary backend arguments remain outside this wrapper.

## Python API mapping

- `gp3bayespy.bayesian_backend_capabilities`
- `gp3bayespy.check_cmdstan_backend`
- `gp3bayespy.fit_duration_model_backend`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/backend_status.py`.
