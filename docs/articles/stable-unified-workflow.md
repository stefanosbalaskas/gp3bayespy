# A Stable Unified Workflow API

> Python-facing port of `stable-unified-workflow.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Why a unified API?

The family-specific `gp3bayespy` functions remain the authoritative low-level
interfaces. Version 0.2.0 adds a small family-neutral layer so an analysis
pipeline can use the same verbs after a binary or duration model has been
fitted. The wrappers dispatch only inside the two approved model families.
They do not accept arbitrary formulas, likelihoods, Stan programs, or fitting
algorithms.

The stable verbs are:

- `diagnose_model_fit()` for numerical sampling diagnostics;
- `summarise_model_posterior()` for family-specific posterior summaries;
- `check_model_ppc()` for family-specific posterior predictive checks;
- `estimate_model_estimands()` for the approved standardized estimands;
- `validate_gp3bayes_object()` for structural object checks; and
- `model_workflow_status()` for a descriptive stage map.

## Build a backend-independent specification

Structural validation is deliberately different from statistical validation:

## Inspect workflow progress

The stage map says what objects are present. It does **not** say the analysis is
adequate, robust, causal, or complete.

## Fit through either approved backend

Full MCMC is optional and intentionally not executed while this vignette is
built.

After fitting, the same verbs work for either approved family:

## What the unified layer does not do

A stable API is not a license to automate scientific judgment. In particular,
these wrappers do not automatically select a model, delete observations,
change a random-effects structure, declare posterior adequacy, or translate an
association into a causal effect. Those boundaries remain explicit throughout
0.2.0.

## Python API mapping

- `gp3bayespy.check_model_ppc`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.diagnose_model_fit`
- `gp3bayespy.estimate_model_estimands`
- `gp3bayespy.fit_binary_model_backend`
- `gp3bayespy.model_workflow_status`
- `gp3bayespy.plot_sampling_diagnostics`
- `gp3bayespy.prepare_hierarchical_binary_data`
- `gp3bayespy.simulate_hierarchical_binary_data`
- `gp3bayespy.specify_binary_model`
- `gp3bayespy.summarise_model_posterior`
- `gp3bayespy.validate_gp3bayes_object`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
