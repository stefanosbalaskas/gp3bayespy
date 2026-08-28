# Advanced Optional Bayesian Workflows

> Python-facing port of `advanced-optional-workflows.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

This article describes the optional post-0.1.1 extensions. They remain
contract-first: neither unrestricted formulas nor automatic model selection
are introduced.

## Capability audit

## Separate interaction priors

The binary advanced default is `normal(0, 0.75)` for population main effects
and `normal(0, 0.50)` for the single approved interaction. The duration
advanced defaults are 0.35 and 0.25 respectively. These are candidate workflow
defaults and still require prior-predictive review.

## Full-MCMC backend selection

The advanced fitting functions accept only `rstan` or `cmdstanr`, and they
always use full MCMC sampling.

## Separation screening

The screen is a fixed-effects design diagnostic. It is not a replacement for
the hierarchical Bayesian fit or its posterior diagnostics.

## PSIS-LOO and model averaging

The comparison reports predictive differences and diagnostics but never
selects a model automatically.

## Power-scaling sensitivity

Low local sensitivity is not a proof of universal robustness.

## Simulation-based calibration

The brms generator and brms inference backend share implementation code.
An independently coded generator is preferable when the goal is to identify
shared implementation defects.

## Python API mapping

- `gp3bayespy.assess_powerscaled_sensitivity`
- `gp3bayespy.bayesian_backend_capabilities`
- `gp3bayespy.compare_psis_loo`
- `gp3bayespy.compute_loo_model_weights`
- `gp3bayespy.compute_psis_loo`
- `gp3bayespy.create_brms_sbc_plan`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.detect_binary_separation`
- `gp3bayespy.fit_binary_model_backend`
- `gp3bayespy.interaction_prior_summary`
- `gp3bayespy.prepare_hierarchical_binary_data`
- `gp3bayespy.run_sbc_plan`
- `gp3bayespy.simulate_hierarchical_binary_data`
- `gp3bayespy.specify_binary_model_with_interaction_prior`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```
