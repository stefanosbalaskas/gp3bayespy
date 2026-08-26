# Implemented API

The pre-alpha foundation exposes the contract/specification gate, hierarchical
binary and duration workflow foundations, and parity-inspection helpers.

## Core contract and specification

::: gp3bayespy.create_model_contract

::: gp3bayespy.audit_model_readiness

::: gp3bayespy.build_model_formula

::: gp3bayespy.create_prior_specification

::: gp3bayespy.validate_prior_specification

::: gp3bayespy.create_model_specification

## Hierarchical binary foundation

::: gp3bayespy.simulate_hierarchical_binary_data

::: gp3bayespy.prepare_hierarchical_binary_data

::: gp3bayespy.specify_binary_model

::: gp3bayespy.check_binary_prior_predictive

## Hierarchical duration foundation

::: gp3bayespy.simulate_hierarchical_duration_data

::: gp3bayespy.prepare_hierarchical_duration_data

::: gp3bayespy.specify_duration_model

::: gp3bayespy.check_duration_prior_predictive

## Restricted fitting bridge

The frozen R 0.5.0 API names are retained for migration parity. In Python,
`translate_*_to_brms()` returns a restricted backend plan that records the R
`brms`/`rstan` source contract but truthfully targets optional PyMC/NUTS. The
formula, family, prior classes, and fitting controls remain locked; arbitrary
backend arguments are not accepted. A returned fit records that sampling
occurred but does **not** claim convergence or posterior adequacy.

A real optional-backend smoke is available at `validation/smoke_fitting_backend.py`
after installing the `bayes` extra.

::: gp3bayespy.translate_binary_model_to_brms

::: gp3bayespy.fit_binary_model

::: gp3bayespy.translate_duration_model_to_brms

::: gp3bayespy.fit_duration_model

### Fitting result objects

::: gp3bayespy.BinaryBackendSpecification

::: gp3bayespy.BinaryFit

::: gp3bayespy.DurationBackendSpecification

::: gp3bayespy.DurationFit

## Backend inspection

::: gp3bayespy.backend_capabilities

## Posterior extraction, summaries, and sampling diagnostics

The posterior layer retains the R 0.5.0 public names while adapting the
`brms`/`posterior` containers to PyMC/ArviZ-native Python structures. Posterior
parameter names are canonicalized back to the R-facing `b_*`, `sd_*`, `cor_*`,
and `sigma` conventions where the fitted design supplies an unambiguous
mapping. R `quantile(type = 8)` intervals use NumPy's `median_unbiased` method.

`extract_posterior_draws(..., format="array")` returns an xarray DataArray;
`"matrix"` and `"df"` return pandas DataFrames; and the R-only `rvars` format
is intentionally adapted to a mapping of variable names to chain-by-draw NumPy
arrays.

Diagnostic status reports prespecified numerical thresholds only. Neither a
`pass` status nor a posterior summary automatically establishes convergence,
posterior adequacy, causal identification, or substantive validity.

::: gp3bayespy.extract_posterior_draws

::: gp3bayespy.diagnose_binary_fit

::: gp3bayespy.summarise_binary_posterior

::: gp3bayespy.diagnose_duration_fit

::: gp3bayespy.summarise_duration_posterior

## Governed predictive foundation

The predictive layer ports the first frozen R 0.5.0 `prediction-support.R`
contracts onto retained PyMC posterior draws. Expected responses, new-outcome
posterior predictive draws, linear-predictor draws, and the lognormal
duration median remain distinct quantities. Support auditing reports numeric
extrapolation, novel categorical/group levels, and missing required variables,
but never drops or rejects prediction rows automatically.

Population-level predictions exclude fitted participant/item effects by
default. Group effects can be requested explicitly; unseen grouping levels
require `allow_new_levels=True`. Predictions are descriptive posterior
quantities only and do not establish causal effects or out-of-sample adequacy.

::: gp3bayespy.create_prediction_grid

::: gp3bayespy.audit_prediction_support

::: gp3bayespy.prediction_support_table

::: gp3bayespy.predict_model

::: gp3bayespy.prediction_table

::: gp3bayespy.extract_expected_predictions

::: gp3bayespy.extract_posterior_predictions

::: gp3bayespy.extract_linear_predictions

::: gp3bayespy.predict_binary_probability

::: gp3bayespy.predict_duration

### Prediction result objects

::: gp3bayespy.PredictionSupport

::: gp3bayespy.Prediction

