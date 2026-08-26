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
