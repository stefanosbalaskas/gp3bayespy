# Troubleshooting multilevel mediation

## Preparation says there is no within-person exposure variation

Confirm whether the manipulation was actually within participant. If it was between participant, prepare with `require_within_x=False` and change the substantive interpretation. Do not inject artificial within variation.

## Lognormal/Gamma mediator fails because of zeros

Zeros are outside the support of those likelihoods. Determine whether zero means genuine non-inspection, measurement failure, or a preprocessing artifact. Do not add an arbitrary epsilon without a scientific rationale.

## Many rows are quality-ineligible

Return to the acquisition/preprocessing audit. Compare participants and conditions with high loss, document the quality rule, and run missingness-policy sensitivity. Do not silently delete the trials.

## Random slopes produce weak identification or convergence problems

Check whether the corresponding predictor varies sufficiently within participants and whether there are enough informative trials per participant. Random slopes are supported where estimable, not required by default.

## R-hat/ESS/divergences/tree-depth fail

Treat the indirect effect as blocked. Inspect parameterization, priors, family choice, influential participants/trials, and model complexity. Increasing iterations alone does not repair a structurally weak model.

## PyMC cannot import

The package raises `BackendUnavailableError`; it does not silently choose another estimator. Repair the optional Bayesian environment or use the corresponding supported R/brms implementation explicitly.
