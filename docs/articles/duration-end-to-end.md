# End-to-End Hierarchical Lognormal Duration Workflow

> Python-facing port of `duration-end-to-end.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Scope

The duration workflow is restricted to strictly positive, finite, uncensored
durations modeled with a hierarchical lognormal likelihood. Zero, negative,
censored, truncated, shifted-lognormal, Gamma, Weibull, survival, and mixture
outcomes are outside this contract.

## Simulate durations with stored truth

The fixed effects and grouping scales are on the log-duration scale. The
baseline is stored as a median in the declared outcome unit.

## Declare the duration contract

The outcome unit is mandatory. The package never guesses a unit from column
names or magnitudes.

## Prepare, convert, and audit

Explicit unit conversion uses `outcome_multiplier` and `converted_unit`
together. For example, converting milliseconds to seconds is recorded as:

## Specify priors and check prior implications

The prior check examines overall medians, upper tails, coefficients of
variation, and condition median ratios. A failure requests review and does not
automatically alter the priors.

## Translate and fit through the restricted backend

The likelihood, formula, priors, backend, and sampling algorithm are derived
from the approved package specification. There is no unrestricted formula or
backend argument.

## Diagnostics and posterior interpretation

Exponentiating a population-level coefficient gives a conditional median
duration ratio under the lognormal model. This ratio is not automatically a
causal effect.

The predictive check compares observed and replicated median, mean, upper-tail,
dispersion, condition-ratio, and grouping summaries. Passing those summaries
does not prove global model adequacy.

## Sensitivity, recovery, and reporting

Recovery results apply to the declared synthetic data-generating process.
Reports preserve the distinction between successful fitting, numerical
sampling diagnostics, predictive behavior, robustness checks, and substantive
interpretation.

## Python API mapping

- `gp3bayespy.assess_duration_prior_sensitivity`
- `gp3bayespy.check_duration_posterior_predictive`
- `gp3bayespy.check_duration_prior_predictive`
- `gp3bayespy.create_duration_model_report`
- `gp3bayespy.create_model_contract`
- `gp3bayespy.diagnose_duration_fit`
- `gp3bayespy.fit_duration_model`
- `gp3bayespy.prepare_hierarchical_duration_data`
- `gp3bayespy.run_duration_recovery`
- `gp3bayespy.simulate_hierarchical_duration_data`
- `gp3bayespy.specify_duration_model`
- `gp3bayespy.summarise_duration_posterior`
- `gp3bayespy.translate_duration_model_to_brms`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/duration_workflow.py`.
