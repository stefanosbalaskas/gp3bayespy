# Prior Sensitivity and Simulation-Based Recovery

> Python-facing port of `sensitivity-and-recovery.Rmd` from the frozen R gp3bayes 0.5.0 reference. The statistical and governance framing below follows the canonical vignette; executable Python workflows use the mapped APIs listed later.

## Different validation questions

Prior sensitivity and parameter recovery answer different questions.

Prior sensitivity asks whether selected posterior summaries change materially
under prespecified defensible prior-scale changes.

Recovery asks whether the complete simulation, preparation, specification,
fitting, and summarization workflow can recover known generating values under a
declared synthetic design.

Neither procedure proves that a model is appropriate for every empirical data
set.

## Prior-scale sensitivity

Binary and duration sensitivity functions refit the same approved formula,
likelihood, backend, and sampling algorithm. Only declared prior scales are
multiplied.

The standardized shift is the absolute change in posterior median divided by
the reference posterior standard deviation. A pass applies only to the
declared multipliers. The object always records `robustness_claim = FALSE`.

## Simulation-based recovery

The recovery functions repeatedly:

1. generate deterministic synthetic data with stored truth;
2. create the approved model contract;
3. prepare and audit the data;
4. specify the approved priors;
5. fit through `brms` and `rstan`;
6. run the sampling diagnostic contract;
7. calculate bias, RMSE, interval coverage, and interval width.

## Minimum repetition rule

The default reporting contract requires at least 20 completed repetitions
before an overall recovery pass is possible. A smaller run can detect obvious
software or workflow failures, but its best possible status is `review`.

This rule prevents a two- or five-repetition smoke test from being described as
validation.

## Failure handling

With `continue_on_error = TRUE`, a failed repetition is retained in the
fit-status registry. It is not silently removed from the denominator. Repeated
fitting failures lower the diagnostic pass fraction and can force review or
failure.

## Interpretation

Recovery is conditional on:

- the selected data-generating parameters;
- sample size and grouping structure;
- the approved prior specification;
- the chosen MCMC settings;
- the interval probability;
- the declared recovery thresholds.

A successful recovery experiment is evidence about that design. It is not a
universal guarantee of unbiased inference, causal identification, or
substantive validity.

## Python API mapping

- `gp3bayespy.assess_binary_prior_sensitivity`
- `gp3bayespy.assess_duration_prior_sensitivity`
- `gp3bayespy.run_binary_recovery`
- `gp3bayespy.run_duration_recovery`

## Python usage

```python
import gp3bayespy as gp

# All functions listed above are available from the package root.
# Use help(gp.<function>) or the API reference for the exact Python signature.
```

An executable workflow for this family is included in `../../examples/sensitivity_workflow.py`.
