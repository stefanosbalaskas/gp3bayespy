# Worked example: prior sensitivity

Prior sensitivity should be planned as a comparison of **defensible prior specifications for the same scientific model**, not as a search for the prior that produces the preferred indirect effect.

## 1. Hold the design and likelihood fixed

Use one prepared dataset, one mediator family, one outcome family, one random-effects structure, and one missingness policy. Change only the prior dimension you are examining.

```python
from gp3bayespy import (
    create_mediation_prior_specification,
    fit_multilevel_gaze_mediation,
)

prior_scenarios = {
    "narrow": create_mediation_prior_specification(coefficient_sd=0.5),
    "default": create_mediation_prior_specification(coefficient_sd=1.0),
    "wide": create_mediation_prior_specification(coefficient_sd=1.5),
}

fits = {
    name: fit_multilevel_gaze_mediation(
        prepared,
        mediator_family="gaussian",
        outcome_family="bernoulli",
        priors=priors,
        random_slopes=("mediator_x",),
        missingness_policy="quality_eligible",
        chains=4,
        draws=1000,
        tune=1000,
        seed=2026,
    )
    for name, priors in prior_scenarios.items()
}
```

The repository also ships a backend-independent contract example:

```bash
python examples/multilevel_gaze_mediation_prior_sensitivity_contract.py
```

That script verifies that the same prepared rows and model structure are retained while only the coefficient-prior scale changes.

## 2. Check each fit before comparing effects

Do not compare indirect effects from a failed fit.

```python
from gp3bayespy import check_mediation_convergence

for name, fit in fits.items():
    print(name, check_mediation_convergence(fit))
```

Also inspect prior predictive behavior and posterior predictive behavior for every scenario. A prior can be numerically convenient while still implying implausible mediator or outcome values.

## 3. Compare the declared estimand

```python
from gp3bayespy import estimate_within_indirect_effect

for name, fit in fits.items():
    print(name, estimate_within_indirect_effect(fit))
```

Compare posterior location, interval width, sign stability, and substantive interpretation. Avoid reducing the exercise to whether an interval happens to cross zero.

## 4. Interpret sensitivity rather than select a winner

A useful summary might say:

> The within-participant coefficient-product indirect effect retained the same sign and similar magnitude across coefficient priors with standard deviations of 0.5, 1.0, and 1.5, while posterior uncertainty widened modestly under the broader prior. Posterior predictive behavior and sampler diagnostics were materially similar across the three specifications.

If the result changes materially, report that sensitivity directly:

> The indirect-effect posterior shifted substantially under modest changes to the coefficient prior. We therefore interpret the mediated pathway as prior-sensitive rather than selecting the specification yielding the largest effect.

## What not to vary simultaneously

Do not call the exercise “prior sensitivity” if you also change the missingness policy, likelihood family, random-slope structure, trial exclusions, or mediator operationalization in the same comparison. Those are separate sensitivity dimensions and should be labeled separately.

## Relationship to model comparison

Prior sensitivity and predictive model comparison answer different questions. A PSIS-LOO comparison may be useful for predictive adequacy, but it does not determine which prior is scientifically “correct,” and it cannot be used when models were fitted to different observations.

See [Model comparison for mediation](model-comparison.md) for the required same-observation rule.
