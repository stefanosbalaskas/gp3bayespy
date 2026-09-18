# Model comparison for multilevel mediation

`compare_multilevel_mediation_models()` provides PSIS-LOO comparison for fitted mediation models, but comparison is valid only when pointwise predictive units correspond to the **same mediator/outcome observations in the same participant-trial order**.

## Valid use

Examples of candidate models that can be compared when fitted to the same analysis rows include:

- alternative defensible random-slope structures;
- alternative mediator likelihoods with compatible observed support;
- alternative outcome likelihoods where the response encoding is unchanged;
- alternative prior specifications when the scientific estimand and data are held fixed.

```python
comparison = compare_multilevel_mediation_models(
    {
        "random_intercept": fit_ri,
        "random_slope": fit_rs,
    }
)
print(comparison)
```

## Invalid use

Do **not** compare fits when any candidate used:

- a different missingness policy that changed the analyzed rows;
- a different quality-eligibility rule;
- manually deleted participants or trials;
- a different mediator or outcome variable;
- reordered or otherwise non-aligned trial observations.

The comparison function checks ordered participant/trial keys and mediator/outcome values and fails before PSIS-LOO when they are not aligned.

## Why the guard matters

PSIS-LOO is pointwise. If model A's first log-likelihood contribution refers to participant 1 trial 1 while model B's first contribution refers to a different trial, the pointwise comparison no longer represents the same predictive units.

The implementation also aligns mediator and outcome log-likelihood observation dimensions before forming the joint pointwise log likelihood. This prevents xarray-style broadcasting from accidentally creating an outer product rather than one joint log-likelihood value per analyzed trial.

## Comparison is not automatic selection

A lower expected predictive loss or higher expected log predictive density does not automatically establish the preferred scientific mediation model. Review:

1. whether both models encode the intended estimand;
2. sampler diagnostics;
3. Pareto-*k* diagnostics and influential observations;
4. posterior predictive adequacy;
5. scientific plausibility and interpretability;
6. sensitivity of substantive conclusions.

Do not let a predictive comparison silently override a preregistered design or a required random-effects structure without justification.

## Recommended reporting

Report the candidate models, the common analysis dataset, the comparison metric, uncertainty in the comparison, Pareto-*k* diagnostics, and whether the substantive indirect-effect conclusion changed.

A concise statement is:

> Candidate mediation models were compared with PSIS-LOO using identical ordered trial-level observations. The comparison was treated as predictive evidence rather than automatic model selection, and Pareto-*k* diagnostics and posterior predictive checks were reviewed alongside the expected log predictive density difference.

## Related API

- `compare_multilevel_mediation_models()`
- `check_mediation_convergence()`
- `posterior_predictive_check_mediation()`
- `estimate_within_indirect_effect()`
- `report_multilevel_gaze_mediation()`
