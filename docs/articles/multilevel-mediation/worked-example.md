# Worked example: repeated AI advice

This synthetic example follows the target design: 100 participants × 20 trials, AI correctness as the trial-level exposure, source inspection as the gaze mediator, and a binary correct-override outcome.

```python
from eyeprocesspy import prepare_multilevel_mediation_data
from gp3bayespy import (
    simulate_multilevel_gaze_mediation,
    fit_multilevel_gaze_mediation,
    check_mediation_convergence,
    estimate_within_indirect_effect,
    posterior_predictive_check_mediation,
)

trials = simulate_multilevel_gaze_mediation(
    n_participants=100,
    trials_per_participant=20,
    seed=2026,
)

prepared = prepare_multilevel_mediation_data(
    trials,
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    quality_col="valid_fraction",
    minimum_quality=0.80,
    source_id="synthetic-ai-advice-v1",
    warn=False,
)

fit = fit_multilevel_gaze_mediation(
    prepared,
    mediator_family="gaussian",
    outcome_family="bernoulli",
    missingness_policy="error",
    random_slopes=("mediator_x",),
    chains=4,
    draws=1000,
    tune=1000,
    seed=2026,
)

print(check_mediation_convergence(fit))
print(estimate_within_indirect_effect(fit))
print(posterior_predictive_check_mediation(fit, draws=200, seed=2026))
```

## Interpretation

The within-person indirect effect asks whether trials with an AI-correctness state above a participant's own average shift source inspection, which in turn is associated with the trial outcome after separating the participant's mean mediator level. The between-person indirect effect describes stable participant differences and should not be substituted for the manipulated within-person mechanism.

## Failure-case variant

Set one `source_dwell` value to missing. The default model specification will refuse to fit until the analyst explicitly selects a missingness policy. Set a participant's exposure to a constant across all trials and the preparation audit will identify the loss of within-person support.

## CI-sized contract example

For documentation and continuous integration, the repository also ships a small backend-independent example that prepares 10 × 8 synthetic trials and creates the complete model specification without sampling. This validates package boundaries, family declarations, priors, random-slope declarations, missingness policy, and provenance even when an optional Bayesian backend is unavailable.

The optional fitting example exits cleanly with `BackendUnavailableError` when PyMC is unavailable or incompatible and explicitly reports that no alternative estimator was selected.


## Prior-sensitivity extension

Repeat the same declared model with at least one defensible narrower and wider coefficient prior rather than treating defaults as fixed truth. Compare the within indirect-effect distribution, posterior predictive checks, and convergence diagnostics. A material shift under modest prior changes should be reported as prior sensitivity, not hidden by selecting the preferred prior.
