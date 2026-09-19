"""Backend-independent trial-level mediation contract example.

Requires eyeprocesspy only for preparation. It deliberately stops before
sampling so it remains fast and deterministic in documentation/CI.
"""

from eyeprocesspy.multilevel_mediation import prepare_multilevel_mediation_data
from gp3bayespy.multilevel_mediation import (
    create_mediation_prior_specification,
    simulate_multilevel_gaze_mediation,
    specify_multilevel_gaze_mediation,
)

raw = simulate_multilevel_gaze_mediation(
    n_participants=10,
    trials_per_participant=8,
    seed=2026,
)
prepared = prepare_multilevel_mediation_data(
    raw,
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    quality_col="valid_fraction",
    minimum_quality=0.8,
    source_id="synthetic-ai-advice",
    warn=False,
)
priors = create_mediation_prior_specification(
    coefficient_sd=0.75,
    group_sd_scale=1.0,
)
spec = specify_multilevel_gaze_mediation(
    prepared,
    mediator_family="gaussian",
    outcome_family="bernoulli",
    priors=priors,
    random_slopes=("mediator_x",),
    missingness_policy="error",
)

print("analysis rows:", spec.analysis_rows)
print("estimable paths:", ", ".join(spec.estimable_paths))
print("missingness policy:", spec.missingness_policy)
print("estimand scale:", spec.provenance["model_specification"]["estimand_scale"])
