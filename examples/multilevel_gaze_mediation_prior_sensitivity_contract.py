"""CI-small prior-sensitivity contract example for multilevel gaze mediation."""

from eyeprocesspy.multilevel_mediation import prepare_multilevel_mediation_data
from gp3bayespy.multilevel_mediation import (
    create_mediation_prior_specification,
    simulate_multilevel_gaze_mediation,
    specify_multilevel_gaze_mediation,
)

trials = simulate_multilevel_gaze_mediation(
    n_participants=8,
    trials_per_participant=8,
    seed=2026,
)

prepared = prepare_multilevel_mediation_data(
    trials,
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    quality_col="valid_fraction",
    minimum_quality=0.80,
    source_id="synthetic-prior-sensitivity-contract",
    warn=False,
)

scenarios = {
    "narrow": create_mediation_prior_specification(coefficient_sd=0.5),
    "default": create_mediation_prior_specification(coefficient_sd=1.0),
    "wide": create_mediation_prior_specification(coefficient_sd=1.5),
}

for label, priors in scenarios.items():
    spec = specify_multilevel_gaze_mediation(
        prepared,
        mediator_family="gaussian",
        outcome_family="bernoulli",
        priors=priors,
        random_slopes=("mediator_x",),
        missingness_policy="error",
    )
    print(
        label,
        "coefficient_sd=", spec.priors.coefficient_sd,
        "analysis_rows=", spec.analysis_rows,
        "between_x_estimable=", "a_between" in spec.estimable_paths,
    )
