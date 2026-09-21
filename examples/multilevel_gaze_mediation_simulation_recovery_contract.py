"""Backend-independent simulation/recovery planning contract for multilevel mediation."""

from eyeprocesspy.multilevel_mediation import prepare_multilevel_mediation_data

from gp3bayespy.multilevel_mediation import (
    simulate_multilevel_gaze_mediation,
    specify_multilevel_gaze_mediation,
)

scenarios = {
    "weak": {"a_within": 0.10, "b_within": 0.10, "cprime_within": 0.20},
    "moderate": {"a_within": 0.45, "b_within": 0.55, "cprime_within": 0.20},
    "strong": {"a_within": 0.90, "b_within": 1.00, "cprime_within": 0.20},
}

for offset, (label, truth) in enumerate(scenarios.items()):
    raw = simulate_multilevel_gaze_mediation(
        n_participants=12,
        trials_per_participant=12,
        a_within=truth["a_within"],
        b_within=truth["b_within"],
        cprime_within=truth["cprime_within"],
        a_between=0.20,
        b_between=0.30,
        seed=2026 + offset,
    )
    prepared = prepare_multilevel_mediation_data(
        raw,
        x_col="ai_correct",
        mediator_col="source_dwell",
        outcome_col="correct_override",
        quality_col="valid_fraction",
        minimum_quality=0.80,
        source_id=f"synthetic-recovery-{label}",
        warn=False,
    )
    spec = specify_multilevel_gaze_mediation(
        prepared,
        mediator_family="gaussian",
        outcome_family="bernoulli",
        random_slopes=("mediator_x",),
        missingness_policy="error",
    )
    intended_indirect = truth["a_within"] * truth["b_within"]
    print(
        label,
        "rows=", spec.analysis_rows,
        "truth_a_within=", truth["a_within"],
        "truth_b_within=", truth["b_within"],
        "truth_indirect=", round(intended_indirect, 3),
        "estimable_paths=", ",".join(spec.estimable_paths),
    )
