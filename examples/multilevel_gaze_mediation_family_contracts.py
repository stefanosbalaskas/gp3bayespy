"""Backend-independent family-contract examples for multilevel gaze mediation."""

import numpy as np

from eyeprocesspy.multilevel_mediation import prepare_multilevel_mediation_data
from gp3bayespy.multilevel_mediation import (
    simulate_multilevel_gaze_mediation,
    specify_multilevel_gaze_mediation,
)

raw = simulate_multilevel_gaze_mediation(
    n_participants=8,
    trials_per_participant=8,
    seed=2026,
)

variants = {
    "lognormal": raw["source_dwell"].clip(lower=1e-6),
    "gamma": raw["source_dwell"].clip(lower=1e-6),
    "beta": 1 / (1 + np.exp(-raw["source_dwell"])),
    "bernoulli": (raw["source_dwell"] > raw["source_dwell"].median()).astype(int),
    "poisson": np.rint(raw["source_dwell"] * 2).astype(int),
    "negative_binomial": np.rint(raw["source_dwell"] * 2).astype(int),
}

for family, mediator in variants.items():
    data = raw.copy()
    data["family_mediator"] = mediator

    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="ai_correct",
        mediator_col="family_mediator",
        outcome_col="correct_override",
        quality_col="valid_fraction",
        minimum_quality=0.80,
        source_id=f"synthetic-{family}-contract",
        warn=False,
    )

    spec = specify_multilevel_gaze_mediation(
        prepared,
        mediator_family=family,
        outcome_family="bernoulli",
        random_slopes=("mediator_x",),
        missingness_policy="error",
    )

    print(
        family,
        "rows=", spec.analysis_rows,
        "mediator_family=", spec.mediator_family,
        "outcome_family=", spec.outcome_family,
        "estimable_paths=", ",".join(spec.estimable_paths),
    )
