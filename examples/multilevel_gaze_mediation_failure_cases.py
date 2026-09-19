"""Failure-case demonstrations for trial-level mediation contracts."""

import numpy as np

from eyeprocesspy.multilevel_mediation import prepare_multilevel_mediation_data
from gp3bayespy.multilevel_mediation import (
    GP3BayesError,
    simulate_multilevel_gaze_mediation,
    specify_multilevel_gaze_mediation,
)

raw = simulate_multilevel_gaze_mediation(n_participants=6, trials_per_participant=6, seed=9)
raw.loc[3, "source_dwell"] = np.nan
prepared = prepare_multilevel_mediation_data(
    raw,
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    warn=False,
)

try:
    specify_multilevel_gaze_mediation(prepared)
except GP3BayesError as exc:
    print("Missingness gate:", exc)

balanced = simulate_multilevel_gaze_mediation(n_participants=6, trials_per_participant=6, seed=10)
balanced["ai_correct"] = np.tile([0, 1, 0, 1, 0, 1], 6)
prepared_balanced = prepare_multilevel_mediation_data(
    balanced,
    x_col="ai_correct",
    mediator_col="source_dwell",
    outcome_col="correct_override",
    warn=False,
)
spec = specify_multilevel_gaze_mediation(prepared_balanced, random_slopes=())
print("Balanced design estimable paths:", ", ".join(spec.estimable_paths))
assert "a_between" not in spec.estimable_paths
assert "cprime_between" not in spec.estimable_paths
