"""Advanced dynamic-pupil simulation, governed fitting, prediction, and audit."""

import numpy as np

import gp3bayespy as gp

sim = gp.simulate_advanced_pupil_timecourse(
    n_participants=6,
    trials_per_participant=4,
    time_points=12,
    missing_fraction=0.0,
    outlier_fraction=0.0,
    seed=11,
)
dt_seconds = np.diff(np.sort(sim.data["time_ms"].unique())).mean() / 1000
contract = gp.create_pupil_contract(
    "pupil",
    "participant_id",
    "trial_id",
    "time_ms",
    "millimetres",
    1 / dt_seconds,
    time_unit="milliseconds",
    condition_col="condition",
    luminance_col="luminance",
)
prepared = gp.prepare_pupil_timecourse(sim.data, contract)
specification = gp.specify_advanced_pupil_timecourse_model(
    prepared,
    temporal_structure="linear",
    covariates=("luminance",),
    allow_high_complexity=True,
)
fit = gp.fit_advanced_pupil_model(specification, chains=1, iter=160, warmup=80, seed=11)
trajectory = gp.predict_advanced_pupil_trajectory(fit, ndraws=50)
audit = gp.audit_pupil_temporal_dependence(prepared, max_lag=4)
print(gp.advanced_pupil_trajectory_table(trajectory).head())
print(gp.pupil_autocorrelation_table(audit))
