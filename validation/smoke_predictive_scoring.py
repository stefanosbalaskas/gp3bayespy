"""Opt-in real PyMC smoke for predictive scoring and calibration diagnostics."""

from __future__ import annotations

import numpy as np
from smoke_fitting_backend import _binary_specification, _duration_specification

import gp3bayespy as gp


def main() -> None:
    binary = gp.fit_binary_model(
        _binary_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4801,
    )
    duration = gp.fit_duration_model(
        _duration_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4802,
    )

    binary_expected = gp.predict_model(
        binary,
        type="expected",
        include_group_effects=True,
        ndraws=25,
    )
    binary_predictive = gp.predict_model(
        binary,
        type="predictive",
        include_group_effects=True,
        ndraws=25,
        seed=4803,
    )
    duration_expected = gp.predict_model(
        duration,
        type="expected",
        include_group_effects=True,
        ndraws=25,
    )
    duration_predictive = gp.predict_model(
        duration,
        type="predictive",
        include_group_effects=True,
        ndraws=25,
        seed=4804,
    )

    binary_scores = gp.binary_prediction_scores(binary_expected)
    threshold_curve = gp.binary_threshold_metrics(
        binary_expected,
        thresholds=(0.25, 0.5, 0.75),
    )
    binary_calibration = gp.binary_calibration_table(binary_expected, bins=4)
    duration_scores = gp.duration_prediction_scores(duration_expected)
    duration_calibration = gp.duration_quantile_calibration(
        duration_predictive,
        quantiles=(0.25, 0.5, 0.75),
    )
    duration_pit = gp.duration_pit_table(duration_predictive)
    coverage = gp.predictive_coverage_table(duration_predictive, levels=(0.5, 0.9))
    summary = gp.posterior_predictive_summary_table(duration_predictive)

    tables = [
        binary_scores,
        threshold_curve,
        binary_calibration,
        duration_scores,
        duration_calibration,
        duration_pit,
        coverage,
        summary,
    ]
    if any(table.empty for table in tables):
        raise RuntimeError("Predictive scoring smoke returned an empty diagnostic table.")
    if not np.isfinite(float(binary_scores.loc[0, "brier"])):
        raise RuntimeError("Binary Brier score must be finite.")
    if not np.isfinite(float(duration_scores.loc[0, "log_rmse"])):
        raise RuntimeError("Duration log-RMSE must be finite.")
    if threshold_curve["automatic_decision"].any():
        raise RuntimeError("Threshold diagnostics must not make automatic decisions.")
    if not duration_pit["pit"].between(0, 1).all():
        raise RuntimeError("Duration PIT values must lie in [0, 1].")
    if not coverage["empirical_coverage"].between(0, 1).all():
        raise RuntimeError("Empirical coverage must lie in [0, 1].")
    if binary_predictive.out_of_sample_adequacy_established:
        raise RuntimeError("Predictive diagnostics must not establish adequacy.")

    print("GPB-PY-08 real PyMC predictive-scoring smoke: PASS")
    print("Binary Brier:", float(binary_scores.loc[0, "brier"]))
    print("Binary AUC:", float(binary_scores.loc[0, "auc"]))
    print("Duration log-RMSE:", float(duration_scores.loc[0, "log_rmse"]))
    print("Calibration bins:", len(binary_calibration))
    print("Automatic decisions: False")
    print("Global adequacy established: False")


if __name__ == "__main__":
    main()
