"""Opt-in real PyMC smoke for predictive contrasts and diagnostics."""

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
        seed=4901,
    )
    duration = gp.fit_duration_model(
        _duration_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=4902,
    )

    binary_expected = gp.predict_model(
        binary,
        type="expected",
        include_group_effects=True,
        ndraws=25,
    )
    duration_expected = gp.predict_model(
        duration,
        type="expected",
        include_group_effects=True,
        ndraws=25,
    )

    difference = gp.prediction_contrast(binary_expected, 1, 2)
    odds_ratio = gp.prediction_contrast(
        binary_expected,
        1,
        2,
        measure="odds_ratio",
    )
    ratio = gp.prediction_contrast(duration_expected, 1, 2, measure="ratio")
    exceedance = gp.prediction_exceedance_probability(binary_expected, 0.5)
    uncertainty = gp.prediction_uncertainty_decomposition(
        duration,
        include_group_effects=True,
        ndraws=25,
        seed=4903,
    )
    grouped = gp.grouped_prediction_check(
        binary,
        "participant_id",
        ndraws=25,
        seed=4904,
    )
    binary_residuals = gp.predictive_residuals(binary, type="pearson", ndraws=25)
    duration_residuals = gp.predictive_residuals(duration, ndraws=25)

    frames = [
        difference,
        odds_ratio,
        ratio,
        exceedance,
        uncertainty.table,
        grouped.table,
        binary_residuals,
        duration_residuals,
    ]
    if any(frame.empty for frame in frames):
        raise RuntimeError("Predictive diagnostic smoke returned an empty table.")
    if difference["automatic_decision"].any() or exceedance["automatic_decision"].any():
        raise RuntimeError("Predictive contrasts must not make automatic decisions.")
    if grouped.automatic_exclusion:
        raise RuntimeError("Grouped checks must not exclude groups automatically.")
    if uncertainty.causal_variance_decomposition:
        raise RuntimeError("Prediction uncertainty must remain non-causal.")
    if not np.isfinite(binary_residuals["residual"]).all():
        raise RuntimeError("Binary predictive residuals must be finite.")
    if not np.isfinite(duration_residuals["residual"]).all():
        raise RuntimeError("Duration predictive residuals must be finite.")
    if not uncertainty.table["residual_component"].ge(0).all():
        raise RuntimeError("Residual uncertainty components must be non-negative.")

    print("GPB-PY-09 real PyMC predictive-diagnostics smoke: PASS")
    print("Difference mean:", float(difference.loc[0, "mean"]))
    print("Odds-ratio mean:", float(odds_ratio.loc[0, "mean"]))
    print("Duration-ratio mean:", float(ratio.loc[0, "mean"]))
    print("Grouped rows:", len(grouped.table))
    print("Automatic decisions: False")
    print("Automatic group exclusion: False")
    print("Causal variance decomposition: False")
    print("Global adequacy established: False")


if __name__ == "__main__":
    main()
