"""Opt-in real PyMC smoke for the advanced predictive-diagnostics family."""

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
        seed=5101,
    )
    duration = gp.fit_duration_model(
        _duration_specification(),
        chains=1,
        iter=100,
        warmup=50,
        cores=1,
        seed=5102,
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
        seed=5103,
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
        seed=5104,
    )

    long = gp.prediction_draws_long(binary_expected, max_draws=10, seed=5105)
    statistic = gp.posterior_predictive_statistic(binary_predictive, statistic="mean")
    statistic_table = gp.ppc_statistic_table(statistic)
    confusion = gp.binary_confusion_table(binary_expected)
    roc = gp.binary_roc_curve(binary_expected)
    precision_recall = gp.binary_precision_recall_curve(binary_expected)
    calibration = gp.binary_calibration_error(binary_expected, bins=4)
    group_calibration = gp.binary_group_calibration(binary_expected, "participant_id")
    qq = gp.duration_qq_table(duration_predictive, probs=[0.1, 0.5, 0.9])
    threshold = float(np.median(duration_predictive.observed))
    tail = gp.duration_tail_check(duration_predictive, threshold)
    grouped = gp.group_prediction_summary(duration_expected, "condition")
    contrasts = gp.prediction_pairwise_contrasts(
        duration_expected,
        rows=[1, 2, 3],
        measure="ratio",
        max_rows=3,
    )
    widths = gp.prediction_interval_width(duration_expected)
    ranks = gp.prediction_rank_probabilities(
        duration_expected,
        rows=[1, 2, 3],
        direction="higher",
        max_rows=3,
    )

    frames = [
        long,
        statistic_table,
        confusion,
        roc,
        precision_recall,
        calibration,
        group_calibration,
        qq,
        tail,
        grouped,
        contrasts,
        widths,
        ranks,
    ]
    if any(frame.empty for frame in frames):
        raise RuntimeError("Advanced predictive diagnostic smoke returned an empty table.")
    if statistic.automatic_adequacy_verdict:
        raise RuntimeError("PPC statistic must not make an automatic adequacy verdict.")
    if bool(calibration.loc[0, "automatic_adequacy_verdict"]):
        raise RuntimeError("Calibration error must not make an adequacy verdict.")
    if bool(tail.loc[0, "automatic_adequacy_verdict"]):
        raise RuntimeError("Duration tail check must not make an adequacy verdict.")
    if contrasts["automatic_decision"].any():
        raise RuntimeError("Pairwise contrasts must not make automatic decisions.")
    if ranks["automatic_selection"].any():
        raise RuntimeError("Ranking probabilities must not select automatically.")
    if confusion["count"].sum() != len(binary_expected.observed):
        raise RuntimeError("Confusion counts must cover all observed binary rows.")
    if not widths["interval_width"].ge(0).all():
        raise RuntimeError("Prediction interval widths must be non-negative.")

    print("GPB-PY-10 real PyMC advanced-predictive-diagnostics smoke: PASS")
    print("ROC rows:", len(roc))
    print("Precision-recall rows:", len(precision_recall))
    print("Calibration ECE:", float(calibration.loc[0, "expected_calibration_error"]))
    print("PPC two-sided tail probability:", statistic.two_sided_tail_probability)
    print("Duration Q-Q rows:", len(qq))
    print("Pairwise contrasts:", len(contrasts))
    print("Automatic adequacy verdicts: False")
    print("Automatic selections: False")
    print("Global adequacy established: False")


if __name__ == "__main__":
    main()
