from __future__ import annotations

from types import SimpleNamespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_pupil_low_level_waveform_acf_and_plot_tail():
    assert np.all(p._waveform(np.array([-2.0, -1.0, 0.0])) == 0)
    rng = np.random.default_rng(1)
    assert p._ar1_noise(rng, 1, 0.5, 1.0).shape == (1,)
    assert p._ar1_noise(rng, 5, 0.5, 1.0).shape == (5,)

    readiness = p.PupilReadiness(
        audit_version="0.1",
        family="pupil",
        status="review",
        summary=pd.DataFrame(
            {
                "metric": ["rows", "trials"],
                "status": ["pass", "review"],
            }
        ),
        by_participant=pd.DataFrame(),
        by_condition=pd.DataFrame(),
        by_trial=pd.DataFrame(),
    )
    assert p.plot_pupil_readiness(readiness).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_readiness(object())

    trajectory = p.PupilTrajectory(
        table=pd.DataFrame(
            {
                "event_time": [0.0, 0.1, 0.2],
                "estimate": [3.0, 3.1, 3.2],
                "lower": [2.9, 3.0, 3.1],
                "upper": [3.1, 3.2, 3.3],
            }
        ),
        unit="millimetres",
        probability=0.9,
        interval="pointwise",
        prediction_type="expected",
        finite_grid_qualification=True,
    )
    assert p.plot_pupil_posterior_trajectory(trajectory).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_posterior_trajectory(object())

    for column in ("mean", "estimate", "value"):
        table = pd.DataFrame({column: [1.0, 2.0]})
        estimand = p.PupilEstimand(
            table=table,
            estimand="window_mean",
            unit="millimetres",
            probability=0.9,
            window=(0.0, 0.2),
        )
        assert p.plot_pupil_estimand(estimand).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_estimand(object())

    plan = SimpleNamespace()
    validation_two = p.PupilValidation(
        target="future_segment",
        strategy="blocked",
        executed=True,
        plan=plan,
        result=None,
        table=pd.DataFrame(
            {
                "fold": [1, 2],
                "score": [0.1, 0.2],
            }
        ),
    )
    assert p.plot_pupil_validation(validation_two).axes

    validation_one = p.PupilValidation(
        target="future_segment",
        strategy="blocked",
        executed=True,
        plan=plan,
        result=None,
        table=pd.DataFrame({"score": [0.1, 0.2]}),
    )
    assert p.plot_pupil_validation(validation_one).axes

    validation_none = p.PupilValidation(
        target="future_segment",
        strategy="blocked",
        executed=False,
        plan=plan,
        result=None,
        table=pd.DataFrame({"label": ["a", "b"]}),
    )
    assert p.plot_pupil_validation(validation_none).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_validation(object())

    comparison = p.PupilSensitivityComparison(
        pd.DataFrame(
            {
                "scenario": ["a", "b"],
                "estimate": [0.1, 0.2],
            }
        )
    )
    assert p.plot_pupil_sensitivity(comparison).axes

    measurement = p.PupilMeasurementAudit(
        pd.DataFrame(
            {
                "metric": ["left_right_difference"],
                "value": [0.1],
            }
        )
    )
    assert p.plot_pupil_measurement_audit(measurement).axes

    measurement_text = p.PupilMeasurementAudit(
        pd.DataFrame({"metric": ["review"], "status": ["pass"]})
    )
    assert p.plot_pupil_measurement_audit(measurement_text).axes
    with pytest.raises(GP3BayesError):
        p.plot_pupil_measurement_audit(object())


def test_pupil_acf_empty_and_mean_lag1_edges():
    empty = p._acf_table(
        np.array([], dtype=float),
        pd.Series([], dtype=str),
        3,
    )
    assert empty["acf"].isna().all()

    values = np.array([1.0, 1.0, 1.0, 2.0])
    series = pd.Series(["a", "a", "a", "b"])
    assert np.isnan(p._mean_lag1(values, series))
