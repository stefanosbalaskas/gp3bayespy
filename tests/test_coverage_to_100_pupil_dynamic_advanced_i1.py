from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _advanced_data(n_time: int = 12) -> pd.DataFrame:
    rows = []
    for participant in ("p1", "p2", "p3"):
        for trial, condition in ((1, "A"), (2, "B"), (3, "A")):
            for t in range(n_time):
                rows.append(
                    {
                        "participant_id": participant,
                        "trial_id": f"{participant}-{trial}",
                        "item_id": f"i{trial}",
                        "condition": condition,
                        "time_ms": float(t * 100),
                        "pupil": 3.0 + 0.2 * (condition == "B") + 0.05 * np.sin(t),
                        "x": float(t),
                        "x_se": 0.1,
                        "response_se": 0.05,
                        "text": "a",
                    }
                )
    return pd.DataFrame(rows)


def _prediction():
    times = np.array([0.0, 0.1, 0.2, 0.3])
    grid = pd.DataFrame(
        {
            "time": np.r_[times, times],
            "condition": ["A"] * 4 + ["B"] * 4,
        }
    )
    draws = np.tile(
        np.r_[
            np.array([0.0, 0.1, 0.3, 0.4]),
            np.array([0.0, 0.05, 0.1, 0.2]),
        ],
        (20, 1),
    )
    draws += np.linspace(-0.01, 0.01, 20)[:, None]
    return SimpleNamespace(
        grid=grid,
        draws=draws,
        specification={"mapping": {"time": "time", "condition": "condition"}},
    )


def test_dynamic_functional_and_gazepoint_guard_matrix():
    pred = _prediction()

    with pytest.raises(GP3BayesError):
        p._functional_parts(object())

    bad_grid = pd.DataFrame({"time": [0.0, 0.0], "condition": ["A", "A"]})
    with pytest.raises(GP3BayesError):
        p._derivative_once(
            bad_grid,
            np.zeros((3, 2)),
            "time",
            "condition",
        )
    with pytest.raises(GP3BayesError):
        p._derivative_once(
            pd.DataFrame({"time": [0.0], "condition": ["A"]}),
            np.zeros((3, 1)),
            "time",
            "condition",
        )

    d1 = p.estimate_pupil_trajectory_derivative(pred, order=1, probability=0.9)
    d2 = p.estimate_pupil_trajectory_derivative(pred, order=2, probability=0.9)
    assert d1.order == 1
    assert d2.order == 2
    assert len(p.pupil_trajectory_derivative_table(d1)) > 0
    assert len(p.pupil_trajectory_derivative_table(d1, 0.8)) > 0
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_trajectory_derivative(pred, order=3)
    with pytest.raises(GP3BayesError):
        p.pupil_trajectory_derivative_table(object())  # type: ignore[arg-type]

    contrast = p.estimate_pupil_dynamic_contrast(
        pred,
        ("A", "B"),
        threshold=0.02,
    )
    assert len(p.pupil_dynamic_contrast_table(contrast)) == 4

    for direction in ("above", "below", "absolute"):
        duration = p.estimate_pupil_threshold_duration(
            contrast,
            direction=direction,
        )
        assert len(duration.summary) == 1

    with pytest.raises(GP3BayesError):
        p.estimate_pupil_dynamic_contrast(pred, ("A",))
    no_condition = SimpleNamespace(
        grid=pred.grid.drop(columns="condition"),
        draws=pred.draws,
        specification={"mapping": {"time": "time", "condition": None}},
    )
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_dynamic_contrast(no_condition, ("A", "B"))

    disjoint = SimpleNamespace(
        grid=pd.DataFrame(
            {
                "time": [0.0, 0.1, 1.0, 1.1],
                "condition": ["A", "A", "B", "B"],
            }
        ),
        draws=np.zeros((5, 4)),
        specification={"mapping": {"time": "time", "condition": "condition"}},
    )
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_dynamic_contrast(disjoint, ("A", "B"))

    with pytest.raises(GP3BayesError):
        p.pupil_dynamic_contrast_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_threshold_duration(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_threshold_duration(
            contrast,
            direction="bad",  # type: ignore[arg-type]
        )

    short = p.PupilDynamicContrast(
        table=pd.DataFrame(),
        grid=pd.DataFrame({"time": [0.0]}),
        draws=np.zeros((5, 1)),
        contrast=("A", "B"),
        threshold=0.0,
        derivative_order=0,
        probability=0.95,
        specification={"mapping": {"time": "time", "condition": "condition"}},
    )
    with pytest.raises(GP3BayesError):
        p.estimate_pupil_threshold_duration(short)

    missing = p.inspect_gazepoint_pupil_schema(pd.DataFrame({"TIME": [0.0]}))
    assert missing.status == "missing_pupil_channel"
    single = p.inspect_gazepoint_pupil_schema(pd.DataFrame({"TIME": [0.0], "LPD": [4.0]}))
    assert single.status == "single_pupil_candidate"
    ambiguous = p.inspect_gazepoint_pupil_schema(
        pd.DataFrame({"TIME": [0.0], "LPD": [4.0], "RPD": [4.1]})
    )
    assert ambiguous.status == "ambiguous_pupil_channel"
    assert len(p.gazepoint_pupil_mapping_table(single)) >= 1
    assert len(p.gazepoint_pupil_mapping_table(pd.DataFrame({"LPD": [4.0]}))) >= 1
    with pytest.raises(GP3BayesError):
        p.inspect_gazepoint_pupil_schema(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.gazepoint_pupil_mapping_table(object())  # type: ignore[arg-type]


def test_advanced_builder_guard_matrix():
    data = _advanced_data()

    with pytest.raises(GP3BayesError):
        p.create_pupil_measurement_model(
            covariate_errors={"": "x_se"},
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_measurement_model(
            baseline_error="x_se",
            covariate_errors={"baseline": "other_se"},
        )
    mm = p.create_pupil_measurement_model(
        covariate_errors={"x": "x_se"},
        response_error="response_se",
    )

    with pytest.raises(GP3BayesError):
        p.create_pupil_missingness_spec(response="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_missingness_spec(assumptions="MNAR")
    missing = p.create_pupil_missingness_spec(
        response="model",
        predictors=("x",),
        auxiliary_predictors=("text",),
    )

    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(kernel="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(basis="bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(scale=1)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_gp_spec(k=4)
    exact = p.create_pupil_gp_spec(basis="exact")
    assert exact.k is None

    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(True, 0)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(4, 0)
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(0, 0)
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(2, 1, covariance=True)
    with pytest.raises(GP3BayesError):
        p.create_pupil_arma_spec(1, 0, covariance=1)  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        p.specify_pupil_distribution("bad")  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.specify_pupil_distribution(residual_scale="bad")  # type: ignore[arg-type]
    dist = p.specify_pupil_distribution("student", "condition")
    assert p.pupil_distribution_table(dist).loc[0, "family"] == "student"
    with pytest.raises(GP3BayesError):
        p.pupil_distribution_table(object())  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        p._advanced_mapping(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p._advanced_mapping(pd.DataFrame({"participant_id": ["p1"]}))
    mapped = p.pupil_advanced_mapping_table(data)
    assert {"response", "time", "participant"}.issubset(set(mapped["role"]))

    assert p._ac_spec("none") is None
    assert p._ac_spec("ar1").p == 1
    assert p._ac_spec("ar2").p == 2
    assert p._ac_spec("arma11").q == 1
    with pytest.raises(GP3BayesError):
        p._ac_spec("bad")

    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            distribution=object(),  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            temporal_structure="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            family="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            residual_scale="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            family="student",
            autocorrelation="ar1",
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            autocorrelation="ar1",
            missingness_model=missing,
            covariates=("x",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            allow_high_complexity=1,  # type: ignore[arg-type]
        )

    no_condition = data.drop(columns="condition")
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            no_condition,
            residual_scale="condition",
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            covariates=("missing",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            covariates=("time_ms",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            covariates=("text",),
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            measurement_model=object(),  # type: ignore[arg-type]
        )

    mm_unknown = p.create_pupil_measurement_model(covariate_errors={"x": "missing_se"})
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            covariates=("x",),
            measurement_model=mm_unknown,
        )

    bad_se = data.copy()
    bad_se.loc[bad_se.index[0], "x_se"] = 0
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            bad_se,
            covariates=("x",),
            measurement_model=mm,
        )

    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            missingness_model=object(),  # type: ignore[arg-type]
        )
    unknown_missing = p.create_pupil_missingness_spec(
        predictors=("unknown",),
    )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            missingness_model=unknown_missing,
        )
    not_cov = p.create_pupil_missingness_spec(
        predictors=("x",),
    )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            missingness_model=not_cov,
        )
    with pytest.raises(GP3BayesError):
        p.specify_advanced_pupil_timecourse_model(
            data,
            prior_scales={"b": -1},
        )

    valid = p.specify_advanced_pupil_timecourse_model(
        data,
        temporal_structure="smooth",
        family="gaussian",
        residual_scale="condition_time",
        covariates=("x",),
        measurement_model=mm,
        missingness_model=missing,
        allow_high_complexity=True,
    )
    assert valid.smooth_basis_dimension_effective <= valid.smooth_basis_dimension_requested
    assert not p.pupil_advanced_compatibility_table().empty
    assert not p.pupil_advanced_capabilities().empty
