from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _objects(seed: int = 2601):
    sim = p.simulate_pupil_timecourse(
        n_participants=3,
        trials_per_participant=3,
        n_items=3,
        sampling_frequency=10,
        time_window=(-0.2, 0.6),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=0,
        seed=seed,
    )
    contract = p.create_pupil_contract(
        outcome_col="pupil_mm",
        participant_col="participant_id",
        trial_col="trial_id",
        time_col="event_time",
        pupil_unit="millimetres",
        sampling_frequency=10,
        item_col="item_id",
        condition_col="condition",
        blink_col="blink",
        gaze_x_col="gaze_x",
        gaze_y_col="gaze_y",
        luminance_col="luminance",
        baseline_window=(-0.2, 0.0),
    )
    prepared = p.prepare_pupil_timecourse(
        sim.data,
        contract,
        baseline_operation="none",
    )
    return sim.data.copy(), contract, prepared


def test_contract_helper_and_prepare_guard_matrix():
    data, contract, prepared = _objects()

    with pytest.raises(GP3BayesError):
        p._scalar_name(None, "x")
    assert p._scalar_name(None, "x", True) is None
    with pytest.raises(GP3BayesError):
        p._scalar_name("", "x")
    with pytest.raises(GP3BayesError):
        p._positive(True, "x")
    with pytest.raises(GP3BayesError):
        p._positive(1.5, "x", True)
    with pytest.raises(GP3BayesError):
        p._probability(True, "p")
    with pytest.raises(GP3BayesError):
        p._probability(0, "p", True)
    with pytest.raises(GP3BayesError):
        p._window(("bad", 1), "w")
    with pytest.raises(GP3BayesError):
        p._window((1, 1), "w")
    assert p._window(None, "w") is None

    with pytest.raises(GP3BayesError):
        p.create_pupil_contract("y", "id", "trial", "time", "bad-unit", 60)
    with pytest.raises(GP3BayesError):
        p.create_pupil_contract(
            "y",
            "id",
            "trial",
            "time",
            "millimetres",
            60,
            time_unit="minutes",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_contract(
            "y",
            "id",
            "trial",
            "time",
            "millimetres",
            60,
            eye="both",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_contract(
            "y",
            "id",
            "trial",
            "time",
            "millimetres",
            60,
            baseline_method="mystery",
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_contract(
            "y",
            "id",
            "trial",
            "time",
            "millimetres",
            60,
            baseline_applied=1,  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_contract(
            "y",
            "id",
            "trial",
            "time",
            "millimetres",
            60,
            item_col="id",
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_contract(
            "y",
            "id",
            "trial",
            "time",
            "millimetres",
            60,
            screen_width=-1,
        )
    with pytest.raises(GP3BayesError):
        p.create_pupil_contract(
            "y",
            "id",
            "trial",
            "time",
            "millimetres",
            60,
            left_pupil_col="left",
            channel_audit_unit="mystery",
        )

    assert p._indicator(pd.Series([True, False]), "x").dtype.name == "boolean"
    assert p._indicator(pd.Series([0, 1]), "x").tolist() == [False, True]
    with pytest.raises(GP3BayesError):
        p._indicator(pd.Series([0, 2]), "x")

    assert np.allclose(
        p._convert_unit(pd.Series([0.001]), "metres", "millimetres"),
        [1.0],
    )
    assert np.allclose(
        p._convert_unit(pd.Series([1.0]), "millimetres", "metres"),
        [0.001],
    )
    assert np.allclose(
        p._convert_unit(pd.Series([1.0]), "pixels", "pixels"),
        [1.0],
    )
    with pytest.raises(GP3BayesError):
        p._convert_unit(pd.Series([1.0]), "pixels", "millimetres")

    one_timing = pd.DataFrame({".series_id": ["a", "a"], ".event_time": [0.0, 0.1]})
    timing = p._timing_summary(one_timing)
    assert timing["cv_dt"] == 0.0
    empty_timing = p._timing_summary(pd.DataFrame({".series_id": ["a"], ".event_time": [0.0]}))
    assert np.isnan(empty_timing["median_dt"])

    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(data, object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(pd.DataFrame(), contract)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(data, contract, max_rows=1)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data,
            contract,
            baseline_operation="mystery",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data.drop(columns="pupil_mm"),
            contract,
        )

    bad = data.copy()
    bad["pupil_mm"] = "x"
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad["event_time"] = "x"
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad.loc[bad.index[0], "participant_id"] = np.nan
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad.loc[bad.index[0], "pupil_mm"] = np.inf
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad.loc[bad.index[0], "pupil_mm"] = 0
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad.loc[bad.index[0], "condition"] = np.nan
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad["gaze_x"] = "x"
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    duplicate = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(duplicate, contract)

    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(data, contract, output_unit="bad")
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(data, contract, output_unit="pixels")

    already = replace(
        contract,
        preprocessing={**contract.preprocessing, "baseline_applied": True},
    )
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data,
            already,
            baseline_operation="subtract",
        )

    no_window = replace(
        contract,
        preprocessing={**contract.preprocessing, "baseline_window": None},
    )
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data,
            no_window,
            baseline_operation="subtract",
        )

    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data,
            contract,
            baseline_operation="subtract",
            baseline_window=(-5, -4),
        )

    arbitrary = replace(contract, pupil_unit="arbitrary_units")
    zero = data.copy()
    mask = zero["event_time"].between(-0.2, 0.0)
    zero.loc[mask, "pupil_mm"] = 0.0
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            zero,
            arbitrary,
            baseline_operation="divide",
        )

    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data,
            contract,
            scale_covariates=("condition",),
        )

    constant = data.copy()
    constant["luminance"] = 1.0
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            constant,
            contract,
            scale_covariates=("luminance",),
        )

    scaled = p.prepare_pupil_timecourse(
        data,
        contract,
        baseline_operation="subtract",
        scale_covariates=("luminance",),
    )
    assert scaled.baseline_values
    assert "luminance" in scaled.scaling
    assert p.pupil_readiness_table(scaled.audit, "summary").shape[0] >= 1
    assert p.pupil_readiness_table(scaled.audit, "participant").shape[0] >= 1
    assert p.pupil_readiness_table(scaled.audit, "condition").shape[0] >= 1
    assert p.pupil_readiness_table(scaled.audit, "trial").shape[0] >= 1
    with pytest.raises(GP3BayesError):
        p.pupil_readiness_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_readiness_table(scaled.audit, "bad")  # type: ignore[arg-type]

    audit = p.audit_pupil_measurement_context(prepared)
    assert len(p.pupil_measurement_audit_table(audit)) >= 1
    with pytest.raises(GP3BayesError):
        p.audit_pupil_measurement_context(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_measurement_audit_table(object())  # type: ignore[arg-type]


def test_base_model_specification_guard_matrix():
    _, _, prepared = _objects(2610)

    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(object())  # type: ignore[arg-type]

    one_participant = replace(
        prepared,
        data=prepared.data[
            prepared.data[".participant"] == prepared.data[".participant"].iloc[0]
        ].copy(),
    )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(one_participant)

    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            temporal_structure="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            autocorrelation="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            smooth_basis_dimension=0,
        )

    irregular = replace(
        prepared,
        timing={
            **prepared.timing,
            "cv_dt": 0.5,
            "irregularity_review_cv": 0.1,
        },
    )
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            irregular,
            autocorrelation="ar1",
        )

    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            covariates=("missing",),
        )

    arbitrary = replace(prepared, model_unit="arbitrary_units")
    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(arbitrary)

    with pytest.raises(GP3BayesError):
        p.specify_pupil_timecourse_model(
            prepared,
            prior_scales={"residual": -1},
        )

    smooth = p.specify_pupil_timecourse_model(
        prepared,
        temporal_structure="smooth",
        autocorrelation="ar1",
        participant_trajectory="factor_smooth",
        condition_trajectory=True,
        item_effects=True,
        covariates=(".luminance",),
    )
    assert "s(.event_time" in smooth.formula
    assert "ar(p=1)" in smooth.formula
