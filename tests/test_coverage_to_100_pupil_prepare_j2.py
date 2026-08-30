from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError


def _sim(seed: int = 3701):
    return p.simulate_pupil_timecourse(
        n_participants=2,
        trials_per_participant=2,
        n_items=2,
        sampling_frequency=10,
        time_window=(-0.2, 0.4),
        baseline_window=(-0.2, 0.0),
        blink_trial_probability=0,
        seed=seed,
    )


def _contract(
    *,
    baseline_applied: bool = False,
    pupil_unit: str = "millimetres",
    validity: bool = False,
    gaze: bool = False,
    left: bool = False,
):
    return p.create_pupil_contract(
        outcome_col="pupil_mm",
        participant_col="participant_id",
        trial_col="trial_id",
        time_col="event_time",
        pupil_unit=pupil_unit,
        sampling_frequency=10,
        item_col="item_id",
        condition_col="condition",
        validity_col="valid" if validity else None,
        gaze_x_col="gaze_x" if gaze else None,
        gaze_y_col="gaze_y" if gaze else None,
        left_pupil_col="left_pupil" if left else None,
        baseline_window=(-0.2, 0.0) if not baseline_applied else (-0.2, 0.0),
        baseline_applied=baseline_applied,
    )


def test_pupil_simulation_validation_and_optional_paths():
    with pytest.raises(GP3BayesError):
        p.simulate_pupil_timecourse(
            n_participants=2,
            trials_per_participant=2,
            sampling_frequency=10,
            time_window=(-0.2, 0.4),
            baseline_window=(-0.5, 0.0),
        )
    with pytest.raises(GP3BayesError):
        p.simulate_pupil_timecourse(
            n_participants=2,
            trials_per_participant=2,
            conditions=(),
        )
    with pytest.raises(GP3BayesError):
        p.simulate_pupil_timecourse(
            n_participants=2,
            trials_per_participant=2,
            baseline_pupil=-1,
        )
    with pytest.raises(GP3BayesError):
        p.simulate_pupil_timecourse(
            n_participants=2,
            trials_per_participant=2,
            ar1=1,
        )
    with pytest.raises(GP3BayesError):
        p.simulate_pupil_timecourse(
            n_participants=2,
            trials_per_participant=2,
            sampling_frequency=10,
            time_window=(-0.2, 0.4),
            baseline_window=(-0.2, 0.0),
            max_rows=2,
        )

    sim = p.simulate_pupil_timecourse(
        n_participants=2,
        trials_per_participant=2,
        n_items=None,
        sampling_frequency=10,
        time_window=(-0.1, 0.2),
        baseline_window=(-0.1, 0.0),
        include_gaze=False,
        include_luminance=False,
        blink_trial_probability=0,
        seed=3702,
    )
    assert sim.data["item_id"].isna().all()
    assert sim.data["gaze_x"].isna().all()
    assert sim.data["luminance"].isna().all()


def test_pupil_indicator_timing_and_unit_helpers():
    assert p._indicator(pd.Series([True, False]), "x").tolist() == [True, False]
    assert p._indicator(pd.Series([1, 0]), "x").tolist() == [True, False]
    with pytest.raises(GP3BayesError):
        p._indicator(pd.Series([0, 2]), "x")

    empty = pd.DataFrame(
        {
            ".series_id": pd.Series([], dtype=object),
            ".event_time": pd.Series([], dtype=float),
        }
    )
    timing = p._timing_summary(empty)
    assert np.isnan(timing["median_dt"])

    one_gap = pd.DataFrame(
        {
            ".series_id": ["a", "a"],
            ".event_time": [0.0, 0.1],
        }
    )
    timing = p._timing_summary(one_gap)
    assert timing["cv_dt"] == 0.0

    vals = pd.Series([1.0, 2.0])
    assert p._convert_unit(vals, "millimetres", "millimetres").tolist() == [1.0, 2.0]
    assert p._convert_unit(pd.Series([0.001]), "metres", "millimetres").iloc[0] == 1.0
    assert p._convert_unit(pd.Series([1.0]), "millimetres", "metres").iloc[0] == 0.001
    with pytest.raises(GP3BayesError):
        p._convert_unit(vals, "pixels", "millimetres")


def test_prepare_pupil_timecourse_guard_matrix_and_transforms():
    sim = _sim()
    data = sim.data.copy()
    contract = _contract()

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
            baseline_operation="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(data.drop(columns="pupil_mm"), contract)

    bad = data.copy()
    bad["pupil_mm"] = bad["pupil_mm"].astype(str)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad["event_time"] = bad["event_time"].astype(str)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad.loc[bad.index[0], "event_time"] = np.nan
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad["participant_id"] = bad["participant_id"].astype(object)
    bad.loc[bad.index[0], "participant_id"] = None
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad.loc[bad.index[0], "pupil_mm"] = np.inf
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad.loc[bad.index[0], "pupil_mm"] = 0.0
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    bad = data.copy()
    bad["condition"] = bad["condition"].astype(object)
    bad.loc[bad.index[0], "condition"] = None
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, contract)

    gaze_contract = _contract(gaze=True)
    bad = data.copy()
    bad["gaze_x"] = bad["gaze_x"].astype(str)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, gaze_contract)

    valid_contract = _contract(validity=True)
    bad = data.copy()
    bad["valid"] = 2
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, valid_contract)

    left_contract = _contract(left=True)
    bad = data.copy()
    bad["left_pupil"] = "bad"
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(bad, left_contract)

    duplicated = pd.concat([data, data.iloc[[0]]], ignore_index=True)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(duplicated, contract)

    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(data, contract, output_unit="bad")

    already = _contract(baseline_applied=True)
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data,
            already,
            baseline_operation="subtract",
        )

    no_window = p.create_pupil_contract(
        outcome_col="pupil_mm",
        participant_col="participant_id",
        trial_col="trial_id",
        time_col="event_time",
        pupil_unit="millimetres",
        sampling_frequency=10,
        item_col="item_id",
        condition_col="condition",
    )
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            data,
            no_window,
            baseline_operation="subtract",
        )

    missing_baseline = data.copy()
    first_trial = missing_baseline["trial_id"].iloc[0]
    mask = (missing_baseline["trial_id"] == first_trial) & (missing_baseline["event_time"] <= 0)
    missing_baseline.loc[mask, "pupil_mm"] = np.nan
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            missing_baseline,
            contract,
            baseline_operation="subtract",
        )

    arbitrary = _contract(pupil_unit="arbitrary_units")
    zero = data.copy()
    zero["pupil_mm"] = 1.0
    zero.loc[zero["event_time"] <= 0, "pupil_mm"] = 0.0
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            zero,
            arbitrary,
            baseline_operation="divide",
        )

    for operation in ("subtract", "divide", "proportion_change", "percent_change"):
        prepared = p.prepare_pupil_timecourse(
            data,
            contract,
            baseline_operation=operation,
        )
        assert prepared.baseline_operation == operation

    converted = p.prepare_pupil_timecourse(
        data,
        contract,
        output_unit="metres",
    )
    assert converted.model_unit == "metres"

    scaled_data = data.copy()
    scaled_data["cov"] = np.linspace(0.0, 1.0, len(scaled_data))
    scaled = p.prepare_pupil_timecourse(
        scaled_data,
        contract,
        scale_covariates=("cov",),
    )
    assert "cov" in scaled.scaling

    bad_cov = data.copy()
    bad_cov["cov"] = "x"
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            bad_cov,
            contract,
            scale_covariates=("cov",),
        )

    constant = data.copy()
    constant["cov"] = 1.0
    with pytest.raises(GP3BayesError):
        p.prepare_pupil_timecourse(
            constant,
            contract,
            scale_covariates=("cov",),
        )
