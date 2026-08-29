from __future__ import annotations

from dataclasses import replace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3bayespy.pupil as p
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _base_fit(seed: int = 2301):
    sim = p.simulate_pupil_timecourse(
        n_participants=3,
        trials_per_participant=3,
        n_items=3,
        sampling_frequency=10,
        time_window=(-0.2, 0.8),
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
    spec = p.specify_pupil_timecourse_model(
        prepared,
        temporal_structure="linear",
        autocorrelation="none",
    )
    fit = p.fit_pupil_model_backend(
        spec,
        backend="analytic",
        chains=1,
        iter=140,
        warmup=40,
        cores=1,
        seed=seed + 1,
    )
    return fit


def test_base_pupil_diagnostics_acf_and_ppc_components():
    fit = _base_fit()

    diagnostics = p.diagnose_pupil_fit(
        fit,
        ndraws=30,
        max_lag=4,
    )
    assert diagnostics.status in {"pass", "review"}
    assert len(diagnostics.evidence) == 5
    assert len(diagnostics.parameter_diagnostics) >= 1
    assert not diagnostics.residuals.empty
    assert not diagnostics.residual_acf.empty

    acf_from_fit = p.pupil_residual_acf(
        fit,
        max_lag=3,
        ndraws=20,
    )
    acf_from_diag = p.pupil_residual_acf(diagnostics)
    assert len(acf_from_fit) == 4
    assert len(acf_from_diag) >= 2

    with pytest.raises(GP3BayesError):
        p.diagnose_pupil_fit(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.diagnose_pupil_fit(fit, ndraws=100, max_cells=10)
    with pytest.raises(GP3BayesError):
        p.pupil_residual_acf(object())  # type: ignore[arg-type]

    empty_acf = p._acf_table(
        np.array([], dtype=float),
        pd.Series([], dtype=str),
        3,
    )
    assert len(empty_acf) == 4
    assert empty_acf["acf"].isna().all()

    constant_lag = p._mean_lag1(
        np.array([1.0, 1.0, 1.0]),
        pd.Series(["a", "a", "a"]),
    )
    assert np.isnan(constant_lag)

    ppc = p.check_pupil_posterior_predictive(
        fit,
        ndraws=30,
        probability=0.9,
        window=(0.0, 0.4),
    )
    for component in (
        "trajectory",
        "distribution",
        "features",
        "residuals",
        "residual_trajectory",
        "autocorrelation",
        "heterogeneity",
        "measurement_context",
    ):
        table = p.pupil_ppc_table(ppc, component)
        assert isinstance(table, pd.DataFrame)
        assert not table.empty

    assert "declared_window_mean" in set(ppc.features["statistic"])
    assert set(ppc.heterogeneity["grouping"]) == {"participant", "trial_series"}

    with pytest.raises(GP3BayesError):
        p.check_pupil_posterior_predictive(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.check_pupil_posterior_predictive(fit, ndraws=100, max_cells=10)
    with pytest.raises(GP3BayesError):
        p.check_pupil_posterior_predictive(
            fit,
            ndraws=20,
            window=(10.0, 11.0),
        )
    with pytest.raises(GP3BayesError):
        p.pupil_ppc_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        p.pupil_ppc_table(ppc, "bad")  # type: ignore[arg-type]

    summary = p.summarise_pupil_posterior(fit, probability=0.9)
    assert "sigma" in set(summary.table["variable"])
    assert summary.outcome_unit == "millimetres"

    with pytest.raises(GP3BayesError):
        p.summarise_pupil_posterior(object())  # type: ignore[arg-type]


def test_base_pupil_single_group_ppc_heterogeneity_branch():
    fit = _base_fit(2310)
    data = fit.specification.prepared.data.copy()

    one_series = data.loc[
        (data[".participant"] == data[".participant"].iloc[0])
        & (data[".trial"] == data[".trial"].iloc[0])
    ].reset_index(drop=True)

    one_prepared = replace(fit.specification.prepared, data=one_series)
    one_specification = replace(fit.specification, prepared=one_prepared)
    one_fit = replace(fit, specification=one_specification)

    diagnostics = p.diagnose_pupil_fit(one_fit, ndraws=10, max_lag=2)
    assert diagnostics.residuals.shape[0] == len(one_series)

    ppc = p.check_pupil_posterior_predictive(
        one_fit,
        ndraws=10,
        probability=0.8,
    )
    assert (ppc.heterogeneity["n_groups"] < 2).any()
    assert ppc.heterogeneity["observed_sd"].isna().any()
