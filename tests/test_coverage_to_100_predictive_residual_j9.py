from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gp3bayespy as gp
import gp3bayespy.binary as binary
import gp3bayespy.duration as duration
import gp3bayespy.predictive as p
from gp3bayespy.exceptions import GP3BayesError


def _arr(values):
    return SimpleNamespace(values=np.asarray(values, dtype=float))


def _binary_fit(seed: int = 9101, random_slope: bool = False):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=random_slope,
    )
    prepared = gp.prepare_hierarchical_binary_data(sim.data, contract)
    spec = gp.specify_binary_model(prepared)
    draws = 8
    pcols = len(prepared.model_matrix_columns)
    npart = prepared.data["participant_id"].nunique()
    nitem = prepared.data["item_id"].nunique()
    posterior = {
        "b_Intercept": _arr(np.linspace(-0.2, 0.2, draws)[None, :]),
        "b": _arr(np.zeros((1, draws, max(pcols - 1, 0)))),
        "sd_item": _arr(np.full((1, draws), 0.15)),
        "item_z": _arr(np.zeros((1, draws, nitem))),
    }
    if random_slope:
        posterior["participant_re"] = _arr(np.zeros((1, draws, npart, 2)))
    else:
        posterior["sd_participant"] = _arr(np.full((1, draws), 0.2))
        posterior["participant_z"] = _arr(np.zeros((1, draws, npart)))
    return binary.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec,
        translation=SimpleNamespace(formula_text="selected ~ condition"),
        backend_fit=SimpleNamespace(posterior=posterior),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )


def _duration_fit(seed: int = 9102):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=4,
        trials_per_participant=6,
        n_items=3,
        seed=seed,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=("participant_covariate", "trial_covariate"),
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(sim.data, contract)
    spec = gp.specify_duration_model(prepared, baseline=500.0)
    draws = 8
    pcols = len(prepared.model_matrix_columns)
    npart = prepared.data["participant_id"].nunique()
    nitem = prepared.data["item_id"].nunique()
    posterior = {
        "b_Intercept": _arr(np.full((1, draws), np.log(500.0))),
        "b": _arr(np.zeros((1, draws, max(pcols - 1, 0)))),
        "sigma": _arr(np.full((1, draws), 0.1)),
        "sd_participant": _arr(np.full((1, draws), 0.15)),
        "participant_z": _arr(np.zeros((1, draws, npart))),
        "sd_item": _arr(np.full((1, draws), 0.10)),
        "item_z": _arr(np.zeros((1, draws, nitem))),
    }
    return duration.DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_duration",
        specification=spec,
        translation=SimpleNamespace(formula_text="duration ~ condition"),
        backend_fit=SimpleNamespace(posterior=posterior),
        backend_model=None,
        outcome_unit="milliseconds",
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )


def _close(fig):
    assert fig is not None
    plt.close(fig)


def test_grid_completion_empty_explicit_and_design_mismatch(monkeypatch):
    fit = _binary_fit()
    data = fit.specification.prepared.data
    grid = p.create_prediction_grid(fit, variables=[])
    assert len(grid) == 1
    assert {"participant_id", "item_id", "condition"}.issubset(grid.columns)

    with pytest.raises(GP3BayesError):
        p.create_prediction_grid(
            fit,
            variables=[],
            at={"participant_id": []},
        )

    original = p._fixed_model_matrix
    monkeypatch.setattr(
        p,
        "_fixed_model_matrix",
        lambda frame, contract: (
            np.ones((len(frame), 1), dtype=float),
            ("wrong-column",),
        ),
    )
    with pytest.raises(GP3BayesError):
        p._fixed_matrix_for_prediction(fit, data.head(2).copy())
    monkeypatch.setattr(p, "_fixed_model_matrix", original)


def test_repeated_novel_group_levels_and_beta_vector_branch():
    fit = _binary_fit(9110)
    data = fit.specification.prepared.data.head(4).copy()

    novel_p = data.copy()
    novel_p["participant_id"] = novel_p["participant_id"].astype(object)
    novel_p.iloc[0, novel_p.columns.get_loc("participant_id")] = "future-p"
    novel_p.iloc[1, novel_p.columns.get_loc("participant_id")] = "future-p"
    assert p._group_effect_matrix(fit, novel_p, ndraws=4, allow_new_levels=True, seed=7).shape == (
        4,
        len(novel_p),
    )

    novel_i = data.copy()
    novel_i["item_id"] = novel_i["item_id"].astype(object)
    novel_i.iloc[0, novel_i.columns.get_loc("item_id")] = "future-i"
    novel_i.iloc[1, novel_i.columns.get_loc("item_id")] = "future-i"
    assert p._group_effect_matrix(fit, novel_i, ndraws=4, allow_new_levels=True, seed=8).shape == (
        4,
        len(novel_i),
    )

    slope = _binary_fit(9111, random_slope=True)
    sdata = slope.specification.prepared.data.head(4).copy()
    sdata["participant_id"] = sdata["participant_id"].astype(object)
    sdata.iloc[0, sdata.columns.get_loc("participant_id")] = "future-slope"
    sdata.iloc[1, sdata.columns.get_loc("participant_id")] = "future-slope"
    assert p._group_effect_matrix(slope, sdata, ndraws=4, allow_new_levels=True, seed=9).shape == (
        4,
        len(sdata),
    )

    original_b = fit.backend_fit.posterior["b"]
    fit.backend_fit.posterior["b"] = _arr(np.zeros((1, 8)))
    with pytest.raises(GP3BayesError):
        p._linear_prediction_matrix(
            fit,
            data,
            include_group_effects=False,
            allow_new_levels=False,
            ndraws=4,
            seed=1,
        )
    fit.backend_fit.posterior["b"] = original_b


def test_predict_model_nonfinite_guard_and_wrapper_edges(monkeypatch):
    bfit = _binary_fit(9120)
    bdata = bfit.specification.prepared.data.head(3).copy()

    original = p._linear_prediction_matrix
    monkeypatch.setattr(
        p,
        "_linear_prediction_matrix",
        lambda *args, **kwargs: np.full((2, len(bdata)), np.nan),
    )
    with pytest.raises(GP3BayesError, match="finite matrix"):
        p.predict_model(bfit, newdata=bdata, type="expected", ndraws=2)
    monkeypatch.setattr(p, "_linear_prediction_matrix", original)

    dfit = _duration_fit(9121)
    with pytest.raises(GP3BayesError):
        p.predict_duration(dfit, type="bad")  # type: ignore[arg-type]

    with pytest.raises(GP3BayesError):
        p._prediction_inputs(["not-numeric"], [1.0])
    with pytest.raises(GP3BayesError):
        p._prediction_inputs([0.2, 0.3], ["bad", "data"])


def test_exceedance_one_draw_uncertainty_and_group_guards():
    bfit = _binary_fit(9130)
    pred = p.predict_model(bfit, type="expected", include_group_effects=False, ndraws=4)
    below = p.prediction_exceedance_probability(pred, 0.5, direction="below")
    assert (below["direction"] == "below").all()

    one = p.prediction_uncertainty_decomposition(
        bfit,
        include_group_effects=False,
        ndraws=1,
        seed=3,
    )
    assert one.table["expected_response_variance"].isna().all()
    assert one.table["total_predictive_variance"].isna().all()

    with pytest.raises(GP3BayesError):
        p.grouped_prediction_check(bfit, group="")
    with pytest.raises(GP3BayesError):
        p.grouped_prediction_check(bfit, group="not_a_column")


def test_residual_observed_guard_and_binary_input_conversion(monkeypatch):
    bfit = _binary_fit(9140)
    base = p.predict_model(bfit, type="expected", include_group_effects=True, ndraws=4)
    no_observed = replace(base, observed=None)
    monkeypatch.setattr(p, "predict_model", lambda *args, **kwargs: no_observed)
    with pytest.raises(GP3BayesError, match="Observed outcomes"):
        p.predictive_residuals(bfit, ndraws=4)

    with pytest.raises(GP3BayesError):
        p._binary_probability_inputs(["bad"], [0])
    with pytest.raises(GP3BayesError):
        p._binary_probability_inputs([0.2, 0.8], ["bad", "bad"])
    with pytest.raises(GP3BayesError):
        p._binary_probability_inputs([0.2], None)


def test_predictive_statistics_thresholds_and_row_selectors():
    bfit = _binary_fit(9150)
    pred = p.predict_model(
        bfit,
        type="predictive",
        include_group_effects=False,
        ndraws=4,
        seed=5,
    )

    with pytest.raises(GP3BayesError):
        p.posterior_predictive_statistic(pred, statistic="tail_rate")
    tail = p.posterior_predictive_statistic(pred, statistic="tail_rate", threshold=0.5)
    assert tail.threshold == 0.5

    values = np.asarray([1.0, 2.0, 3.0])
    for name in ("mean", "sd", "median", "q90", "q95", "max"):
        assert np.isfinite(p._statistic_values(values, name, None, axis=None))
    assert np.isfinite(p._statistic_values(values, "tail_rate", 1.5, axis=None))

    with pytest.raises(GP3BayesError):
        p._binary_curve_thresholds(np.asarray([0.2, 0.8]), ["bad"])
    with pytest.raises(GP3BayesError):
        p._binary_curve_thresholds(np.asarray([0.2, 0.8]), [[0.2, 0.8]])
    custom = p._binary_curve_thresholds(np.asarray([0.2, 0.8]), [0.1, 0.9, 0.1])
    assert custom.tolist() == [0.9, 0.1]

    with pytest.raises(GP3BayesError):
        p._prediction_rows(["bad"], 3)
    with pytest.raises(GP3BayesError):
        p._prediction_rows([0], 3)
    assert p._prediction_rows([2, 2, 1], 3) == [2, 1]

    expected = p.predict_model(bfit, type="expected", include_group_effects=False, ndraws=4)
    with pytest.raises(GP3BayesError):
        p.prediction_pairwise_contrasts(expected, rows=[1], max_rows=3)
    with pytest.raises(GP3BayesError):
        p.prediction_pairwise_contrasts(expected, rows=[1, 2, 3], max_rows=2)


def test_contrast_ratio_odds_and_profile_guards():
    bfit = _binary_fit(9160)
    bpred = p.predict_model(bfit, type="expected", include_group_effects=False, ndraws=4)

    zero_draws = np.asarray(bpred.draws, dtype=float).copy()
    zero_draws[:, 0] = 0.0
    zero_pred = replace(bpred, draws=zero_draws)
    with pytest.raises(GP3BayesError, match="positive denominator"):
        p.prediction_contrast(zero_pred, row1=1, row2=2, measure="ratio")

    dfit = _duration_fit(9161)
    dpred = p.predict_model(dfit, type="expected", include_group_effects=False, ndraws=4)
    with pytest.raises(GP3BayesError, match="Odds-ratio"):
        p.prediction_contrast(dpred, row1=1, row2=2, measure="odds_ratio")

    with pytest.raises(GP3BayesError):
        p.create_prediction_contrast_profile(
            bfit,
            variable="trial_covariate",
            contrast_variable="condition",
            contrast_levels=[-0.5],
            n=3,
            ndraws=4,
        )

    with pytest.raises(GP3BayesError, match="binary fit"):
        p.create_prediction_contrast_profile(
            dfit,
            variable="trial_covariate",
            contrast_variable="condition",
            contrast_levels=[-0.5, 0.5],
            values=[-0.2, 0.2],
            measure="odds_ratio",
            ndraws=4,
        )


def test_atlas_guards_quantiles_and_small_statistic():
    bfit = _binary_fit(9170)
    one = p._atlas_stat(np.asarray([1.0]))
    assert np.isnan(one["sd"])

    with pytest.raises(GP3BayesError):
        p._atlas_get(object(), 4, False, 1)  # type: ignore[arg-type]

    atlas = p.create_predictive_distribution_atlas(
        bfit,
        ndraws=4,
        include_group_effects=False,
        seed=2,
    )
    assert not p.predictive_distribution_atlas_table(atlas).empty
    assert p._atlas_get(atlas, 4, False, 1) is atlas

    with pytest.raises(GP3BayesError):
        p.predictive_quantile_envelope(
            atlas,
            probabilities=["bad"],  # type: ignore[list-item]
        )
    with pytest.raises(GP3BayesError):
        p.predictive_quantile_envelope(atlas, probabilities=[0.0, 0.5])


def test_plot_validation_and_optional_branches():
    bfit = _binary_fit(9180)
    pred = p.predict_model(bfit, type="expected", include_group_effects=False, ndraws=4)

    with pytest.raises(GP3BayesError):
        p.plot_prediction_draws(
            pred,
            observations=["bad"],  # type: ignore[list-item]
        )
    with pytest.raises(GP3BayesError):
        p.plot_prediction_draws(pred, observations=[999])

    with pytest.raises(GP3BayesError):
        p.plot_binary_group_calibration(pred)
    with pytest.raises(GP3BayesError):
        p.plot_binary_group_calibration(pd.DataFrame({"x": [1]}))

    group_table = pd.DataFrame(
        {
            "group": ["a", "b"],
            "predicted_median": [0.2, 0.8],
            "lower": [0.1, 0.7],
            "upper": [0.3, 0.9],
        }
    )
    _close(p.plot_group_predictions(group_table, "group"))

    group_with_observed = group_table.copy()
    group_with_observed["observed"] = [0.0, np.nan]
    _close(p.plot_group_predictions(group_with_observed, "group"))

    with pytest.raises(GP3BayesError):
        p.plot_group_predictions(pd.DataFrame({"group": ["a"]}), "group")

    with pytest.raises(GP3BayesError):
        p.plot_prediction_interval_width(pd.DataFrame({"observation": [1]}))
    with pytest.raises(GP3BayesError):
        p.plot_prediction_rank_probabilities(pd.DataFrame({"observation": [1]}))

    roc = p.binary_roc_curve([0.1, 0.9], [0, 1])
    pr = p.binary_precision_recall_curve([0.1, 0.9], [0, 1])
    _close(p.plot_binary_roc(roc))
    _close(p.plot_binary_precision_recall(pr))
