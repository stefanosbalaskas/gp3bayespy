from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as binary
import gp3bayespy.duration as duration
import gp3bayespy.predictive as p
from gp3bayespy.exceptions import GP3BayesError


def _arr(values):
    return SimpleNamespace(values=np.asarray(values, dtype=float))


def _binary_fit(seed: int = 3401, random_slope: bool = False):
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
    fit = binary.BinaryFit(
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
    return fit


def _duration_fit(seed: int = 3402):
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
    fit = duration.DurationFit(
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
    return fit


def test_predictive_posterior_and_group_effect_branch_matrix():
    fit = _binary_fit()
    data = fit.specification.prepared.data.head(4).copy()
    posterior = fit.backend_fit.posterior

    assert p._draw_count(fit) == 8
    assert p._take_draws(np.arange(8), 3).tolist() == [0, 1, 2]

    with pytest.raises(GP3BayesError):
        p._posterior_values(fit, "missing")

    original = posterior["b_Intercept"]
    posterior["b_Intercept"] = _arr(np.ones((8,)))
    with pytest.raises(GP3BayesError):
        p._posterior_values(fit, "b_Intercept")
    posterior["b_Intercept"] = _arr(np.array([[0.0, np.nan]]))
    with pytest.raises(GP3BayesError):
        p._posterior_values(fit, "b_Intercept")
    posterior["b_Intercept"] = original

    fixed_name = fit.specification.prepared.model_matrix_columns
    matrix, names = p._fixed_matrix_for_prediction(fit, data)
    assert matrix.shape[0] == len(data)
    assert tuple(names) == tuple(fixed_name)

    missing_fixed = data.drop(columns=["trial_covariate"])
    with pytest.raises(GP3BayesError):
        p._fixed_matrix_for_prediction(fit, missing_fixed)

    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            fit,
            data.drop(columns=["participant_id"]),
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )

    novel_participant = data.copy()
    novel_participant["participant_id"] = novel_participant["participant_id"].astype(object)
    novel_participant.loc[novel_participant.index[0], "participant_id"] = "new-participant"
    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            fit,
            novel_participant,
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )
    allowed = p._group_effect_matrix(
        fit,
        novel_participant,
        ndraws=4,
        allow_new_levels=True,
        seed=1,
    )
    assert allowed.shape == (4, len(data))

    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            fit,
            data.drop(columns=["item_id"]),
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )

    novel_item = data.copy()
    novel_item["item_id"] = novel_item["item_id"].astype(object)
    novel_item.loc[novel_item.index[0], "item_id"] = "new-item"
    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            fit,
            novel_item,
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )
    assert p._group_effect_matrix(
        fit,
        novel_item,
        ndraws=4,
        allow_new_levels=True,
        seed=2,
    ).shape == (4, len(data))

    original_z = posterior["participant_z"]
    posterior["participant_z"] = _arr(np.zeros((1, 8, 4, 1)))
    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            fit,
            data,
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )
    posterior["participant_z"] = original_z

    original_item_z = posterior["item_z"]
    posterior["item_z"] = _arr(np.zeros((1, 8, 3, 1)))
    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            fit,
            data,
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )
    posterior["item_z"] = original_item_z

    with pytest.raises(GP3BayesError):
        p._linear_prediction_matrix(
            fit,
            data,
            include_group_effects=False,
            allow_new_levels=False,
            ndraws=20,
            seed=1,
        )

    original_b = posterior["b"]
    posterior["b"] = _arr(np.zeros((1, 8, 1)))
    with pytest.raises(GP3BayesError):
        p._linear_prediction_matrix(
            fit,
            data,
            include_group_effects=False,
            allow_new_levels=False,
            ndraws=4,
            seed=1,
        )
    posterior["b"] = original_b

    eta = p._linear_prediction_matrix(
        fit,
        data,
        include_group_effects=True,
        allow_new_levels=False,
        ndraws=4,
        seed=1,
    )
    assert eta.shape == (4, len(data))


def test_predict_model_binary_duration_and_random_slope_matrix():
    bfit = _binary_fit(3410)
    bdata = bfit.specification.prepared.data.head(6).copy()

    for kind in ("expected", "predictive", "linear"):
        prediction = p.predict_model(
            bfit,
            newdata=bdata,
            type=kind,
            include_group_effects=True,
            ndraws=5,
            seed=11,
        )
        assert prediction.draws.shape == (5, len(bdata))
        assert len(p.prediction_table(prediction)) == len(bdata)

    with pytest.raises(GP3BayesError):
        p.predict_model(bfit, newdata=bdata, type="median")
    with pytest.raises(GP3BayesError):
        p.predict_model(bfit, newdata=bdata, type="bad")
    with pytest.raises(GP3BayesError):
        p.predict_model(bfit, newdata=pd.DataFrame())

    dfit = _duration_fit()
    ddata = dfit.specification.prepared.data.head(6).copy()
    for kind in ("expected", "predictive", "linear", "median"):
        prediction = p.predict_model(
            dfit,
            newdata=ddata,
            type=kind,
            include_group_effects=False,
            ndraws=5,
            seed=12,
        )
        assert np.isfinite(prediction.draws).all()

    slope = _binary_fit(3420, random_slope=True)
    sdata = slope.specification.prepared.data.head(5).copy()
    assert p._group_effect_matrix(
        slope,
        sdata,
        ndraws=4,
        allow_new_levels=False,
        seed=1,
    ).shape == (4, len(sdata))

    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            slope,
            sdata.drop(columns=["condition"]),
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )

    novel = sdata.copy()
    novel["participant_id"] = novel["participant_id"].astype(object)
    novel.loc[novel.index[0], "participant_id"] = "future-p"
    assert p._group_effect_matrix(
        slope,
        novel,
        ndraws=4,
        allow_new_levels=True,
        seed=5,
    ).shape == (4, len(novel))

    original = slope.backend_fit.posterior["participant_re"]
    slope.backend_fit.posterior["participant_re"] = _arr(np.zeros((1, 8, 4, 3)))
    with pytest.raises(GP3BayesError):
        p._group_effect_matrix(
            slope,
            sdata,
            ndraws=4,
            allow_new_levels=False,
            seed=1,
        )
    slope.backend_fit.posterior["participant_re"] = original


def test_prediction_support_missing_extrapolation_novel_and_helpers():
    fit = _binary_fit(3430)
    train = fit.specification.prepared.data
    data = train.head(3).copy()

    missing = data.drop(columns=["trial_covariate"])
    audit = p.audit_prediction_support(fit, missing)
    assert audit.has_missing_required

    extrap = data.copy()
    extrap["trial_covariate"] = 1e6
    audit = p.audit_prediction_support(fit, extrap)
    assert audit.has_extrapolation

    novel = data.copy()
    novel["participant_id"] = novel["participant_id"].astype(object)
    novel.loc[novel.index[0], "participant_id"] = "novel"
    audit = p.audit_prediction_support(fit, novel)
    assert audit.has_novel_levels
    assert not p.prediction_support_table(audit).empty

    with pytest.raises(GP3BayesError):
        p.prediction_support_table(object())

    assert (
        p._prediction_summary(
            np.ones((1, 2)),
            (0.025, 0.5, 0.975),
            None,
        )["predicted_sd"]
        .eq(0)
        .all()
    )

    with pytest.raises(GP3BayesError):
        p._validate_fit(SimpleNamespace())
