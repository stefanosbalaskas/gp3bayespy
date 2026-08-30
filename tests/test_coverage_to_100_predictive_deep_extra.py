from __future__ import annotations

from itertools import product
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
import gp3bayespy.binary as binary
import gp3bayespy.duration as duration
import gp3bayespy.predictive as p
from gp3bayespy.exceptions import GP3BayesError


def _binary_fit(seed: int = 1901):
    sim = gp.simulate_hierarchical_binary_data(
        n_participants=5,
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
        predictors=(
            "participant_covariate",
            "trial_covariate",
        ),
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        sim.data,
        contract,
    )
    spec = gp.specify_binary_model(prepared)
    return binary.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec,
        translation=SimpleNamespace(formula_text="selected ~ condition"),
        backend_fit=SimpleNamespace(posterior={"dummy": np.ones((2, 10))}),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )


def _duration_fit(seed: int = 1902):
    sim = gp.simulate_hierarchical_duration_data(
        n_participants=5,
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
        predictors=(
            "participant_covariate",
            "trial_covariate",
        ),
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        sim.data,
        contract,
    )
    spec = gp.specify_duration_model(
        prepared,
        baseline=500.0,
    )
    return duration.DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_duration",
        specification=spec,
        translation=SimpleNamespace(formula_text="duration ~ condition"),
        backend_fit=SimpleNamespace(posterior={"dummy": np.ones((2, 10))}),
        backend_model=None,
        outcome_unit="milliseconds",
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"seed": seed},
        package_versions={"gp3bayespy": "0.5.0"},
    )


def _support(rows: int):
    return p.PredictionSupport(
        table=pd.DataFrame(
            {
                "variable": ["x"],
                "type": ["numeric"],
                "training_min": [0.0],
                "training_max": [1.0],
                "outside_support": [0],
                "novel_levels": [np.nan],
                "missing_values": [0],
                "detail": ["within"],
            }
        ),
        rows=rows,
        has_extrapolation=False,
        has_novel_levels=False,
        has_missing_required=False,
    )


def _install_prediction_stubs(monkeypatch):
    def fake_grid(
        fit,
        variables=None,
        at=None,
        max_rows=5000,
        **kwargs,
    ):
        data = fit.specification.prepared.data
        vars_list = [variables] if isinstance(variables, str) else list(variables or [])
        at_map = dict(at or {})
        values = []
        for name in vars_list:
            value = at_map.get(name)
            if value is None:
                values.append([data[name].iloc[0]])
            elif isinstance(value, (str, bytes)) or np.isscalar(value):
                values.append([value])
            else:
                values.append(list(value))

        combos = list(product(*values)) if vars_list else [()]
        if len(combos) > max_rows:
            raise GP3BayesError("grid too large")

        frame = pd.DataFrame(
            combos,
            columns=vars_list,
        )
        for name in data.columns:
            if name in frame:
                continue
            series = data[name]
            if pd.api.types.is_numeric_dtype(series):
                frame[name] = float(
                    pd.to_numeric(
                        series,
                        errors="coerce",
                    ).median()
                )
            else:
                frame[name] = series.iloc[0]
        return frame

    def fake_predict(
        fit,
        newdata=None,
        type="expected",
        include_group_effects=False,
        allow_new_levels=False,
        ndraws=None,
        probs=(0.025, 0.5, 0.975),
        seed=1,
        **kwargs,
    ):
        data = fit.specification.prepared.data if newdata is None else newdata
        rows = len(data)
        draws_n = int(ndraws or 60)
        x = (
            pd.to_numeric(
                data["participant_covariate"],
                errors="coerce",
            ).to_numpy(float)
            if "participant_covariate" in data
            else np.linspace(
                0.0,
                1.0,
                rows,
            )
        )
        if fit.family == "binary":
            cond = (
                data["condition"].astype(str).to_numpy()
                if "condition" in data
                else np.array(["A"] * rows)
            )
            base = 1 / (1 + np.exp(-0.5 * x))
            base = np.clip(
                base
                + np.where(
                    cond == "B",
                    0.1,
                    0.0,
                ),
                0.02,
                0.98,
            )
            draws = np.tile(
                base,
                (draws_n, 1),
            )
        else:
            base = 500.0 + 50.0 * x
            draws = np.tile(
                base,
                (draws_n, 1),
            )

        summary = p._prediction_summary(
            draws,
            tuple(probs),
            None,
        )
        return p.Prediction(
            family=fit.family,
            type=type,
            scale="response",
            draws=draws,
            summary=summary,
            newdata=data.reset_index(drop=True).copy(),
            observed=None,
            support=_support(rows),
            include_group_effects=(include_group_effects),
            allow_new_levels=(allow_new_levels),
            probs=tuple(probs),
            seed=int(seed),
        )

    monkeypatch.setattr(
        p,
        "create_prediction_grid",
        fake_grid,
    )
    monkeypatch.setattr(
        p,
        "predict_model",
        fake_predict,
    )


def test_profile_surface_contrast_constructors(monkeypatch):
    _install_prediction_stubs(monkeypatch)
    fit = _binary_fit()

    numeric = p._profile_numeric(
        fit,
        "participant_covariate",
    )
    assert pd.api.types.is_numeric_dtype(numeric)
    with pytest.raises(GP3BayesError):
        p._profile_numeric(
            fit,
            "missing",
        )
    with pytest.raises(GP3BayesError):
        p._profile_numeric(
            fit,
            "participant_id",
        )

    values = p._profile_values(
        numeric,
        None,
        5,
        "participant_covariate",
    )
    assert len(values) == 5
    assert np.all(np.diff(values) > 0)

    with pytest.raises(GP3BayesError):
        p._profile_values(
            numeric,
            None,
            1,
            "participant_covariate",
        )
    with pytest.raises(GP3BayesError):
        p._profile_values(
            pd.Series([1.0, 1.0]),
            None,
            5,
            "x",
        )
    with pytest.raises(GP3BayesError):
        p._profile_values(
            numeric,
            ["x", "y"],
            5,
            "x",
        )
    with pytest.raises(GP3BayesError):
        p._profile_values(
            numeric,
            [1.0],
            5,
            "x",
        )
    with pytest.raises(GP3BayesError):
        p._profile_values(
            numeric,
            [1.0, np.inf],
            5,
            "x",
        )

    assert p._named_at(None) == {}
    assert p._named_at({"x": 1}) == {"x": 1}
    with pytest.raises(GP3BayesError):
        p._named_at(
            {1: "x"}  # type: ignore[dict-item]
        )

    profile = p.create_prediction_profile(
        fit,
        "participant_covariate",
        n=6,
        ndraws=40,
    )
    assert len(p.prediction_profile_table(profile)) == 6
    assert len(p.prediction_gradient_table(profile)) == 5

    surface = p.create_prediction_surface(
        fit,
        "participant_covariate",
        "trial_covariate",
        x_values=(-1.0, 0.0, 1.0),
        y_values=(-1.0, 1.0),
        ndraws=30,
    )
    assert len(p.prediction_surface_table(surface)) == 6

    with pytest.raises(GP3BayesError):
        p.create_prediction_surface(
            fit,
            "participant_covariate",
            "participant_covariate",
        )
    with pytest.raises(GP3BayesError):
        p.create_prediction_surface(
            fit,
            "participant_covariate",
            "trial_covariate",
            x_values=(0.0, 1.0),
            y_values=(0.0, 1.0),
            max_rows=3,
        )
    with pytest.raises(GP3BayesError):
        p.create_prediction_surface(
            fit,
            "participant_covariate",
            "trial_covariate",
            x_values=(
                0.0,
                1.0,
                2.0,
            ),
            y_values=(
                0.0,
                1.0,
                2.0,
            ),
            max_rows=4,
        )

    difference = p.create_prediction_contrast_profile(
        fit,
        "participant_covariate",
        "condition",
        values=(-1.0, 0.0, 1.0),
        measure="difference",
        ndraws=30,
    )
    assert set(difference.table["measure"]) == {"difference"}

    ratio = p.create_prediction_contrast_profile(
        fit,
        "participant_covariate",
        "condition",
        contrast_levels=("A", "B"),
        values=(-1.0, 1.0),
        measure="ratio",
        ndraws=30,
    )
    assert np.isfinite(ratio.draws).all()

    odds = p.create_prediction_contrast_profile(
        fit,
        "participant_covariate",
        "condition",
        contrast_levels=("A", "B"),
        values=(-1.0, 1.0),
        measure="odds_ratio",
        ndraws=30,
    )
    assert np.isfinite(odds.draws).all()

    with pytest.raises(GP3BayesError):
        p.create_prediction_contrast_profile(
            fit,
            "participant_covariate",
            "missing",
        )
    with pytest.raises(GP3BayesError):
        p.create_prediction_contrast_profile(
            fit,
            "participant_covariate",
            "condition",
            contrast_levels=("A",),
        )
    with pytest.raises(GP3BayesError):
        p.create_prediction_contrast_profile(
            fit,
            "participant_covariate",
            "condition",
            measure="bad",
        )

    dfit = _duration_fit()
    with pytest.raises(GP3BayesError):
        p.create_prediction_contrast_profile(
            dfit,
            "participant_covariate",
            "condition",
            contrast_levels=("A", "B"),
            values=(-1.0, 1.0),
            measure="odds_ratio",
        )


def test_predictive_atlas_score_and_calibration_uncertainty(monkeypatch):
    _install_prediction_stubs(monkeypatch)
    bfit = _binary_fit(1910)
    dfit = _duration_fit(1911)

    atlas = p.create_predictive_distribution_atlas(
        dfit,
        ndraws=30,
        include_group_effects=False,
    )
    assert len(p.predictive_distribution_atlas_table(atlas)) == 30
    assert atlas.observed_statistics["mean"] > 0

    score_b = p.prediction_score_uncertainty(
        bfit,
        ndraws=30,
    )
    assert set(score_b.summary["metric"]) == {"brier", "log_loss"}
    assert score_b.scope == ("fitted_prepared_data")

    supplied_b = bfit.specification.prepared.data.head(8).copy()
    score_b2 = p.prediction_score_uncertainty(
        bfit,
        newdata=supplied_b,
        ndraws=20,
    )
    assert score_b2.scope == "supplied_data"

    score_d = p.prediction_score_uncertainty(
        dfit,
        ndraws=25,
    )
    assert set(score_d.summary["metric"]) == {"rmse", "mae"}

    outcome = bfit.specification.contract.mappings["outcome"]
    missing_outcome = supplied_b.drop(columns=[outcome])
    with pytest.raises(GP3BayesError):
        p.prediction_score_uncertainty(
            bfit,
            newdata=missing_outcome,
        )

    calibration = p.binary_calibration_uncertainty(
        bfit,
        bins=4,
        ndraws=30,
    )
    assert not (p.binary_calibration_uncertainty_table(calibration).empty)
    assert calibration.scope == ("fitted_prepared_data")

    supplied = bfit.specification.prepared.data.head(10).copy()
    calibration2 = p.binary_calibration_uncertainty(
        bfit,
        newdata=supplied,
        bins=3,
        ndraws=20,
    )
    assert calibration2.scope == ("supplied_data")

    with pytest.raises(GP3BayesError):
        p.binary_calibration_uncertainty(
            bfit,
            bins=1,
        )

    bad = supplied.copy()
    bad.loc[
        bad.index[0],
        outcome,
    ] = 2
    with pytest.raises(GP3BayesError):
        p.binary_calibration_uncertainty(
            bfit,
            newdata=bad,
        )

    with pytest.raises(GP3BayesError):
        p.binary_calibration_uncertainty(
            bfit,
            newdata=supplied.drop(columns=[outcome]),
        )

    with pytest.raises(GP3BayesError):
        p.binary_calibration_uncertainty(
            dfit  # type: ignore[arg-type]
        )

    one = p._atlas_stat(np.array([1.0]))
    assert np.isnan(one["sd"])
