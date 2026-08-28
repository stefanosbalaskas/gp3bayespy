from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy.exceptions import GP3BayesError


class _Variable:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values


class _Posterior(dict[str, _Variable]):
    pass


def _binary_spec() -> gp.BinaryModelSpecification:
    simulation = gp.simulate_hierarchical_binary_data(
        n_participants=6,
        trials_per_participant=6,
        n_items=3,
        random_slope_sd=0.0,
        seed=8101,
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        random_slope=False,
    )
    prepared = gp.prepare_hierarchical_binary_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
    )
    return gp.specify_binary_model(prepared, baseline=0.35)


def _duration_spec() -> gp.DurationModelSpecification:
    simulation = gp.simulate_hierarchical_duration_data(
        n_participants=6,
        trials_per_participant=6,
        n_items=3,
        random_slope_sd=0.0,
        seed=8102,
    )
    contract = gp.create_model_contract(
        family="duration",
        outcome_col="duration",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        random_slope=False,
        outcome_unit="milliseconds",
    )
    prepared = gp.prepare_hierarchical_duration_data(
        simulation.data,
        contract,
        condition_levels=["control", "treatment"],
    )
    return gp.specify_duration_model(prepared, baseline=500.0)


def _posterior(spec, *, duration: bool) -> _Posterior:
    data = spec.prepared.data
    participants = len(pd.unique(data["participant_id"]))
    items = len(pd.unique(data["item_id"]))
    chains, draws = 2, 4
    posterior = _Posterior(
        {
            "b_Intercept": _Variable(np.full((chains, draws), -0.2 if not duration else 6.0)),
            "b": _Variable(np.full((chains, draws, 1), 0.8 if not duration else 0.2)),
            "sd_participant": _Variable(np.full((chains, draws), 0.3)),
            "participant_z": _Variable(
                np.linspace(-1, 1, chains * draws * participants).reshape(
                    chains, draws, participants
                )
            ),
            "sd_item": _Variable(np.full((chains, draws), 0.15)),
            "item_z": _Variable(
                np.linspace(-0.5, 0.5, chains * draws * items).reshape(chains, draws, items)
            ),
        }
    )
    if duration:
        posterior["sigma"] = _Variable(np.full((chains, draws), 0.25))
    return posterior


def _binary_fit() -> gp.BinaryFit:
    spec = _binary_spec()
    translation = gp.translate_binary_model_to_brms(spec)
    return gp.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec,
        translation=translation,
        backend_fit=SimpleNamespace(posterior=_posterior(spec, duration=False)),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"chains": 2, "iter": 8, "warmup": 0},
        package_versions={},
    )


def _duration_fit() -> gp.DurationFit:
    spec = _duration_spec()
    translation = gp.translate_duration_model_to_brms(spec)
    return gp.DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_lognormal_duration",
        specification=spec,
        translation=translation,
        backend_fit=SimpleNamespace(posterior=_posterior(spec, duration=True)),
        backend_model=None,
        outcome_unit="milliseconds",
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"chains": 2, "iter": 8, "warmup": 0},
        package_versions={},
    )


def test_prediction_grid_uses_explicit_condition_values_and_completes_groups():
    spec = _binary_spec()
    grid = gp.create_prediction_grid(
        spec,
        variables="condition",
        at={"condition": [-0.5, 0.5]},
    )
    assert grid["condition"].tolist() == [-0.5, 0.5]
    assert {"participant_id", "item_id"}.issubset(grid.columns)


def test_prediction_grid_defaults_numeric_condition_to_typical_value():
    grid = gp.create_prediction_grid(_binary_spec())
    assert len(grid) == 1
    assert float(grid.loc[0, "condition"]) == pytest.approx(0.0)


def test_prediction_grid_validates_named_at_and_row_limit():
    spec = _binary_spec()
    with pytest.raises(GP3BayesError, match="Unknown `at`"):
        gp.create_prediction_grid(spec, at={"not_a_column": [1]})
    with pytest.raises(GP3BayesError, match="would contain"):
        gp.create_prediction_grid(
            spec,
            variables=["condition", "participant_id"],
            at={
                "condition": [-0.5, 0.5],
                "participant_id": ["a", "b", "c"],
            },
            max_rows=5,
        )


def test_prediction_grid_rejects_unsupported_object():
    with pytest.raises(GP3BayesError, match="fit or model specification"):
        gp.create_prediction_grid(object())


def test_support_audit_flags_extrapolation_without_rejection():
    fit = _binary_fit()
    newdata = fit.specification.prepared.data.head(2).copy()
    newdata.loc[:, "condition"] = [5.0, -5.0]
    support = gp.audit_prediction_support(fit, newdata)
    assert support.has_extrapolation
    assert not support.automatic_rejection
    assert gp.prediction_support_table(support).equals(support.table)


def test_support_audit_flags_missing_required_without_row_removal():
    fit = _binary_fit()
    newdata = fit.specification.prepared.data.head(2).drop(columns=["item_id"])
    support = gp.audit_prediction_support(fit, newdata)
    assert support.rows == 2
    assert support.has_missing_required
    assert not support.automatic_rejection


def test_binary_expected_and_linear_predictions_are_distinct_and_finite():
    fit = _binary_fit()
    newdata = gp.create_prediction_grid(
        fit,
        variables="condition",
        at={"condition": [-0.5, 0.5]},
    )
    linear = gp.predict_model(fit, newdata, type="linear", ndraws=5)
    expected = gp.predict_model(fit, newdata, type="expected", ndraws=5)
    assert linear.scale == "log_odds"
    assert expected.scale == "response"
    assert linear.draws.shape == expected.draws.shape == (5, 2)
    np.testing.assert_allclose(expected.draws, 1 / (1 + np.exp(-linear.draws)))
    assert np.isfinite(expected.draws).all()


def test_binary_predictive_draws_are_seeded_zero_one_values():
    fit = _binary_fit()
    newdata = fit.specification.prepared.data.head(3)
    one = gp.extract_posterior_predictions(fit, newdata, ndraws=6, seed=22)
    two = gp.extract_posterior_predictions(fit, newdata, ndraws=6, seed=22)
    np.testing.assert_array_equal(one, two)
    assert set(np.unique(one)).issubset({0.0, 1.0})


def test_binary_probability_wrapper_and_table_are_conservative():
    fit = _binary_fit()
    prediction = gp.predict_binary_probability(fit, ndraws=4)
    assert prediction.family == "binary"
    assert prediction.type == "expected"
    assert not prediction.automatic_decision
    assert not prediction.causal_effect_established
    assert not prediction.out_of_sample_adequacy_established
    table = gp.prediction_table(prediction)
    assert len(table) == len(fit.specification.prepared.data)
    assert "observed" in table.columns


def test_binary_median_prediction_is_rejected():
    with pytest.raises(GP3BayesError, match="duration models"):
        gp.predict_model(_binary_fit(), type="median")


def test_duration_median_expected_and_predictive_are_distinct():
    fit = _duration_fit()
    newdata = gp.create_prediction_grid(
        fit,
        variables="condition",
        at={"condition": [-0.5, 0.5]},
    )
    median = gp.predict_duration(fit, newdata, type="median", ndraws=5)
    expected = gp.predict_duration(fit, newdata, type="expected", ndraws=5)
    predictive = gp.predict_duration(fit, newdata, type="predictive", ndraws=5, seed=24)
    assert median.scale == "duration_median"
    assert expected.scale == predictive.scale == "response"
    assert np.all(expected.draws > median.draws)
    assert np.all(predictive.draws > 0)


def test_duration_linear_predictions_are_on_log_location_scale():
    fit = _duration_fit()
    out = gp.extract_linear_predictions(
        fit,
        fit.specification.prepared.data.head(2),
        ndraws=3,
    )
    pred = gp.predict_model(
        fit,
        fit.specification.prepared.data.head(2),
        type="linear",
        ndraws=3,
    )
    assert pred.scale == "log_duration_location"
    np.testing.assert_allclose(out, pred.draws)


def test_expected_extractor_matches_predict_model():
    fit = _binary_fit()
    data = fit.specification.prepared.data.head(2)
    expected = gp.extract_expected_predictions(fit, data, ndraws=4)
    direct = gp.predict_model(fit, data, type="expected", ndraws=4)
    np.testing.assert_allclose(expected, direct.draws)


def test_group_effects_change_predictions_for_known_levels():
    fit = _binary_fit()
    data = fit.specification.prepared.data.head(4)
    population = gp.predict_model(fit, data, ndraws=4)
    conditional = gp.predict_model(
        fit,
        data,
        include_group_effects=True,
        ndraws=4,
    )
    assert not np.allclose(population.draws, conditional.draws)


def test_new_group_levels_require_explicit_permission():
    fit = _binary_fit()
    data = fit.specification.prepared.data.head(2).copy()
    data.loc[:, "participant_id"] = "new-participant"
    with pytest.raises(GP3BayesError, match="allow_new_levels"):
        gp.predict_model(fit, data, include_group_effects=True, ndraws=4)


def test_new_group_levels_are_supported_when_explicitly_allowed():
    fit = _binary_fit()
    data = fit.specification.prepared.data.head(2).copy()
    data.loc[:, "participant_id"] = "new-participant"
    out = gp.predict_model(
        fit,
        data,
        include_group_effects=True,
        allow_new_levels=True,
        ndraws=4,
        seed=77,
    )
    assert out.draws.shape == (4, 2)
    assert out.support.has_novel_levels


def test_missing_group_columns_do_not_block_population_predictions():
    fit = _binary_fit()
    data = fit.specification.prepared.data.head(2).drop(columns=["participant_id", "item_id"])
    out = gp.predict_model(fit, data, include_group_effects=False, ndraws=4)
    assert out.draws.shape == (4, 2)
    assert out.support.has_missing_required


def test_missing_group_columns_block_conditional_predictions():
    fit = _binary_fit()
    data = fit.specification.prepared.data.head(2).drop(columns=["participant_id"])
    with pytest.raises(GP3BayesError, match="grouping variable"):
        gp.predict_model(fit, data, include_group_effects=True, ndraws=4)


def test_ndraws_cannot_exceed_available_posterior_draws():
    fit = _binary_fit()
    with pytest.raises(GP3BayesError, match="exceeds"):
        gp.predict_model(fit, ndraws=9)


def test_prediction_input_controls_are_validated():
    fit = _binary_fit()
    with pytest.raises(GP3BayesError, match="TRUE or FALSE"):
        gp.predict_model(fit, include_group_effects=1)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="non-negative integer"):
        gp.predict_model(fit, seed=-1)
    with pytest.raises(GP3BayesError, match="strictly increasing"):
        gp.predict_model(fit, probs=(0.5, 0.1, 0.9))


def test_fixed_effect_novel_categorical_level_changes_are_rejected():
    spec = _binary_spec()
    prepared = spec.prepared
    data = prepared.data.copy()
    data["extra"] = pd.Categorical(
        np.where(np.arange(len(data)) % 2 == 0, "a", "b"),
        categories=["a", "b"],
    )
    contract = gp.create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
        predictors=["extra"],
        random_slope=False,
    )
    prepared2 = gp.prepare_hierarchical_binary_data(
        data,
        contract,
        condition_levels=[-0.5, 0.5],
    )
    spec2 = gp.specify_binary_model(prepared2)
    translation = gp.translate_binary_model_to_brms(spec2)
    posterior = _posterior(spec2, duration=False)
    # Add one fixed coefficient for the categorical contrast.
    posterior["b"] = _Variable(np.full((2, 4, 2), 0.2))
    fit = gp.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=spec2,
        translation=translation,
        backend_fit=SimpleNamespace(posterior=posterior),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={"chains": 2, "iter": 8, "warmup": 0},
        package_versions={},
    )
    newdata = prepared2.data.head(2).copy()
    newdata["extra"] = "c"
    with pytest.raises(GP3BayesError, match="unsupported fixed-effect level"):
        gp.predict_model(fit, newdata, ndraws=4)


def test_prediction_repr_does_not_claim_decision_or_adequacy():
    text = repr(gp.predict_binary_probability(_binary_fit(), ndraws=3))
    assert "Automatic decision: FALSE" in text
    assert "adequacy" not in text.lower()


def test_wrong_family_convenience_wrappers_are_rejected():
    with pytest.raises(GP3BayesError, match="binary family"):
        gp.predict_binary_probability(_duration_fit())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError, match="duration family"):
        gp.predict_duration(_binary_fit())  # type: ignore[arg-type]


def test_prediction_accessors_reject_wrong_objects():
    with pytest.raises(GP3BayesError):
        gp.prediction_table(object())  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        gp.prediction_support_table(object())  # type: ignore[arg-type]
