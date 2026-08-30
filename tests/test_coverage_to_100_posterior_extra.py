from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.posterior as post
from gp3bayespy.exceptions import GP3BayesError


def _fit():
    chains, draws = 2, 20
    posterior = {
        "b": np.arange(
            chains * draws * 2,
            dtype=float,
        ).reshape(chains, draws, 2),
        "sd_participant": np.ones((chains, draws)),
        "sd_item": np.ones((chains, draws)) * 2,
        "participant_chol_stds": np.ones((chains, draws, 2)),
        "participant_chol_corr": np.tile(
            np.array([[1.0, 0.2], [0.2, 1.0]]),
            (chains, draws, 1, 1),
        ),
        "sigma": np.ones((chains, draws)) * 0.5,
        "theta": np.ones((chains, draws, 2, 2)),
    }
    sample_stats = {
        "diverging": np.array([[0] * draws, [0] * (draws - 1) + [1]]),
        "tree_depth": np.array([[5] * draws, [10] * draws]),
        "energy": np.vstack(
            [
                np.linspace(1.0, 2.0, draws),
                np.linspace(2.0, 1.0, draws),
            ]
        ),
    }
    contract = SimpleNamespace(
        mappings={
            "participant": "participant_id",
            "item": "item_id",
            "condition": "condition",
        }
    )
    prepared = SimpleNamespace(
        model_matrix_columns=("Intercept", "condition_B", "x"),
        contract=contract,
    )
    specification = SimpleNamespace(
        prepared=prepared,
        contract=contract,
    )
    return SimpleNamespace(
        fit_performed=True,
        family="binary",
        backend_fit=SimpleNamespace(
            posterior=posterior,
            sample_stats=sample_stats,
        ),
        specification=specification,
    )


def test_posterior_validation_component_and_selection_paths():
    fit = _fit()
    assert post._validate_fit_like(fit) is fit
    assert post._validate_fit_like(fit, "binary") is fit

    with pytest.raises(GP3BayesError):
        post._validate_fit_like(object())
    with pytest.raises(GP3BayesError):
        post._validate_fit_like(
            SimpleNamespace(
                fit_performed=True,
                family="pupil",
                backend_fit=SimpleNamespace(posterior={}),
            )
        )
    with pytest.raises(GP3BayesError):
        post._validate_fit_like(fit, "duration")
    with pytest.raises(GP3BayesError):
        post._validate_fit_like(
            SimpleNamespace(
                fit_performed=True,
                family="binary",
                backend_fit=SimpleNamespace(posterior=None),
            )
        )

    assert post._posterior_data_vars(fit) is fit.backend_fit.posterior
    with pytest.raises(GP3BayesError):
        post._posterior_data_vars(SimpleNamespace(backend_fit=SimpleNamespace(posterior=object())))

    components = post._posterior_components(fit)
    assert {
        "b_condition_B",
        "b_x",
        "sd_participant_id__Intercept",
        "sd_item_id__Intercept",
        "sd_participant_id__condition",
        "cor_participant_id__Intercept__condition",
        "sigma",
        "theta[1,1]",
        "theta[2,2]",
    }.issubset(components)

    selected = post._select_components(
        fit,
        regex=r"^b_",
        parameters_only=True,
    )
    assert set(selected) == {"b_condition_B", "b_x"}
    assert set(post._select_components(fit, variables="sigma")) == {"sigma"}

    for bad in ([], [""]):
        with pytest.raises(GP3BayesError):
            post._select_components(fit, variables=bad)
    with pytest.raises(GP3BayesError):
        post._select_components(fit, variables=("missing",))
    with pytest.raises(GP3BayesError):
        post._select_components(fit, regex=5)  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        post._select_components(fit, regex="[")
    with pytest.raises(GP3BayesError):
        post._select_components(fit, regex=r"^never$")

    bad_dim = _fit()
    bad_dim.backend_fit.posterior = {"x": np.ones(5)}
    with pytest.raises(GP3BayesError):
        post._posterior_components(bad_dim)

    no_components = _fit()
    no_components.backend_fit.posterior = {}
    with pytest.raises(GP3BayesError):
        post._posterior_components(no_components)

    nonparam = _fit()
    nonparam.backend_fit.posterior = {"latent": np.ones((2, 10))}
    with pytest.raises(GP3BayesError):
        post._select_components(
            nonparam,
            parameters_only=True,
        )

    assert post._component_name("x", (0, 1)) == "x[1,2]"
    assert post._mapping_name(fit, "item") == "item_id"
    assert post._mapping_name(fit, "missing") is None
    assert post._prepared_model_columns(fit) == (
        "Intercept",
        "condition_B",
        "x",
    )
    assert np.allclose(post._array_values([1, 2]), [1, 2])


def test_extract_draw_formats_numeric_classifiers_and_chain_stats():
    fit = _fit()
    array = post.extract_draws(
        fit,
        variables=("sigma",),
        format="array",
    )
    assert array.shape == (2, 20, 1)

    matrix = post.extract_draws(
        fit,
        variables=(
            "sigma",
            "sd_participant_id__Intercept",
        ),
        format="matrix",
    )
    assert matrix.shape == (40, 2)

    rvars = post.extract_draws(
        fit,
        variables="sigma",
        format="rvars",
    )
    assert set(rvars) == {"sigma"}

    frame = post.extract_draws(
        fit,
        variables="sigma",
        format="df",
    )
    assert {
        ".chain",
        ".iteration",
        ".draw",
        "sigma",
    }.issubset(frame.columns)

    with pytest.raises(GP3BayesError):
        post.extract_draws(fit, format="bad")

    for value in (True, "x", np.inf):
        with pytest.raises(GP3BayesError):
            post._numeric_scalar(value, "x")
    assert (
        post._numeric_scalar(
            0.5,
            "x",
            lower=0,
            upper=1,
        )
        == 0.5
    )
    with pytest.raises(GP3BayesError):
        post._numeric_scalar(
            0,
            "x",
            lower=0,
            lower_open=True,
        )
    assert (
        post._validate_probability(
            0.5,
            "p",
            open=True,
        )
        == 0.5
    )

    assert (
        post._dataset_scalar(
            {"x": np.array([3.0])},
            "x",
        )
        == 3.0
    )
    assert np.isnan(post._dataset_scalar({}, "x"))
    assert np.isnan(
        post._dataset_scalar(
            {"x": np.array([])},
            "x",
        )
    )

    assert post._classify_upper(np.nan, 1, 2) == "not_assessed"
    assert post._classify_upper(1, 1, 2) == "pass"
    assert post._classify_upper(1.5, 1, 2) == "review"
    assert post._classify_upper(3, 1, 2) == "fail"

    assert post._classify_lower(np.nan, 2, 1) == "not_assessed"
    assert post._classify_lower(2, 2, 1) == "pass"
    assert post._classify_lower(1.5, 2, 1) == "review"
    assert post._classify_lower(0, 2, 1) == "fail"

    assert post._worst_status([]) == "review"
    assert post._worst_status(["pass"]) == "pass"
    assert post._worst_status(["pass", "review"]) == "review"
    assert post._worst_status(["pass", "fail"]) == "fail"
    assert post._worst_status(["not_applicable"]) == "review"

    stats = fit.backend_fit.sample_stats
    assert post._sample_stats_array(
        stats,
        ("diverging",),
    ).shape == (2, 20)
    assert (
        post._sample_stats_array(
            stats,
            ("missing",),
        )
        is None
    )
    assert (
        post._sample_stats_array(
            object(),
            ("x",),
        )
        is None
    )
    assert (
        post._sample_stats_array(
            {"x": np.ones(3)},
            ("x",),
        )
        is None
    )

    chains = post._chain_table(
        fit,
        max_treedepth=10,
    )
    assert len(chains) == 2
    assert chains["divergences"].sum() == 1
    assert chains["treedepth_hits"].sum() == 20
    assert np.isfinite(chains["ebfmi"]).all()

    no_stats = _fit()
    no_stats.backend_fit.sample_stats = None
    empty = post._chain_table(
        no_stats,
        max_treedepth=10,
    )
    assert empty["divergences"].isna().all()
    assert empty["treedepth_hits"].isna().all()


def test_posterior_result_reprs():
    sampling = post._SamplingDiagnosticsResult(
        "0.1",
        "binary",
        "review",
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {"rhat": 1.01},
    )
    assert "gp3bayes_sampling_diagnostics" in repr(sampling)

    summary = post._PosteriorSummaryResult(
        "0.1",
        "duration",
        0.95,
        pd.DataFrame({"variable": ["x"]}),
        {"x": "response"},
        outcome_unit="milliseconds",
    )
    text = repr(summary)
    assert "gp3bayes_duration_posterior_summary" in text
    assert "Outcome unit: milliseconds" in text
