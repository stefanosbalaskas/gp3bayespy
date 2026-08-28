import inspect
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import gp3bayespy as gp
from gp3bayespy import GP3BayesError


def _specification() -> Any:
    prepared = SimpleNamespace(model_matrix_columns=("(Intercept)", "condition", "covariate"))
    contract = SimpleNamespace(
        mappings={
            "outcome": "y",
            "participant": "participant_id",
            "item": "item_id",
            "trial": "trial_id",
            "condition": "condition",
            "time": None,
        }
    )
    return SimpleNamespace(prepared=prepared, contract=contract)


def _backend(*, include_sigma: bool, seed: int = 1) -> Any:
    rng = np.random.default_rng(seed)
    chains = 4
    draws = 250
    posterior = {
        "b_Intercept": rng.normal(-0.4, 0.15, size=(chains, draws)),
        "b": np.stack(
            [
                rng.normal(0.7, 0.15, size=(chains, draws)),
                rng.normal(0.2, 0.10, size=(chains, draws)),
            ],
            axis=2,
        ),
        "sd_participant": np.abs(rng.normal(0.6, 0.08, size=(chains, draws))),
        "sd_item": np.abs(rng.normal(0.3, 0.05, size=(chains, draws))),
        "participant_z": rng.normal(size=(chains, draws, 3)),
    }
    if include_sigma:
        posterior["sigma"] = np.abs(rng.normal(0.35, 0.04, size=(chains, draws)))
    sample_stats = {
        "diverging": np.zeros((chains, draws), dtype=int),
        "tree_depth": np.full((chains, draws), 4, dtype=int),
        "energy": rng.normal(size=(chains, draws)),
    }
    return SimpleNamespace(posterior=posterior, sample_stats=sample_stats)


def _binary_fit(seed: int = 1) -> gp.BinaryFit:
    return gp.BinaryFit(
        fit_version="0.1",
        family="binary",
        model_family="hierarchical_binary",
        specification=cast(Any, _specification()),
        translation=cast(Any, None),
        backend_fit=_backend(include_sigma=False, seed=seed),
        backend_model=None,
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={
            "chains": 4,
            "iter": 500,
            "warmup": 250,
            "post_warmup_iterations": 250,
            "cores": 1,
            "seed": seed,
            "adapt_delta": 0.95,
            "max_treedepth": 12,
            "refresh": 0,
        },
        package_versions={"pymc": "test", "arviz": "test"},
    )


def _duration_fit(seed: int = 2) -> gp.DurationFit:
    return gp.DurationFit(
        fit_version="0.1",
        family="duration",
        model_family="hierarchical_lognormal_duration",
        specification=cast(Any, _specification()),
        translation=cast(Any, None),
        backend_fit=_backend(include_sigma=True, seed=seed),
        backend_model=None,
        outcome_unit="ms",
        backend_interface="pymc",
        sampling_backend="pymc",
        algorithm="NUTS",
        sampling={
            "chains": 4,
            "iter": 500,
            "warmup": 250,
            "post_warmup_iterations": 250,
            "cores": 1,
            "seed": seed,
            "adapt_delta": 0.95,
            "max_treedepth": 12,
            "refresh": 0,
        },
        package_versions={"pymc": "test", "arviz": "test"},
    )


def test_extract_posterior_draws_matrix_preserves_canonical_names():
    matrix = gp.extract_posterior_draws(_binary_fit(), format="matrix")
    assert isinstance(matrix, pd.DataFrame)
    assert matrix.shape[0] == 1000
    assert list(matrix.columns[:5]) == [
        "b_Intercept",
        "b_condition",
        "b_covariate",
        "sd_participant_id__Intercept",
        "sd_item_id__Intercept",
    ]
    assert "participant_z[1]" in matrix.columns


def test_extract_posterior_draws_array_retains_named_dimensions():
    draws = gp.extract_posterior_draws(
        _binary_fit(), variables=["b_Intercept", "b_condition"], format="array"
    )
    assert draws.dims == ("chain", "draw", "variable")
    assert draws.shape == (4, 250, 2)
    assert draws.coords["variable"].values.tolist() == ["b_Intercept", "b_condition"]


def test_extract_posterior_draws_df_adds_draw_metadata():
    frame = gp.extract_posterior_draws(_binary_fit(), variables="b_Intercept", format="df")
    assert list(frame.columns) == ["b_Intercept", ".chain", ".iteration", ".draw"]
    assert frame[".chain"].nunique() == 4
    assert frame[".draw"].iloc[-1] == 1000


def test_extract_posterior_draws_rvars_is_python_mapping_adaptation():
    result = gp.extract_posterior_draws(
        _binary_fit(), variables=["b_Intercept", "b_condition"], format="rvars"
    )
    assert set(result) == {"b_Intercept", "b_condition"}
    assert result["b_condition"].shape == (4, 250)


def test_extract_posterior_draws_supports_regex_selection():
    matrix = gp.extract_posterior_draws(_duration_fit(), regex=r"^sd_", format="matrix")
    assert list(matrix.columns) == [
        "sd_participant_id__Intercept",
        "sd_item_id__Intercept",
    ]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"variables": ["missing"]}, "Unknown posterior variables"),
        ({"regex": r"^not-present$"}, "No posterior variables remain"),
        ({"format": "bad"}, "format"),
    ],
)
def test_extract_posterior_draws_validates_selection(kwargs, message):
    with pytest.raises(GP3BayesError, match=message):
        gp.extract_posterior_draws(_binary_fit(), **kwargs)


def test_extract_posterior_draws_rejects_non_fit_objects():
    with pytest.raises(GP3BayesError, match="gp3bayes_fit"):
        gp.extract_posterior_draws(object())


def test_binary_summary_has_r_style_columns_and_odds_ratios():
    summary = gp.summarise_binary_posterior(_binary_fit(), probability=0.90)
    assert summary.family == "binary"
    assert summary.probability == 0.90
    expected = {
        "variable",
        "mean",
        "median",
        "sd",
        "lower",
        "upper",
        "probability_positive",
        "rhat",
        "ess_bulk",
        "ess_tail",
        "odds_ratio_median",
        "odds_ratio_lower",
        "odds_ratio_upper",
    }
    assert expected <= set(summary.table.columns)
    population = summary.table[summary.table["variable"] == "b_condition"].iloc[0]
    assert population["odds_ratio_median"] > 1
    group_sd = summary.table[summary.table["variable"] == "sd_participant_id__Intercept"].iloc[0]
    assert np.isnan(group_sd["odds_ratio_median"])
    assert summary.posterior_summarised is True
    assert summary.convergence_claim is False
    assert summary.posterior_adequacy_established is False


def test_duration_summary_has_median_ratio_and_unit():
    summary = gp.summarise_duration_posterior(_duration_fit(), probability=0.95)
    assert summary.family == "duration"
    assert summary.outcome_unit == "ms"
    assert "sigma" in set(summary.table["variable"])
    condition = summary.table[summary.table["variable"] == "b_condition"].iloc[0]
    assert condition["median_ratio"] > 1
    sigma = summary.table[summary.table["variable"] == "sigma"].iloc[0]
    assert np.isnan(sigma["median_ratio"])


def test_summary_variables_are_exact_and_restricted_to_supported_parameters():
    summary = gp.summarise_binary_posterior(_binary_fit(), variables=["b_Intercept", "b_condition"])
    assert summary.table["variable"].tolist() == ["b_Intercept", "b_condition"]
    assert not any(summary.table["variable"].str.startswith("participant_z"))


@pytest.mark.parametrize("probability", [0, 1, -0.1, 1.1])
def test_posterior_summaries_require_open_probability(probability):
    with pytest.raises(GP3BayesError, match="probability"):
        gp.summarise_binary_posterior(_binary_fit(), probability=probability)


def test_binary_diagnostics_pass_clean_synthetic_chains_without_claiming_convergence():
    diagnostics = gp.diagnose_binary_fit(_binary_fit())
    assert diagnostics.family == "binary"
    assert diagnostics.status in {"pass", "review"}
    assert diagnostics.component_table["component"].tolist() == [
        "rhat",
        "bulk_ess_per_chain",
        "tail_ess_per_chain",
        "divergences",
        "treedepth_saturation",
        "energy_ebfmi",
    ]
    assert (
        diagnostics.component_table.loc[
            diagnostics.component_table["component"] == "divergences", "status"
        ].iloc[0]
        == "pass"
    )
    assert diagnostics.diagnostics_assessed is True
    assert diagnostics.convergence_claim is False
    assert diagnostics.posterior_adequacy_established is False


def test_duration_diagnostics_use_same_conservative_contract():
    diagnostics = gp.diagnose_duration_fit(_duration_fit())
    assert diagnostics.family == "duration"
    assert diagnostics.thresholds["rhat_pass"] == 1.01
    assert len(diagnostics.chain_table) == 4
    assert set(diagnostics.chain_table["divergences"]) == {0}


def test_diagnostics_fail_on_any_divergence():
    fit = _binary_fit()
    fit.backend_fit.sample_stats["diverging"][0, 0] = 1
    diagnostics = gp.diagnose_binary_fit(fit)
    row = diagnostics.component_table[
        diagnostics.component_table["component"] == "divergences"
    ].iloc[0]
    assert row["status"] == "fail"
    assert diagnostics.status == "fail"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rhat_pass": 0.99},
        {"rhat_pass": 1.05, "rhat_fail": 1.01},
        {"ess_per_chain_pass": 20, "ess_per_chain_fail": 30},
        {"maximum_treedepth_fraction": 1.1},
        {"ebfmi_pass": 0.2, "ebfmi_fail": 0.3},
    ],
)
def test_diagnostic_thresholds_are_restricted(kwargs):
    with pytest.raises(GP3BayesError):
        gp.diagnose_binary_fit(_binary_fit(), **kwargs)


def test_binary_and_duration_posterior_api_signatures_match_r_argument_order():
    expected = {
        "diagnose_binary_fit": [
            "fit",
            "rhat_pass",
            "rhat_fail",
            "ess_per_chain_pass",
            "ess_per_chain_fail",
            "maximum_treedepth_fraction",
            "ebfmi_pass",
            "ebfmi_fail",
        ],
        "summarise_binary_posterior": ["fit", "probability", "variables"],
        "diagnose_duration_fit": [
            "fit",
            "rhat_pass",
            "rhat_fail",
            "ess_per_chain_pass",
            "ess_per_chain_fail",
            "maximum_treedepth_fraction",
            "ebfmi_pass",
            "ebfmi_fail",
        ],
        "summarise_duration_posterior": ["fit", "probability", "variables"],
        "extract_posterior_draws": ["fit", "variables", "regex", "format"],
    }
    for name, arguments in expected.items():
        assert list(inspect.signature(getattr(gp, name)).parameters) == arguments
