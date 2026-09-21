import numpy as np
import pandas as pd
import pytest

try:
    from eyeprocesspy.multilevel_mediation import prepare_multilevel_mediation_data
except ImportError:  # package-isolation test path
    prepare_multilevel_mediation_data = None
from gp3bayespy.multilevel_mediation import (
    BackendUnavailableError,
    GP3BayesError,
    MultilevelMediationFit,
    check_mediation_convergence,
    compare_multilevel_mediation_models,
    create_mediation_prior_specification,
    estimate_between_indirect_effect,
    estimate_within_indirect_effect,
    posterior_direct_effect,
    posterior_total_effect,
    report_multilevel_gaze_mediation,
    simulate_multilevel_gaze_mediation,
    specify_multilevel_gaze_mediation,
    summarise_multilevel_mediation,
)


class PreparedFixture:
    def __init__(self, data: pd.DataFrame):
        frame = data.copy()
        x_mean = frame.groupby("participant_id")["ai_correct"].transform("mean")
        m_mean = frame.groupby("participant_id")["source_dwell"].transform("mean")
        frame["X_within"] = frame["ai_correct"] - x_mean
        frame["X_between"] = x_mean
        frame["M_within"] = frame["source_dwell"] - m_mean
        frame["M_between"] = m_mean
        frame["mediation_analysis_eligible"] = (
            frame[["ai_correct", "source_dwell", "correct_override"]].notna().all(axis=1)
        )
        self.data = frame
        self.columns = {
            "participant": "participant_id",
            "trial": "trial_id",
            "x": "ai_correct",
            "mediator": "source_dwell",
            "outcome": "correct_override",
            "quality": "valid_fraction",
        }
        self.provenance = {"fixture": True, "row_position_convention": {"base": 0}}


def _prepare(data: pd.DataFrame):
    if prepare_multilevel_mediation_data is None:
        return PreparedFixture(data)
    return prepare_multilevel_mediation_data(
        data,
        x_col="ai_correct",
        mediator_col="source_dwell",
        outcome_col="correct_override",
        quality_col="valid_fraction",
        minimum_quality=0.8,
        source_id="synthetic-ai-advice",
        warn=False,
    )


def prepared_binary(n=8, trials=6, seed=3):
    data = simulate_multilevel_gaze_mediation(
        n_participants=n, trials_per_participant=trials, seed=seed
    )
    return _prepare(data)


def fake_fit(prepared, diagnostics=None, scale=1.0):
    spec = specify_multilevel_gaze_mediation(
        prepared,
        mediator_family="gaussian",
        outcome_family="bernoulli",
        missingness_policy="error",
        random_slopes=(),
    )
    x = np.linspace(-0.2, 0.4, 1000)
    posterior = {
        "a_within": x + 0.6,
        "b_within": x + 0.5,
        "a_between": x + 0.2,
        "b_between": x + 0.1,
        "cprime_within": x * 0.2 + 0.15,
        "cprime_between": x * 0.1 + 0.05,
    }
    posterior["indirect_within"] = posterior["a_within"] * posterior["b_within"] * scale
    posterior["indirect_between"] = posterior["a_between"] * posterior["b_between"] * scale
    posterior["total_within"] = posterior["cprime_within"] + posterior["indirect_within"]
    posterior["total_between"] = posterior["cprime_between"] + posterior["indirect_between"]
    return MultilevelMediationFit(
        specification=spec,
        posterior=posterior,
        sampler_diagnostics=diagnostics
        or {"max_rhat": 1.001, "min_ess_bulk": 800, "divergences": 0, "treedepth_hits": 0},
        backend_fit=None,
        backend_model=None,
        backend="fixture",
    )


def test_simulator_is_deterministic():
    a = simulate_multilevel_gaze_mediation(n_participants=5, trials_per_participant=4, seed=19)
    b = simulate_multilevel_gaze_mediation(n_participants=5, trials_per_participant=4, seed=19)
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 20
    assert set(a["correct_override"].unique()).issubset({0, 1})


def test_specification_preserves_within_between_contract():
    prepared = prepared_binary()
    spec = specify_multilevel_gaze_mediation(
        prepared,
        mediator_family="normal",
        outcome_family="binary",
        random_slopes=("mediator_x",),
    )
    assert spec.mediator_family == "gaussian"
    assert spec.outcome_family == "bernoulli"
    assert spec.analysis_rows == spec.input_rows
    assert spec.excluded_row_positions == ()
    assert "preparation" in spec.provenance
    assert spec.provenance["model_specification"]["estimand_scale"] == "linear_predictor_product"


def test_missingness_policy_defaults_to_error():
    data = simulate_multilevel_gaze_mediation(n_participants=5, trials_per_participant=4, seed=4)
    data.loc[3, "source_dwell"] = np.nan
    prepared = PreparedFixture(data)
    with pytest.raises(GP3BayesError, match="explicit `missingness_policy`"):
        specify_multilevel_gaze_mediation(prepared)
    spec = specify_multilevel_gaze_mediation(prepared, missingness_policy="complete_case")
    assert spec.input_rows == 20
    assert spec.analysis_rows == 19
    assert spec.excluded_row_positions == (3,)


def test_quality_eligible_policy_is_explicit():
    data = simulate_multilevel_gaze_mediation(n_participants=5, trials_per_participant=4, seed=5)
    data.loc[4, "valid_fraction"] = 0.2
    prepared = PreparedFixture(data)
    prepared.data.loc[4, "mediation_analysis_eligible"] = False
    spec = specify_multilevel_gaze_mediation(prepared, missingness_policy="quality_eligible")
    assert spec.analysis_rows == 19
    assert spec.excluded_row_positions == (4,)


def test_family_validation_rejects_invalid_lognormal_mediator():
    prepared = prepared_binary()
    prepared.data.loc[prepared.data.index[0], "source_dwell"] = 0.0
    prepared.data.loc[prepared.data.index[0], "M_within"] = 0.0
    with pytest.raises(GP3BayesError, match="strictly positive"):
        specify_multilevel_gaze_mediation(prepared, mediator_family="lognormal")


def test_beta_and_count_family_validation():
    prepared = prepared_binary()
    with pytest.raises(GP3BayesError, match="beta"):
        specify_multilevel_gaze_mediation(prepared, mediator_family="beta")
    with pytest.raises(GP3BayesError, match="integer counts"):
        specify_multilevel_gaze_mediation(prepared, mediator_family="poisson")


def test_priors_are_explicit_and_validated():
    priors = create_mediation_prior_specification(coefficient_sd=0.5, group_sd_scale=0.8)
    assert priors.coefficient_sd == 0.5
    assert priors.group_sd_scale == 0.8
    with pytest.raises(GP3BayesError, match="positive"):
        create_mediation_prior_specification(coefficient_sd=0)


def test_convergence_gate_blocks_indirect_effects():
    fit = fake_fit(
        prepared_binary(),
        diagnostics={"max_rhat": 1.08, "min_ess_bulk": 70, "divergences": 2, "treedepth_hits": 1},
    )
    result = check_mediation_convergence(fit)
    assert result.status == "fail"
    assert not result.passed
    assert len(result.issues) == 4
    with pytest.raises(GP3BayesError, match="Critical convergence checks failed"):
        estimate_within_indirect_effect(fit)


def test_effect_extraction_and_summary_after_passing_diagnostics():
    fit = fake_fit(prepared_binary())
    within = estimate_within_indirect_effect(fit)
    between = estimate_between_indirect_effect(fit)
    direct = posterior_direct_effect(fit)
    total = posterior_total_effect(fit)
    assert within.scale == "linear_predictor_product"
    assert within.mean > between.mean
    assert direct.effect == "cprime_within"
    assert total.effect == "total_within"
    summary = summarise_multilevel_mediation(fit)
    assert {"indirect_within", "indirect_between", "total_within", "total_between"}.issubset(
        set(summary["effect"])
    )


def test_report_states_scale_missingness_and_diagnostics():
    fit = fake_fit(prepared_binary())
    report = report_multilevel_gaze_mediation(fit)
    assert "trial-level multilevel mediation" in report
    assert "linear-predictor product scale" in report
    assert "missingness policy" in report
    assert "Sampler diagnostics status: pass" in report
    assert "not be interpreted as probability-scale natural indirect effects" in report


def test_random_slope_names_are_validated():
    with pytest.raises(GP3BayesError, match="Unknown random slope"):
        specify_multilevel_gaze_mediation(prepared_binary(), random_slopes=("magic",))


def test_backend_failure_is_informative_in_current_test_environment():
    # The tranche unit suite validates backend-independent contracts regardless of
    # optional backend state. If PyMC imports successfully, this test is skipped.
    import importlib

    try:
        importlib.import_module("pymc")
    except Exception:
        from gp3bayespy.multilevel_mediation import _load_pymc

        with pytest.raises(BackendUnavailableError, match="PyMC could not be imported"):
            _load_pymc()
    else:
        pytest.skip("PyMC backend imports successfully in this environment")


def test_serial_specification_requires_eyeprocess_prepared_second_mediator():
    ep = pytest.importorskip("eyeprocesspy.multilevel_mediation")
    add_multilevel_mediation_component = ep.add_multilevel_mediation_component
    from gp3bayespy.multilevel_mediation import specify_multilevel_serial_gaze_mediation

    data = simulate_multilevel_gaze_mediation(n_participants=6, trials_per_participant=5, seed=17)
    rng = np.random.default_rng(17)
    data["trust"] = (
        3.0
        + 0.003 * data["source_dwell"]
        + 0.4 * data["ai_correct"]
        + rng.normal(0, 0.2, len(data))
    )
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="ai_correct",
        mediator_col="source_dwell",
        outcome_col="correct_override",
        warn=False,
    )
    with pytest.raises(GP3BayesError, match="mediator2"):
        specify_multilevel_serial_gaze_mediation(prepared)
    augmented = add_multilevel_mediation_component(
        prepared,
        value_col="trust",
        semantic="mediator2",
        within_col="M2_within",
        between_col="M2_between",
    )
    spec = specify_multilevel_serial_gaze_mediation(augmented)
    assert spec.model_kind == "serial"
    assert spec.serial is True
    assert spec.extra_columns["mediator2"] == "trust"
    assert spec.analysis_rows == len(data)


def test_moderated_specification_and_conditional_indirect_effect():
    ep = pytest.importorskip("eyeprocesspy.multilevel_mediation")
    add_multilevel_mediation_component = ep.add_multilevel_mediation_component
    from gp3bayespy.multilevel_mediation import (
        posterior_conditional_indirect_effect,
        specify_multilevel_moderated_gaze_mediation,
    )

    data = simulate_multilevel_gaze_mediation(n_participants=6, trials_per_participant=5, seed=23)
    data["warning_salience"] = np.tile([-0.5, 0.5, -0.5, 0.5, 0.0], 6)
    prepared = prepare_multilevel_mediation_data(
        data,
        x_col="ai_correct",
        mediator_col="source_dwell",
        outcome_col="correct_override",
        warn=False,
    )
    augmented = add_multilevel_mediation_component(
        prepared,
        value_col="warning_salience",
        semantic="moderator",
        within_col="Z_within",
        between_col="Z_between",
    )
    spec = specify_multilevel_moderated_gaze_mediation(
        augmented,
        moderation_path="a",
        moderator_component="within",
    )
    assert spec.model_kind == "moderated"
    assert spec.extra_columns["moderator"] == "Z_within"

    draws = np.linspace(-0.05, 0.05, 500)
    fit = MultilevelMediationFit(
        specification=spec,
        posterior={
            "a_within": np.full(500, 0.6),
            "b_within": np.full(500, 0.7),
            "moderation": np.full(500, 0.2),
            "indirect_within": np.full(500, 0.42),
            "a_between": np.full(500, 0.1),
            "b_between": np.full(500, 0.2),
            "indirect_between": np.full(500, 0.02),
            "cprime_within": draws,
            "cprime_between": draws,
        },
        sampler_diagnostics={
            "max_rhat": 1.001,
            "min_ess_bulk": 900,
            "divergences": 0,
            "treedepth_hits": 0,
        },
        backend="fixture",
    )
    low = posterior_conditional_indirect_effect(fit, moderator_value=-1)
    high = posterior_conditional_indirect_effect(fit, moderator_value=1)
    assert high.mean > low.mean
    assert low.mean == pytest.approx((0.6 - 0.2) * 0.7)
    assert high.mean == pytest.approx((0.6 + 0.2) * 0.7)


def test_balanced_within_design_marks_between_x_paths_not_estimable():
    rows = []
    for i in range(8):
        for j in range(6):
            x = j % 2
            m = 1.0 + 0.5 * x + 0.1 * i + 0.02 * j
            rows.append(
                {
                    "participant_id": f"p{i}",
                    "trial_id": j + 1,
                    "ai_correct": x,
                    "source_dwell": m,
                    "correct_override": int((i + j) % 2 == 0),
                    "valid_fraction": 1.0,
                }
            )
    prepared = _prepare(pd.DataFrame(rows))
    spec = specify_multilevel_gaze_mediation(prepared, random_slopes=())
    assert "a_within" in spec.estimable_paths
    assert "a_between" not in spec.estimable_paths
    assert "cprime_between" not in spec.estimable_paths

    x = np.linspace(-0.1, 0.1, 200)
    fit = MultilevelMediationFit(
        specification=spec,
        posterior={
            "a_within": x + 0.6,
            "b_within": x + 0.5,
            "cprime_within": x + 0.1,
            "indirect_within": (x + 0.6) * (x + 0.5),
            "total_within": x + 0.1 + (x + 0.6) * (x + 0.5),
        },
        sampler_diagnostics={
            "max_rhat": 1.001,
            "min_ess_bulk": 900,
            "divergences": 0,
            "treedepth_hits": 0,
        },
        backend="fixture",
    )
    with pytest.raises(GP3BayesError, match="unavailable"):
        estimate_between_indirect_effect(fit)
    report = report_multilevel_gaze_mediation(fit)
    assert "between-participant indirect effect was not estimable" in report


def test_serial_and_moderated_extensions_reject_unimplemented_random_slopes():
    ep = pytest.importorskip("eyeprocesspy.multilevel_mediation")
    from gp3bayespy.multilevel_mediation import (
        specify_multilevel_moderated_gaze_mediation,
        specify_multilevel_serial_gaze_mediation,
    )

    data = simulate_multilevel_gaze_mediation(n_participants=8, trials_per_participant=6, seed=41)
    data["trust"] = 2.0 + 0.4 * data["ai_correct"] + 0.2 * data["source_dwell"]
    data["moderator"] = np.tile([-0.5, 0.5, -0.5, 0.5, -0.5, 0.5], 8)
    prepared = _prepare(data)
    serial = ep.add_multilevel_mediation_component(
        prepared,
        value_col="trust",
        semantic="mediator2",
        within_col="M2_within",
        between_col="M2_between",
    )
    with pytest.raises(GP3BayesError, match="not yet implemented"):
        specify_multilevel_serial_gaze_mediation(serial, random_slopes=("mediator_x",))
    moderated = ep.add_multilevel_mediation_component(
        prepared,
        value_col="moderator",
        semantic="moderator",
        within_col="Z_within",
        between_col="Z_between",
    )
    with pytest.raises(GP3BayesError, match="not yet implemented"):
        specify_multilevel_moderated_gaze_mediation(moderated, random_slopes=("outcome_m",))


def test_sampling_controls_fail_before_optional_backend_load():
    from gp3bayespy.multilevel_mediation import fit_multilevel_gaze_mediation

    prepared = prepared_binary()
    with pytest.raises(GP3BayesError, match="draws"):
        fit_multilevel_gaze_mediation(prepared, random_slopes=(), draws=10)
    with pytest.raises(GP3BayesError, match="cores"):
        fit_multilevel_gaze_mediation(prepared, random_slopes=(), chains=2, cores=3)
    with pytest.raises(GP3BayesError, match="target_accept"):
        fit_multilevel_gaze_mediation(prepared, random_slopes=(), target_accept=1.0)


def test_continuous_outcome_and_count_mediator_contracts():
    data = simulate_multilevel_gaze_mediation(n_participants=10, trials_per_participant=6, seed=51)
    data["source_dwell"] = np.round(data["source_dwell"] * 2).astype(int).clip(lower=0)
    data["correct_override"] = 0.4 * data["ai_correct"] + 0.2 * data["source_dwell"]
    prepared = PreparedFixture(data)
    spec = specify_multilevel_gaze_mediation(
        prepared, mediator_family="poisson", outcome_family="gaussian", random_slopes=()
    )
    assert spec.mediator_family == "poisson"
    assert spec.outcome_family == "gaussian"


def test_ordinal_outcome_requires_contiguous_categories():
    prepared = prepared_binary()
    prepared.data["correct_override"] = np.tile([1, 3, 1, 3, 1, 3], len(prepared.data) // 6)
    with pytest.raises(GP3BayesError, match="contiguous integer categories"):
        specify_multilevel_gaze_mediation(prepared, outcome_family="ordinal", random_slopes=())


def test_model_comparison_requires_identical_ordered_observations():
    prepared_a = prepared_binary(seed=31)
    prepared_b = prepared_binary(seed=31)
    prepared_b.data.loc[prepared_b.data.index[0], "correct_override"] = 1 - int(
        prepared_b.data.loc[prepared_b.data.index[0], "correct_override"]
    )
    spec_a = specify_multilevel_gaze_mediation(prepared_a, random_slopes=())
    spec_b = specify_multilevel_gaze_mediation(prepared_b, random_slopes=())
    fit_a = MultilevelMediationFit(
        specification=spec_a,
        posterior={},
        sampler_diagnostics={},
        backend_fit=object(),
        backend="fixture",
    )
    fit_b = MultilevelMediationFit(
        specification=spec_b,
        posterior={},
        sampler_diagnostics={},
        backend_fit=object(),
        backend="fixture",
    )
    with pytest.raises(GP3BayesError, match="same mediator/outcome observations"):
        compare_multilevel_mediation_models({"a": fit_a, "b": fit_b})


def test_strong_synthetic_path_signal_exceeds_weak_signal():
    strong = simulate_multilevel_gaze_mediation(
        n_participants=120, trials_per_participant=12, a_within=1.0, b_within=1.0, seed=61
    )
    weak = simulate_multilevel_gaze_mediation(
        n_participants=120, trials_per_participant=12, a_within=0.05, b_within=0.05, seed=61
    )

    def within_cov(frame):
        xw = frame["ai_correct"] - frame.groupby("participant_id")["ai_correct"].transform("mean")
        mw = frame["source_dwell"] - frame.groupby("participant_id")["source_dwell"].transform(
            "mean"
        )
        return abs(float(np.cov(xw, mw)[0, 1]))

    assert within_cov(strong) > within_cov(weak) * 3
