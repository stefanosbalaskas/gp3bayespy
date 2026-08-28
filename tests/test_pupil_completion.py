import matplotlib
import numpy as np

matplotlib.use("Agg")

from gp3bayespy import pupil as gp


def _advanced_fixture():
    sim = gp.simulate_advanced_pupil_timecourse(
        n_participants=6,
        trials_per_participant=4,
        time_points=12,
        missing_fraction=0.0,
        outlier_fraction=0.0,
        seed=11,
    )
    dt = np.diff(np.sort(sim.data["time_ms"].unique())).mean() / 1000
    contract = gp.create_pupil_contract(
        "pupil",
        "participant_id",
        "trial_id",
        "time_ms",
        "millimetres",
        1 / dt,
        time_unit="milliseconds",
        condition_col="condition",
        luminance_col="luminance",
    )
    prepared = gp.prepare_pupil_timecourse(sim.data, contract)
    specification = gp.specify_advanced_pupil_timecourse_model(
        prepared,
        temporal_structure="linear",
        covariates=("luminance",),
        allow_high_complexity=True,
    )
    fit = gp.fit_advanced_pupil_model(specification, chains=1, iter=160, warmup=80, seed=3)
    return sim, prepared, specification, fit


def test_advanced_pupil_execution_and_validation_paths():
    _, prepared, specification, fit = _advanced_fixture()
    prior = gp.create_advanced_pupil_prior_specification(specification)
    prior_check = gp.check_advanced_pupil_prior_predictive(
        specification, chains=1, iter=80, warmup=40, seed=3
    )
    trajectory = gp.predict_advanced_pupil_trajectory(fit, ndraws=40)
    temporal = gp.audit_pupil_temporal_dependence(prepared, max_lag=4)
    diagnostics = gp.diagnose_advanced_pupil_fit(fit)
    posterior = gp.summarise_pupil_posterior(fit)
    identifiability = gp.audit_advanced_pupil_identifiability(specification)
    calibration = gp.audit_pupil_predictive_calibration(fit, specification.data.head(20), ndraws=20)

    assert not prior.table.empty
    assert prior_check.executed is True
    assert prior_check.adequacy_certified is False
    assert trajectory.draws.shape[0] == 40
    assert not gp.pupil_autocorrelation_table(temporal).empty
    assert not diagnostics.parameter_summary.empty
    assert not posterior.table.empty
    assert identifiability.certification is False
    assert calibration.adequacy_certified is False


def test_binocular_pupil_execution_paths():
    sim = gp.simulate_binocular_pupil_timecourse(
        n_participants=5,
        trials_per_participant=4,
        time_points=10,
        missing_fraction=0.0,
        seed=12,
    )
    prepared = gp.prepare_binocular_pupil_timecourse(sim.data)
    audit = gp.audit_binocular_pupil_readiness(prepared)
    specification = gp.specify_binocular_pupil_model(
        prepared, temporal_structure="linear", smooth_basis_dimension=6
    )
    fit = gp.fit_binocular_pupil_model(specification, chains=1, iter=160, warmup=80, seed=2)
    trajectory = gp.estimate_binocular_pupil_trajectory(fit, ndraws=30)

    assert audit.status in {"pass", "review", "failure"}
    assert trajectory.left_draws.shape == trajectory.right_draws.shape
    assert not gp.pupil_binocular_difference(trajectory).empty
    assert not gp.pupil_binocular_correlation(fit).empty
    agreement = gp.pupil_binocular_agreement_table(trajectory)
    assert agreement["probability_within_tolerance"].between(0, 1).all()


def test_response_shape_execution_and_graphics():
    simulation = gp.simulate_pupil_response_shape(
        n_participants=5,
        trials_per_participant=3,
        time_points=15,
        seed=13,
    )
    specification = gp.specify_pupil_response_shape_model(simulation.data)
    fit = gp.fit_pupil_response_shape_model(specification, chains=1, iter=200, warmup=100, seed=2)
    estimand = gp.estimate_pupil_response_parameters(fit)
    table = gp.pupil_response_parameter_table(estimand)
    fig = gp.plot_pupil_response_parameters(estimand)

    assert specification.experimental is True
    assert fit.fit_performed is True
    assert not table.empty
    assert fig.__class__.__name__ == "Figure"


def test_pupil_model_comparison_never_selects_automatically():
    _, prepared, _, fit = _advanced_fixture()
    alternate = gp.fit_advanced_pupil_model(
        gp.specify_advanced_pupil_timecourse_model(
            prepared,
            temporal_structure="smooth",
            smooth_basis_dimension=5,
            allow_high_complexity=True,
        ),
        chains=1,
        iter=160,
        warmup=80,
        seed=4,
    )
    model_set = gp.create_pupil_model_set({"linear": fit, "smooth": alternate})
    comparison = gp.compare_pupil_models(model_set)
    weights = gp.pupil_model_weights(comparison)
    plan = gp.create_pupil_lfo_plan(fit, initial_fraction=0.6, horizon=2, step=2, max_refits=2)
    validation = gp.validate_pupil_leave_future_out(fit, plan, execute=False)

    assert model_set.automatic_winner is False
    assert not gp.pupil_model_comparison_table(comparison).empty
    assert np.isclose(weights["weight"].sum(), 1.0)
    assert not weights["automatic_selection"].any()
    assert validation.executed is False
