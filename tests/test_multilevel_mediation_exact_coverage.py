from __future__ import annotations

from types import SimpleNamespace
import copy
import importlib as pyimportlib

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.multilevel_mediation as mm


class Prepared:
    def __init__(self, data: pd.DataFrame, provenance=None):
        frame = data.copy(deep=True)
        xmean = frame.groupby("participant_id")["ai_correct"].transform("mean")
        mmean = frame.groupby("participant_id")["source_dwell"].transform("mean")
        frame["X_within"] = frame["ai_correct"] - xmean
        frame["X_between"] = xmean
        frame["M_within"] = frame["source_dwell"] - mmean
        frame["M_between"] = mmean
        frame["mediation_analysis_eligible"] = frame[
            ["ai_correct", "source_dwell", "correct_override"]
        ].notna().all(axis=1)
        self.data = frame
        self.columns = {
            "participant": "participant_id",
            "trial": "trial_id",
            "x": "ai_correct",
            "mediator": "source_dwell",
            "outcome": "correct_override",
        }
        self.provenance = {"fixture": True} if provenance is None else provenance


def prepared(n=8, trials=6, seed=81):
    return Prepared(mm.simulate_multilevel_gaze_mediation(
        n_participants=n, trials_per_participant=trials, seed=seed
    ))


def augment_serial(p: Prepared, *, constant=False):
    p = copy.deepcopy(p)
    d = p.data
    d["trust"] = 2.0 if constant else 2 + 0.4 * d["ai_correct"] + 0.2 * d["source_dwell"] + 0.01 * np.arange(len(d))
    mean = d.groupby("participant_id")["trust"].transform("mean")
    d["M2_within"] = d["trust"] - mean
    d["M2_between"] = mean
    p.columns.update({"mediator2": "trust", "mediator2_within": "M2_within", "mediator2_between": "M2_between"})
    return p


def augment_moderator(p: Prepared, *, component="within", constant=False):
    p = copy.deepcopy(p)
    d = p.data
    if constant:
        d["moderator"] = 1.0
    else:
        d["moderator"] = np.resize(np.array([-1.0, 0.0, 1.0]), len(d))
    mean = d.groupby("participant_id")["moderator"].transform("mean")
    d["Z_within"] = d["moderator"] - mean
    d["Z_between"] = mean
    # ensure a varying between component when explicitly requested
    if component == "between" and not constant:
        participant_code = pd.factorize(d["participant_id"])[0].astype(float)
        d["Z_between"] = participant_code
    p.columns.update({"moderator": "moderator", "moderator_within": "Z_within", "moderator_between": "Z_between"})
    return p


class FakeModel:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class FakePM:
    def __init__(self, n_rows=48):
        self.n_rows = n_rows
        self.model = FakeModel()
        self.record = {}
        self.math = SimpleNamespace(exp=np.exp, sigmoid=lambda x: 1 / (1 + np.exp(-np.asarray(x))))
        self.distributions = SimpleNamespace(transforms=SimpleNamespace(ordered=object()))

    def Model(self):
        return self.model

    def Normal(self, name, mu=0.0, sigma=1.0, shape=None, **kwargs):
        value = np.full(shape, 0.1) if shape is not None else 0.1
        self.record[name] = value
        return value

    def HalfNormal(self, name, sigma=1.0, **kwargs):
        self.record[name] = 0.2
        return 0.2

    def Exponential(self, name, lam=1.0, **kwargs):
        self.record[name] = 1.5
        return 1.5

    def Deterministic(self, name, value):
        arr = np.asarray(value)
        self.record[name] = arr
        return arr

    def _obs(self, name, **kwargs):
        self.record[name] = kwargs
        return kwargs

    def LogNormal(self, name, **kwargs): return self._obs(name, **kwargs)
    def Gamma(self, name, **kwargs): return self._obs(name, **kwargs)
    def Beta(self, name, **kwargs): return self._obs(name, **kwargs)
    def Bernoulli(self, name, **kwargs): return self._obs(name, **kwargs)
    def Poisson(self, name, **kwargs): return self._obs(name, **kwargs)
    def NegativeBinomial(self, name, **kwargs): return self._obs(name, **kwargs)
    def OrderedLogistic(self, name, **kwargs): return self._obs(name, **kwargs)

    def sample(self, **kwargs):
        names = [
            "a_within", "a_between", "b_within", "b_between", "cprime_within", "cprime_between",
            "indirect_within", "indirect_between", "total_within", "total_between",
            "participant_a_slope", "participant_b_slope", "participant_indirect_within",
            "a1_within", "a1_between", "a2_within", "a2_between", "d_within", "d_between",
            "b1_within", "b1_between", "b2_within", "b2_between", "serial_indirect_within",
            "serial_indirect_between", "m1_indirect_within", "m2_indirect_within",
            "total_indirect_within", "moderation",
        ]
        posterior = {name: np.ones((2, 3)) * (i + 1) / 10 for i, name in enumerate(names)}
        return SimpleNamespace(
            posterior=posterior,
            sample_stats={"diverging": np.zeros((2, 3), dtype=bool), "tree_depth": np.ones((2, 3), dtype=int)},
        )

    def sample_posterior_predictive(self, *args, **kwargs):
        n = self.n_rows
        return SimpleNamespace(posterior_predictive={
            "M_obs": np.arange(5 * n, dtype=float).reshape(1, 5, n) / max(n, 1),
            "Y_obs": np.resize(np.array([0.0, 1.0]), 5 * n).reshape(1, 5, n),
        })

    def sample_prior_predictive(self, draws, random_seed):
        n = self.n_rows
        return SimpleNamespace(prior_predictive={
            "M_obs": np.ones((1, draws, n)),
            "Y_obs": np.zeros((1, draws, n)),
        })


class FakeDataVars:
    def __init__(self, values):
        self.data_vars = {f"v{i}": np.asarray(v) for i, v in enumerate(values)}


class FakeArviz:
    def __init__(self, *, loo_fail=False):
        self.loo_fail = loo_fail
    def rhat(self, idata):
        return FakeDataVars([np.array([1.001, 1.005])])
    def ess(self, idata, method="bulk"):
        return FakeDataVars([np.array([700.0, 600.0])])
    def loo(self, idata, var_name=None):
        if self.loo_fail:
            raise RuntimeError("loo failed")
        offset = float(np.asarray(idata.log_likelihood["joint"]).mean())
        return SimpleNamespace(elpd_loo=10 + offset, se=0.5, p_loo=1.2, warning=False)


class FakeBackendFit:
    def __init__(self, offset=0.0):
        self.log_likelihood = {
            "M_obs": np.ones((2, 3, 4)) * offset,
            "Y_obs": np.ones((2, 3, 4)) * (offset + 0.1),
        }
    def copy(self):
        out = FakeBackendFit()
        out.log_likelihood = {k: np.array(v, copy=True) for k, v in self.log_likelihood.items()}
        return out


def passing_fit(p=None, posterior=None, backend_fit=None, backend_model=None, kind="simple"):
    p = p or prepared()
    spec = mm.specify_multilevel_gaze_mediation(p, random_slopes=())
    if kind != "simple":
        spec = mm.MultilevelMediationSpecification(
            **{**{name: getattr(spec, name) for name in spec.__dataclass_fields__}, "model_kind": kind}
        )
    x = np.linspace(-0.1, 0.1, 20)
    default = {
        "a_within": x + 0.6, "b_within": x + 0.7,
        "a_between": x + 0.2, "b_between": x + 0.3,
        "cprime_within": x + 0.1, "cprime_between": x + 0.05,
        "indirect_within": (x + 0.6) * (x + 0.7),
        "indirect_between": (x + 0.2) * (x + 0.3),
        "total_within": x + 0.1 + (x + 0.6) * (x + 0.7),
        "total_between": x + 0.05 + (x + 0.2) * (x + 0.3),
    }
    if posterior is not None:
        default = posterior
    return mm.MultilevelMediationFit(
        specification=spec, posterior=default,
        sampler_diagnostics={"max_rhat": 1.001, "min_ess_bulk": 800, "divergences": 0, "treedepth_hits": 0},
        backend_fit=backend_fit, backend_model=backend_model, backend="fixture",
    )


def test_low_level_validation_contracts(monkeypatch):
    for bad in [None, "", "   "]:
        with pytest.raises(mm.GP3BayesError, match="non-empty family"):
            mm._normalize_family(bad, {"gaussian"}, "family")
    assert mm._normalize_family("negative-binomial", mm._MEDIATOR_FAMILIES, "family") == "negative_binomial"
    assert mm._normalize_family("normal", mm._MEDIATOR_FAMILIES, "family") == "gaussian"
    with pytest.raises(mm.GP3BayesError, match="Unsupported"):
        mm._normalize_family("weird", {"gaussian"}, "family")
    for bad in [True, 0, -1, np.inf, "x"]:
        with pytest.raises(mm.GP3BayesError, match="finite positive"):
            mm._finite_positive(bad, "x")
    priors = mm.create_mediation_prior_specification()
    assert priors.as_dict()["intercept_sd"] == 2.5

    with pytest.raises(mm.GP3BayesError, match="preparation object"):
        mm._prepared_contract(object())
    x = SimpleNamespace(data=pd.DataFrame(), columns=[], provenance={})
    with pytest.raises(mm.GP3BayesError, match="canonical mediation column contract"):
        mm._prepared_contract(x)
    x.columns = {"participant": "p", "trial": "t", "mediator": "m", "outcome": "y"}
    x.provenance = "not-map"
    x.data = pd.DataFrame({"X_within": [], "X_between": [], "M_within": [], "M_between": [], "mediation_analysis_eligible": []})
    _, _, provenance = mm._prepared_contract(x)
    assert provenance == {}
    x.columns = {"participant": "p"}
    with pytest.raises(mm.GP3BayesError, match="missing canonical mediation semantics"):
        mm._prepared_contract(x)
    x.columns = {"participant": "p", "trial": "t", "mediator": "m", "outcome": "y"}
    x.data = pd.DataFrame()
    with pytest.raises(mm.GP3BayesError, match="missing canonical columns"):
        mm._prepared_contract(x)

    monkeypatch.setattr(mm, "version", lambda name: (_ for _ in ()).throw(mm.PackageNotFoundError()))
    assert mm._package_version("missing") == "not_installed"
    fake = object()
    monkeypatch.setattr(mm, "import_module", lambda name: fake)
    assert mm._load_pymc() is fake


def test_family_data_all_failures_and_valid_ordinal():
    base = pd.DataFrame({"m": [1.0, 2.0, 3.0], "y": [0, 1, 2]})
    with pytest.raises(mm.GP3BayesError, match="bernoulli.*mediator"):
        mm._validate_family_data(base, "m", "y", "bernoulli", "ordinal")
    with pytest.raises(mm.GP3BayesError, match="bernoulli.*outcome"):
        mm._validate_family_data(pd.DataFrame({"m": [1., 2.], "y": [0., 2.]}), "m", "y", "gaussian", "bernoulli")
    with pytest.raises(mm.GP3BayesError, match="Outcome family.*integer counts"):
        mm._validate_family_data(pd.DataFrame({"m": [1., 2.], "y": [0.2, 1.]}), "m", "y", "gaussian", "poisson")
    mm._validate_family_data(base, "m", "y", "gaussian", "ordinal")
    with pytest.raises(mm.GP3BayesError, match="contiguous integer"):
        mm._validate_family_data(pd.DataFrame({"m": [1., 2.], "y": [1., 2.]}), "m", "y", "gaussian", "ordinal")


def test_estimability_and_sampling_helpers_cover_edges():
    assert not mm._has_variation([1.0])
    assert not mm._has_variation([1.0, 1.0])
    assert mm._has_variation([1.0, 2.0])
    df = pd.DataFrame({
        "X_within": [0, 1], "X_between": [0, 1], "M_within": [0, 1], "M_between": [0, 1]
    })
    assert set(mm._estimable_simple_paths(df)) == {"a_within", "cprime_within", "a_between", "cprime_between", "b_within", "b_between"}
    zeros = df * 0
    assert mm._estimable_simple_paths(zeros) == ()

    controls = mm._validate_sampling_controls(chains=3, draws=50, tune=0, cores=None, seed=0, target_accept=.9, max_treedepth=5)
    assert controls["cores"] == 2
    controls2 = mm._validate_sampling_controls(chains=2, draws=50, tune=1, cores=1, seed=2, target_accept=.8, max_treedepth=6)
    assert controls2["cores"] == 1
    cases = [
        ({"chains": 1}, "chains"), ({"tune": -1}, "tune"), ({"max_treedepth": 4}, "max_treedepth"),
        ({"cores": 0}, "cores"), ({"seed": -1}, "seed"), ({"target_accept": True}, "target_accept"),
    ]
    defaults = dict(chains=2, draws=50, tune=0, cores=1, seed=0, target_accept=.9, max_treedepth=5)
    for update, msg in cases:
        args = {**defaults, **update}
        with pytest.raises(mm.GP3BayesError, match=msg):
            mm._validate_sampling_controls(**args)
    for d, s, msg in [(0, 1, "draws"), (True, 1, "draws"), (1, -1, "seed"), (1, True, "seed")]:
        with pytest.raises(mm.GP3BayesError, match=msg):
            mm._validate_draw_seed(d, s)
    assert mm._validate_draw_seed(2, 3) == (2, 3)


def test_specification_validation_remaining_branches(monkeypatch):
    p = prepared()
    with pytest.raises(mm.GP3BayesError, match="missingness_policy"):
        mm.specify_multilevel_gaze_mediation(p, missingness_policy="magic", random_slopes=())
    with pytest.raises(mm.GP3BayesError, match="duplicates"):
        mm.specify_multilevel_gaze_mediation(p, random_slopes=("mediator_x", "mediator_x"))
    with pytest.raises(mm.GP3BayesError, match="MediationPriorSpecification"):
        mm.specify_multilevel_gaze_mediation(p, priors={"bad": 1}, random_slopes=())

    q = copy.deepcopy(p)
    q.data = q.data.drop(columns=["correct_override"])
    with pytest.raises(mm.GP3BayesError, match="missing model columns"):
        mm.specify_multilevel_gaze_mediation(q, random_slopes=())

    q = copy.deepcopy(p)
    q.data["mediation_analysis_eligible"] = False
    with pytest.raises(mm.GP3BayesError, match="No rows remain"):
        mm.specify_multilevel_gaze_mediation(q, missingness_policy="quality_eligible", random_slopes=())

    q = copy.deepcopy(p)
    q.data = q.data[q.data["participant_id"].eq(q.data["participant_id"].iloc[0])].copy()
    with pytest.raises(mm.GP3BayesError, match="At least two participants"):
        mm.specify_multilevel_gaze_mediation(q, random_slopes=())

    q = copy.deepcopy(p)
    q.data["M_within"] = 0.0
    with pytest.raises(mm.GP3BayesError, match="not estimable"):
        mm.specify_multilevel_gaze_mediation(q, random_slopes=())

    # Artificially isolate the guard for requested but unsupported slope paths.
    monkeypatch.setattr(mm, "_estimable_simple_paths", lambda data: ("a_within", "b_within"))
    with pytest.raises(mm.GP3BayesError, match="Requested random slope"):
        mm.specify_multilevel_gaze_mediation(p, random_slopes=("outcome_x",))


def test_random_slope_repeated_support_guard(monkeypatch):
    p = prepared(n=2, trials=3)
    # Retain one repeated participant and one singleton, but preserve data variation.
    d = pd.concat([p.data[p.data.participant_id.eq("p001")], p.data[p.data.participant_id.eq("p002")].head(1)], ignore_index=True)
    p.data = d
    monkeypatch.setattr(mm, "_estimable_simple_paths", lambda data: ("a_within", "b_within", "cprime_within"))
    with pytest.raises(mm.GP3BayesError, match="Random slopes require repeated trials"):
        mm.specify_multilevel_gaze_mediation(p, random_slopes=("mediator_x",))


def test_observation_families_and_model_builders(monkeypatch):
    fake = FakePM()
    priors = mm.create_mediation_prior_specification()
    eta = np.array([0.1, 0.2, 0.3])
    families = {
        "gaussian": np.array([1., 2., 3.]), "lognormal": np.array([1., 2., 3.]),
        "gamma": np.array([1., 2., 3.]), "beta": np.array([.2, .4, .6]),
        "bernoulli": np.array([0., 1., 0.]), "poisson": np.array([0., 1., 2.]),
        "negative_binomial": np.array([0., 1., 2.]), "ordinal": np.array([0., 1., 2.]),
    }
    for fam, obs in families.items():
        mm._observe_family(fake, f"obs_{fam}", fam, eta, obs, priors)
    with pytest.raises(mm.GP3BayesError, match="Unsupported family"):
        mm._observe_family(fake, "bad", "bad", eta, eta, priors)
    assert len(mm._normal_paths(fake, 1.0)) == 6
    idx = np.array([0, 1, 0])
    assert np.asarray(mm._random_effect(fake, "r", 2, idx, 1.0)).shape == (3,)
    assert np.asarray(mm._random_slope(fake, "s", eta, 2, idx, 1.0)).shape == (3,)

    p = prepared()
    spec = mm.specify_multilevel_gaze_mediation(
        p, random_slopes=("mediator_x", "outcome_x", "outcome_m")
    )
    fake.n_rows = spec.analysis_rows
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    assert mm._build_pymc_model(spec) is fake.model
    assert "participant_indirect_within" in fake.record

    # Balanced-X setup omits between-X deterministic branches.
    q = prepared(n=4, trials=4)
    for _, gidx in q.data.groupby("participant_id").groups.items():
        q.data.loc[gidx, "ai_correct"] = [0, 1, 0, 1]
    xmean = q.data.groupby("participant_id")["ai_correct"].transform("mean")
    q.data["X_within"] = q.data["ai_correct"] - xmean
    q.data["X_between"] = xmean
    spec2 = mm.specify_multilevel_gaze_mediation(q, random_slopes=())
    mm._build_pymc_model(spec2)


def test_flatten_posterior_and_sampler_diagnostics(monkeypatch):
    assert mm._flatten_posterior(SimpleNamespace(), ["a"]) == {}
    idata = SimpleNamespace(posterior={"a": np.arange(12).reshape(2, 3, 2), "scalar": np.array([1.0])})
    out = mm._flatten_posterior(idata, ["a", "missing", "scalar"])
    assert out["a"].shape == (6, 2)
    assert out["scalar"].shape == (1,)

    fake_az = FakeArviz()
    monkeypatch.setattr(mm, "import_module", lambda name: fake_az if name == "arviz" else pyimportlib.import_module(name))
    idata2 = SimpleNamespace(sample_stats={"diverging": np.array([[False, True]]), "tree_depth": np.array([[4, 12]])})
    d = mm._sampler_diagnostics(idata2, max_treedepth=12)
    assert d["max_rhat"] == pytest.approx(1.005)
    assert d["min_ess_bulk"] == 600
    assert d["divergences"] == 1
    assert d["treedepth_hits"] == 1

    # ArviZ failure + fallback treedepth key + missing diverging field.
    def fail_arviz(name):
        if name == "arviz": raise RuntimeError("no arviz")
        return pyimportlib.import_module(name)
    monkeypatch.setattr(mm, "import_module", fail_arviz)
    d2 = mm._sampler_diagnostics(SimpleNamespace(sample_stats={"treedepth": np.array([[5, 6]])}), max_treedepth=6)
    assert np.isnan(d2["max_rhat"])
    assert d2["treedepth_hits"] == 1
    d3 = mm._sampler_diagnostics(SimpleNamespace(), max_treedepth=6)
    assert d3["divergences"] == 0


def test_convergence_effect_and_summary_failure_edges():
    with pytest.raises(mm.GP3BayesError, match="MultilevelMediationFit"):
        mm.check_mediation_convergence(object())
    fit = passing_fit()
    nanfit = mm.MultilevelMediationFit(
        specification=fit.specification, posterior=fit.posterior,
        sampler_diagnostics={"max_rhat": np.nan, "min_ess_bulk": np.nan, "divergences": 0, "treedepth_hits": 0},
    )
    conv = mm.check_mediation_convergence(nanfit)
    assert "R-hat was not available" in conv.issues and "bulk ESS was not available" in conv.issues
    with pytest.raises(mm.GP3BayesError, match="missing or non-finite"):
        mm._effect_from_draws("x", np.array([]), .95, "x")
    with pytest.raises(mm.GP3BayesError, match="missing or non-finite"):
        mm._effect_from_draws("x", np.array([np.nan]), .95, "x")
    with pytest.raises(mm.GP3BayesError, match="probability"):
        mm._effect_from_draws("x", np.array([1., 2.]), 1.0, "x")
    mm._require_acceptable_fit(fit, False)
    with pytest.raises(mm.GP3BayesError, match="level"):
        mm.posterior_indirect_effect(fit, level="bad")
    fallback = passing_fit(posterior={"a_within": np.ones(5), "b_within": np.ones(5) * 2, "cprime_within": np.ones(5)})
    assert mm.posterior_indirect_effect(fallback).mean == 2
    with pytest.raises(mm.GP3BayesError, match="unavailable"):
        mm.posterior_indirect_effect(fallback, level="between")
    assert mm.estimate_indirect_effect(fallback).mean == 2
    with pytest.raises(mm.GP3BayesError, match="level"):
        mm.posterior_direct_effect(fit, level="bad")
    with pytest.raises(mm.GP3BayesError, match="unavailable"):
        mm.posterior_direct_effect(fallback, level="between")
    with pytest.raises(mm.GP3BayesError, match="level"):
        mm.posterior_total_effect(fit, level="bad")
    assert mm.posterior_total_effect(fallback).mean == 3
    no_c = passing_fit(posterior={"a_within": np.ones(5), "b_within": np.ones(5)})
    with pytest.raises(mm.GP3BayesError, match="unavailable"):
        mm.posterior_total_effect(no_c)
    summary = mm.summarise_multilevel_mediation(no_c)
    assert set(summary.effect) == {"a_within", "b_within", "indirect_within"}


def test_predictive_checks_with_fake_backend(monkeypatch):
    p = prepared()
    spec = mm.specify_multilevel_gaze_mediation(p, random_slopes=())
    fake_pm = FakePM(spec.analysis_rows)
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake_pm)
    monkeypatch.setattr(mm, "_build_pymc_model", lambda s: fake_pm.model)
    fit = passing_fit(p, backend_fit=object(), backend_model=fake_pm.model)
    ppc = mm.posterior_predictive_check_mediation(fit, draws=2, seed=2)
    assert list(ppc.variable) == ["M_obs", "Y_obs"]
    assert np.isfinite(ppc.predictive_mean).all()
    prior = mm.prior_predictive_check_mediation(spec, draws=3, seed=2)
    assert (prior.draws == 3).all()

    empty_fit = passing_fit()
    with pytest.raises(mm.GP3BayesError, match="fitted backend model"):
        mm.posterior_predictive_check_mediation(empty_fit)
    with pytest.raises(mm.GP3BayesError, match="specification"):
        mm.prior_predictive_check_mediation(object())

    class NoPPC(FakePM):
        def sample_posterior_predictive(self, *args, **kwargs): return SimpleNamespace()
        def sample_prior_predictive(self, draws, random_seed): return SimpleNamespace()
    bad = NoPPC(spec.analysis_rows)
    monkeypatch.setattr(mm, "_load_pymc", lambda: bad)
    with pytest.raises(mm.GP3BayesError, match="posterior predictive"):
        mm.posterior_predictive_check_mediation(fit)
    monkeypatch.setattr(mm, "_build_pymc_model", lambda s: bad.model)
    with pytest.raises(mm.GP3BayesError, match="prior predictive"):
        mm.prior_predictive_check_mediation(spec)


def test_model_comparison_success_and_failures(monkeypatch):
    fit1 = passing_fit(backend_fit=FakeBackendFit(0.0), backend_model=FakeModel())
    fit2 = passing_fit(backend_fit=FakeBackendFit(1.0), backend_model=FakeModel())
    with pytest.raises(mm.GP3BayesError, match="at least two"):
        mm.compare_multilevel_mediation_models({"one": fit1})

    base = fit1.specification
    fields = {name: getattr(base, name) for name in base.__dataclass_fields__}
    assert mm._same_comparison_observations(base, fit2.specification)
    renamed = mm.MultilevelMediationSpecification(**{**fields, "mediator_col": "different_mediator"})
    assert not mm._same_comparison_observations(base, renamed)
    count_changed = mm.MultilevelMediationSpecification(**{**fields, "analysis_rows": base.analysis_rows + 1})
    assert not mm._same_comparison_observations(base, count_changed)
    key_data = base.data.copy()
    key_data.loc[key_data.index[0], base.trial_col] = 999
    key_changed = mm.MultilevelMediationSpecification(**{**fields, "data": key_data})
    assert not mm._same_comparison_observations(base, key_changed)
    value_data = base.data.copy()
    value_data.loc[value_data.index[0], base.outcome_col] = 1 - int(value_data.loc[value_data.index[0], base.outcome_col])
    value_changed = mm.MultilevelMediationSpecification(**{**fields, "data": value_data})
    assert not mm._same_comparison_observations(base, value_changed)
    mismatch_fit = mm.MultilevelMediationFit(
        specification=value_changed, posterior=fit2.posterior, sampler_diagnostics=fit2.sampler_diagnostics,
        backend_fit=fit2.backend_fit, backend_model=fit2.backend_model, backend="fixture",
    )
    with pytest.raises(mm.GP3BayesError, match="same mediator/outcome observations"):
        mm.compare_multilevel_mediation_models({"a": fit1, "mismatch": mismatch_fit})

    real_import = pyimportlib.import_module
    monkeypatch.setattr(mm, "import_module", lambda name: FakeArviz() if name == "arviz" else real_import(name))
    out = mm.compare_multilevel_mediation_models({"a": fit1, "b": fit2})
    assert list(out.model) == ["b", "a"]
    with pytest.raises(mm.GP3BayesError, match="not a fitted"):
        mm.compare_multilevel_mediation_models({"a": fit1, "bad": object()})
    badfit = passing_fit(backend_fit=SimpleNamespace(log_likelihood={}), backend_model=FakeModel())
    with pytest.raises(mm.GP3BayesError, match="lacks mediator/outcome"):
        mm.compare_multilevel_mediation_models({"a": fit1, "bad": badfit})
    monkeypatch.setattr(mm, "import_module", lambda name: FakeArviz(loo_fail=True) if name == "arviz" else real_import(name))
    with pytest.raises(mm.GP3BayesError, match="Could not compute"):
        mm.compare_multilevel_mediation_models({"a": fit1, "b": fit2})
    def no_arviz(name):
        if name == "arviz": raise ImportError("missing")
        return real_import(name)
    monkeypatch.setattr(mm, "import_module", no_arviz)
    with pytest.raises(mm.BackendUnavailableError, match="ArviZ"):
        mm.compare_multilevel_mediation_models({"a": fit1, "b": fit2})


def test_plots_reports_and_simulation_guards(monkeypatch):
    import matplotlib.pyplot as plt
    fit = passing_fit()
    ax = mm.plot_indirect_effect_distribution(fit)
    assert ax.get_title()
    fig, supplied = plt.subplots()
    assert mm.plot_indirect_effect_distribution(fit, ax=supplied) is supplied
    ax2 = mm.plot_mediation_posteriors(fit)
    assert ax2.get_title()
    fig2, supplied2 = plt.subplots()
    assert mm.plot_mediation_posteriors(fit, ax=supplied2) is supplied2

    participant = passing_fit(posterior={**fit.posterior, "participant_indirect_within": np.ones((10, 3))})
    ax3 = mm.plot_participant_mediation_effects(participant)
    assert ax3.get_xlabel() == "Participant"
    with pytest.raises(mm.GP3BayesError, match="Participant-specific"):
        mm.plot_participant_mediation_effects(fit)

    failed = mm.MultilevelMediationFit(
        specification=fit.specification, posterior=fit.posterior,
        sampler_diagnostics={"max_rhat": 1.2, "min_ess_bulk": 10, "divergences": 1, "treedepth_hits": 1},
    )
    with pytest.raises(mm.GP3BayesError, match="substantive report"):
        mm.report_multilevel_gaze_mediation(failed)
    assert "Sampler diagnostics status: fail" in mm.report_multilevel_gaze_mediation(failed, require_convergence=False)
    with pytest.raises(mm.GP3BayesError, match="at least two participants"):
        mm.simulate_multilevel_gaze_mediation(n_participants=1)
    with pytest.raises(mm.GP3BayesError, match="two trials"):
        mm.simulate_multilevel_gaze_mediation(trials_per_participant=1)

    real_import = pyimportlib.import_module
    def no_mpl(name):
        if name == "matplotlib.pyplot": raise ImportError("no mpl")
        return real_import(name)
    monkeypatch.setattr(mm, "import_module", no_mpl)
    with pytest.raises(mm.BackendUnavailableError, match="Matplotlib"):
        mm.plot_indirect_effect_distribution(fit)
    with pytest.raises(mm.BackendUnavailableError, match="Matplotlib"):
        mm.plot_mediation_posteriors(fit)
    with pytest.raises(mm.BackendUnavailableError, match="Matplotlib"):
        mm.plot_participant_mediation_effects(participant)


def test_select_rows_and_serial_spec_edge_cases(monkeypatch):
    p = augment_serial(prepared())
    d = p.data
    required = ["participant_id", "trust"]
    with pytest.raises(mm.GP3BayesError, match="missingness_policy"):
        mm._select_analysis_rows(d, required=required, missingness_policy="bad")
    d2 = d.copy(); d2.loc[d2.index[0], "trust"] = np.nan
    with pytest.raises(mm.GP3BayesError, match="explicit `missingness_policy`"):
        mm._select_analysis_rows(d2, required=required, missingness_policy="error")
    a, ex = mm._select_analysis_rows(d2, required=required, missingness_policy="complete_case")
    assert len(ex) == 1 and len(a) == len(d2)-1
    d3 = d.copy(); d3["mediation_analysis_eligible"] = False
    with pytest.raises(mm.GP3BayesError, match="No rows remain"):
        mm._select_analysis_rows(d3, required=required, missingness_policy="quality_eligible")

    with pytest.raises(mm.GP3BayesError, match="MediationPriorSpecification"):
        mm.specify_multilevel_serial_gaze_mediation(p, priors={"bad": 1})
    q = copy.deepcopy(p); q.data = q.data.drop(columns=["M2_within"])
    with pytest.raises(mm.GP3BayesError, match="missing model columns"):
        mm.specify_multilevel_serial_gaze_mediation(q)
    q = copy.deepcopy(p); q.data = q.data[q.data.participant_id.eq(q.data.participant_id.iloc[0])].copy()
    with pytest.raises(mm.GP3BayesError, match="At least two participants"):
        mm.specify_multilevel_serial_gaze_mediation(q)
    flat = augment_serial(prepared(), constant=True)
    with pytest.raises(mm.GP3BayesError, match="serial indirect effect is not estimable"):
        mm.specify_multilevel_serial_gaze_mediation(flat)


def test_serial_build_fit_and_effects(monkeypatch):
    p = augment_serial(prepared())
    spec = mm.specify_multilevel_serial_gaze_mediation(p)
    fake = FakePM(spec.analysis_rows)
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    assert mm._build_serial_pymc_model(spec) is fake.model
    monkeypatch.setattr(mm, "_sampler_diagnostics", lambda idata, max_treedepth: {"max_rhat":1.0,"min_ess_bulk":1000,"divergences":0,"treedepth_hits":0})
    fit = mm.fit_multilevel_serial_gaze_mediation(p, chains=2, draws=50, tune=0, cores=1)
    assert fit.specification.model_kind == "serial"
    fit2 = mm.MultilevelMediationFit(
        specification=fit.specification,
        posterior={"serial_indirect_within": np.ones(10), "serial_indirect_between": np.ones(10)*2},
        sampler_diagnostics={"max_rhat":1.0,"min_ess_bulk":1000,"divergences":0,"treedepth_hits":0},
    )
    assert mm.posterior_serial_indirect_effect(fit2).mean == 1
    assert mm.posterior_serial_indirect_effect(fit2, level="between").mean == 2
    with pytest.raises(mm.GP3BayesError, match="level"):
        mm.posterior_serial_indirect_effect(fit2, level="bad")
    missing = mm.MultilevelMediationFit(specification=fit.specification, posterior={}, sampler_diagnostics=fit2.sampler_diagnostics)
    with pytest.raises(mm.GP3BayesError, match="unavailable"):
        mm.posterior_serial_indirect_effect(missing)
    with pytest.raises(mm.GP3BayesError, match="not a serial"):
        mm.posterior_serial_indirect_effect(passing_fit())

    monkeypatch.setattr(mm, "_sampler_diagnostics", lambda idata, max_treedepth: {"max_rhat":1.2,"min_ess_bulk":10,"divergences":1,"treedepth_hits":0})
    with pytest.warns(RuntimeWarning, match="Serial mediation fit failed"):
        mm.fit_multilevel_serial_gaze_mediation(p, chains=2, draws=50, tune=0, cores=1)


def test_moderated_spec_build_fit_and_effect_edges(monkeypatch):
    p = augment_moderator(prepared())
    with pytest.raises(mm.GP3BayesError, match="moderation_path"):
        mm.specify_multilevel_moderated_gaze_mediation(p, moderation_path="x")
    with pytest.raises(mm.GP3BayesError, match="moderator_component"):
        mm.specify_multilevel_moderated_gaze_mediation(p, moderator_component="x")
    q = copy.deepcopy(p); del q.columns["moderator_within"]
    with pytest.raises(mm.GP3BayesError, match="moderator_within"):
        mm.specify_multilevel_moderated_gaze_mediation(q)
    q = copy.deepcopy(p); q.data = q.data.drop(columns=["Z_within"])
    with pytest.raises(mm.GP3BayesError, match="missing"):
        mm.specify_multilevel_moderated_gaze_mediation(q)
    q = copy.deepcopy(p); q.data.loc[q.data.index[0], "Z_within"] = np.nan
    with pytest.raises(mm.GP3BayesError, match="Moderator contains missing"):
        mm.specify_multilevel_moderated_gaze_mediation(q)
    spec_cc = mm.specify_multilevel_moderated_gaze_mediation(q, missingness_policy="complete_case")
    assert spec_cc.analysis_rows == len(q.data)-1
    constant = augment_moderator(prepared(), constant=True)
    with pytest.raises(mm.GP3BayesError, match="moderator component has no"):
        mm.specify_multilevel_moderated_gaze_mediation(constant)

    spec_a = mm.specify_multilevel_moderated_gaze_mediation(p, moderation_path="a")
    fake = FakePM(spec_a.analysis_rows)
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    assert mm._build_moderated_pymc_model(spec_a) is fake.model
    spec_b = mm.specify_multilevel_moderated_gaze_mediation(p, moderation_path="b")
    assert mm._build_moderated_pymc_model(spec_b) is fake.model
    monkeypatch.setattr(mm, "_sampler_diagnostics", lambda idata, max_treedepth: {"max_rhat":1.0,"min_ess_bulk":1000,"divergences":0,"treedepth_hits":0})
    fit = mm.fit_multilevel_moderated_gaze_mediation(p, chains=2, draws=50, tune=0, cores=1)
    assert fit.specification.moderated

    posterior = {"a_within": np.ones(10)*.5, "b_within": np.ones(10)*.6, "moderation": np.ones(10)*.2}
    fit_a = mm.MultilevelMediationFit(specification=spec_a, posterior=posterior, sampler_diagnostics={"max_rhat":1.,"min_ess_bulk":1000,"divergences":0,"treedepth_hits":0})
    fit_b = mm.MultilevelMediationFit(specification=spec_b, posterior=posterior, sampler_diagnostics=fit_a.sampler_diagnostics)
    assert mm.posterior_conditional_indirect_effect(fit_a, moderator_value=1).mean == pytest.approx(.7*.6)
    assert mm.posterior_conditional_indirect_effect(fit_b, moderator_value=1).mean == pytest.approx(.5*.8)
    with pytest.raises(mm.GP3BayesError, match="finite numeric"):
        mm.posterior_conditional_indirect_effect(fit_a, moderator_value=np.inf)
    missing = mm.MultilevelMediationFit(specification=spec_a, posterior={}, sampler_diagnostics=fit_a.sampler_diagnostics)
    with pytest.raises(mm.GP3BayesError, match="unavailable"):
        mm.posterior_conditional_indirect_effect(missing, moderator_value=0)
    with pytest.raises(mm.GP3BayesError, match="not a moderated"):
        mm.posterior_conditional_indirect_effect(passing_fit(), moderator_value=0)

    monkeypatch.setattr(mm, "_sampler_diagnostics", lambda idata, max_treedepth: {"max_rhat":1.2,"min_ess_bulk":10,"divergences":1,"treedepth_hits":0})
    with pytest.warns(RuntimeWarning, match="Moderated mediation fit failed"):
        mm.fit_multilevel_moderated_gaze_mediation(p, chains=2, draws=50, tune=0, cores=1)


def test_full_simple_fit_with_fake_sampler_and_warning(monkeypatch):
    p = prepared()
    fake = FakePM(len(p.data))
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    monkeypatch.setattr(mm, "_sampler_diagnostics", lambda idata, max_treedepth: {
        "max_rhat": 1.0, "min_ess_bulk": 1000, "divergences": 0, "treedepth_hits": 0
    })
    fit = mm.fit_multilevel_gaze_mediation(
        p, random_slopes=(), chains=2, draws=50, tune=0, cores=1, seed=2, target_accept=.9, max_treedepth=6
    )
    assert fit.fit_performed and fit.backend_fit is not None
    assert "indirect_within" in fit.posterior

    monkeypatch.setattr(mm, "_sampler_diagnostics", lambda idata, max_treedepth: {
        "max_rhat": 1.2, "min_ess_bulk": 20, "divergences": 1, "treedepth_hits": 1
    })
    with pytest.warns(RuntimeWarning, match="failed one or more critical convergence"):
        mm.fit_multilevel_gaze_mediation(
            p, random_slopes=(), chains=2, draws=50, tune=0, cores=1, max_treedepth=6
        )


def test_sampler_empty_metrics_missing_stats_and_ppc_no_truncation(monkeypatch):
    class EmptyAZ:
        def rhat(self, idata): return FakeDataVars([])
        def ess(self, idata, method="bulk"): return FakeDataVars([])
    real_import = pyimportlib.import_module
    monkeypatch.setattr(mm, "import_module", lambda name: EmptyAZ() if name == "arviz" else real_import(name))
    d = mm._sampler_diagnostics(SimpleNamespace(sample_stats={}), max_treedepth=6)
    assert np.isnan(d["max_rhat"]) and np.isnan(d["min_ess_bulk"])
    assert d["treedepth_hits"] == 0

    p = prepared()
    spec = mm.specify_multilevel_gaze_mediation(p, random_slopes=())
    fake = FakePM(spec.analysis_rows)
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    fit = passing_fit(p, backend_fit=object(), backend_model=fake.model)
    out = mm.posterior_predictive_check_mediation(fit, draws=10)
    assert len(out) == 2

    import matplotlib.pyplot as plt
    participant = passing_fit(posterior={"participant_indirect_within": np.ones((4, 3))})
    _, ax = plt.subplots()
    assert mm.plot_participant_mediation_effects(participant, ax=ax) is ax


def test_serial_variation_false_branches_and_between_optional_builder(monkeypatch):
    p = augment_serial(prepared())
    original = mm._has_variation
    # Each invocation reaches a different false branch; keep required within paths true unless testing its guard.
    for false_index in [1, 3, 5]:  # X_between, M_between, M2_between
        calls = {"i": 0}
        def sequence(values, tolerance=1e-12, *, _idx=false_index):
            i = calls["i"]; calls["i"] += 1
            return False if i == _idx else True
        monkeypatch.setattr(mm, "_has_variation", sequence)
        spec = mm.specify_multilevel_serial_gaze_mediation(p)
        assert spec.serial
    # Required within branches false trigger the estimability guard.
    for false_index in [0, 2, 4]:
        calls = {"i": 0}
        def sequence_required(values, tolerance=1e-12, *, _idx=false_index):
            i = calls["i"]; calls["i"] += 1
            return False if i == _idx else True
        monkeypatch.setattr(mm, "_has_variation", sequence_required)
        with pytest.raises(mm.GP3BayesError, match="serial indirect effect is not estimable"):
            mm.specify_multilevel_serial_gaze_mediation(p)
    monkeypatch.setattr(mm, "_has_variation", original)

    # Build a valid serial model with all between paths removed to exercise optional branches.
    spec = mm.specify_multilevel_serial_gaze_mediation(p)
    no_between = tuple(x for x in spec.estimable_paths if not x.endswith("_between"))
    spec2 = mm.MultilevelMediationSpecification(
        **{**{name: getattr(spec, name) for name in spec.__dataclass_fields__}, "estimable_paths": no_between}
    )
    fake = FakePM(spec2.analysis_rows)
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    assert mm._build_serial_pymc_model(spec2) is fake.model


def test_moderated_empty_interaction_and_postfilter_estimability(monkeypatch):
    p = augment_moderator(prepared())
    q = copy.deepcopy(p)
    q.data["Z_within"] = np.nan
    with pytest.raises(mm.GP3BayesError, match="No rows remain after applying moderator"):
        mm.specify_multilevel_moderated_gaze_mediation(q, missingness_policy="complete_case")

    base = mm.specify_multilevel_gaze_mediation(p, random_slopes=())
    original_specify = mm.specify_multilevel_gaze_mediation
    monkeypatch.setattr(mm, "specify_multilevel_gaze_mediation", lambda *args, **kwargs: base)
    original_has = mm._has_variation
    calls = {"i": 0}
    def moderator_then_no_interaction(values, tolerance=1e-12):
        calls["i"] += 1
        return calls["i"] == 1
    monkeypatch.setattr(mm, "_has_variation", moderator_then_no_interaction)
    with pytest.raises(mm.GP3BayesError, match="interaction has no observed variation"):
        mm.specify_multilevel_moderated_gaze_mediation(p)
    monkeypatch.setattr(mm, "_has_variation", original_has)

    monkeypatch.setattr(mm, "_estimable_simple_paths", lambda data: ("a_within",))
    with pytest.raises(mm.GP3BayesError, match="indirect effect is not estimable after moderator filtering"):
        mm.specify_multilevel_moderated_gaze_mediation(p)
    monkeypatch.setattr(mm, "specify_multilevel_gaze_mediation", original_specify)


def test_moderated_builder_optional_between_paths_false(monkeypatch):
    p = augment_moderator(prepared())
    spec = mm.specify_multilevel_moderated_gaze_mediation(p, moderation_path="a")
    no_between = tuple(x for x in spec.estimable_paths if not x.endswith("_between"))
    spec2 = mm.MultilevelMediationSpecification(
        **{**{name: getattr(spec, name) for name in spec.__dataclass_fields__}, "estimable_paths": no_between}
    )
    fake = FakePM(spec2.analysis_rows)
    monkeypatch.setattr(mm, "_load_pymc", lambda: fake)
    assert mm._build_moderated_pymc_model(spec2) is fake.model


def test_serial_and_moderated_contract_guards_without_eyeprocess_dependency():
    base = prepared()
    with pytest.raises(mm.GP3BayesError, match="Serial mediation requires"):
        mm.specify_multilevel_serial_gaze_mediation(base)
    serial = augment_serial(base)
    with pytest.raises(mm.GP3BayesError, match="random slopes are not yet implemented"):
        mm.specify_multilevel_serial_gaze_mediation(serial, random_slopes=("mediator_x",))
    moderated = augment_moderator(base)
    with pytest.raises(mm.GP3BayesError, match="random slopes are not yet implemented"):
        mm.specify_multilevel_moderated_gaze_mediation(moderated, random_slopes=("outcome_m",))
