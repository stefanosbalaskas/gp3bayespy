"""Bayesian multilevel mediation for trial-level gaze and behavioural data.

This module consumes the canonical preparation contract produced by
``eyeprocesspy.prepare_multilevel_mediation_data``. It does not reimplement
within/between decomposition.

The primary estimands are products of path coefficients on the model's linear
predictor scale. For nonlinear outcome families these are *not* probability-
scale natural indirect effects; the scale is therefore stored and reported
explicitly.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np
import pandas as pd

GP3BayesError: type[ValueError]
BackendUnavailableError: type[RuntimeError]

try:  # integration path inside gp3bayespy
    from .exceptions import (
        BackendUnavailableError as _PackageBackendUnavailableError,
    )
    from .exceptions import (
        GP3BayesError as _PackageGP3BayesError,
    )

    GP3BayesError = _PackageGP3BayesError
    BackendUnavailableError = _PackageBackendUnavailableError

except Exception:  # standalone tranche validation

    class _FallbackGP3BayesError(ValueError):
        pass

    class _FallbackBackendUnavailableError(RuntimeError):
        pass

    GP3BayesError = _FallbackGP3BayesError
    BackendUnavailableError = _FallbackBackendUnavailableError


_MEDIATOR_FAMILIES = {
    "gaussian",
    "lognormal",
    "gamma",
    "beta",
    "bernoulli",
    "poisson",
    "negative_binomial",
}
_OUTCOME_FAMILIES = {"gaussian", "bernoulli", "poisson", "negative_binomial", "ordinal"}
_MISSING_POLICIES = {"error", "complete_case", "quality_eligible"}
_RANDOM_SLOPES = {"mediator_x", "outcome_x", "outcome_m"}


@dataclass(frozen=True, slots=True)
class MediationPriorSpecification:
    intercept_sd: float = 2.5
    coefficient_sd: float = 1.0
    group_sd_scale: float = 1.0
    residual_sd_scale: float = 1.0
    dispersion_rate: float = 1.0
    beta_precision_rate: float = 0.1

    def as_dict(self) -> dict[str, float]:
        return {
            "intercept_sd": self.intercept_sd,
            "coefficient_sd": self.coefficient_sd,
            "group_sd_scale": self.group_sd_scale,
            "residual_sd_scale": self.residual_sd_scale,
            "dispersion_rate": self.dispersion_rate,
            "beta_precision_rate": self.beta_precision_rate,
        }


@dataclass(frozen=True, slots=True)
class MultilevelMediationSpecification:
    data: pd.DataFrame
    participant_col: str
    trial_col: str
    outcome_col: str
    mediator_col: str
    mediator_family: str
    outcome_family: str
    priors: MediationPriorSpecification
    random_slopes: tuple[str, ...]
    missingness_policy: str
    input_rows: int
    analysis_rows: int
    excluded_row_positions: tuple[int, ...]
    provenance: Mapping[str, Any]
    model_kind: str = "simple"
    extra_columns: Mapping[str, str] = field(default_factory=dict)
    moderation: Mapping[str, Any] = field(default_factory=dict)
    serial: bool = False
    moderated: bool = False
    estimable_paths: tuple[str, ...] = ()
    model_version: str = "0.1"
    fitting_engine: str = "none"
    fit_performed: bool = False


@dataclass(frozen=True, slots=True)
class MultilevelMediationFit:
    specification: MultilevelMediationSpecification
    posterior: Mapping[str, np.ndarray]
    sampler_diagnostics: Mapping[str, Any]
    backend_fit: Any = None
    backend_model: Any = None
    backend: str = "pymc"
    package_versions: Mapping[str, str] = field(default_factory=dict)
    fit_performed: bool = True


@dataclass(frozen=True, slots=True)
class MediationConvergence:
    status: str
    passed: bool
    max_rhat: float
    min_ess_bulk: float
    divergences: int
    treedepth_hits: int
    thresholds: Mapping[str, float]
    issues: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MediationEffect:
    effect: str
    draws: np.ndarray
    scale: str
    probability: float
    mean: float
    median: float
    sd: float
    lower: float
    upper: float
    probability_positive: float
    probability_negative: float


def _normalize_family(value: str, allowed: set[str], name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise GP3BayesError(f"`{name}` must be a non-empty family name.")
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "normal": "gaussian",
        "log_normal": "lognormal",
        "binary": "bernoulli",
        "binomial": "bernoulli",
        "count": "poisson",
        "negbin": "negative_binomial",
        "negativebinomial": "negative_binomial",
        "ordered": "ordinal",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in allowed:
        raise GP3BayesError(
            f"Unsupported `{name}` `{value}`. Supported values: {', '.join(sorted(allowed))}."
        )
    return normalized


def _finite_positive(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise GP3BayesError(f"`{name}` must be one finite positive number.")
    return float(value)


def create_mediation_prior_specification(
    *,
    intercept_sd: float = 2.5,
    coefficient_sd: float = 1.0,
    group_sd_scale: float = 1.0,
    residual_sd_scale: float = 1.0,
    dispersion_rate: float = 1.0,
    beta_precision_rate: float = 0.1,
) -> MediationPriorSpecification:
    """Create explicit weakly-informative default priors for mediation paths."""
    return MediationPriorSpecification(
        intercept_sd=_finite_positive(intercept_sd, "intercept_sd"),
        coefficient_sd=_finite_positive(coefficient_sd, "coefficient_sd"),
        group_sd_scale=_finite_positive(group_sd_scale, "group_sd_scale"),
        residual_sd_scale=_finite_positive(residual_sd_scale, "residual_sd_scale"),
        dispersion_rate=_finite_positive(dispersion_rate, "dispersion_rate"),
        beta_precision_rate=_finite_positive(beta_precision_rate, "beta_precision_rate"),
    )


def _prepared_contract(
    prepared: Any,
) -> tuple[pd.DataFrame, Mapping[str, str | None], Mapping[str, Any]]:
    if not hasattr(prepared, "data") or not isinstance(prepared.data, pd.DataFrame):
        raise GP3BayesError(
            "`prepared` must be an eyeprocess/eyeprocesspy multilevel mediation preparation object."
        )
    columns = getattr(prepared, "columns", None)
    provenance = getattr(prepared, "provenance", None)
    if not isinstance(columns, Mapping):
        raise GP3BayesError("`prepared` does not expose the canonical mediation column contract.")
    if not isinstance(provenance, Mapping):
        provenance = {}
    required_semantic = {"participant", "trial", "mediator", "outcome"}
    if not required_semantic.issubset(columns):
        raise GP3BayesError("`prepared$columns` is missing canonical mediation semantics.")
    required_data = {
        "X_within",
        "X_between",
        "M_within",
        "M_between",
        "mediation_analysis_eligible",
    }
    missing = sorted(required_data - set(prepared.data.columns))
    if missing:
        raise GP3BayesError(
            "Prepared data is missing canonical columns: " + ", ".join(missing) + "."
        )
    return prepared.data.copy(deep=True), columns, provenance


def _validate_family_data(
    data: pd.DataFrame,
    mediator_col: str,
    outcome_col: str,
    mediator_family: str,
    outcome_family: str,
) -> None:
    m = pd.to_numeric(data[mediator_col], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(data[outcome_col], errors="raise").to_numpy(dtype=float)
    if mediator_family in {"lognormal", "gamma"} and not np.all(m > 0):
        raise GP3BayesError(
            f"Mediator family `{mediator_family}` requires strictly positive mediator values."
        )
    if mediator_family == "beta" and not np.all((m > 0) & (m < 1)):
        raise GP3BayesError(
            "Mediator family `beta` requires mediator values strictly between 0 and 1."
        )
    if mediator_family == "bernoulli" and not np.all(np.isin(m, [0, 1])):
        raise GP3BayesError("Mediator family `bernoulli` requires mediator values coded 0/1.")
    if mediator_family in {"poisson", "negative_binomial"} and not np.all(
        (m >= 0) & (np.floor(m) == m)
    ):
        raise GP3BayesError(
            f"Mediator family `{mediator_family}` requires non-negative integer counts."
        )
    if outcome_family == "bernoulli" and not np.all(np.isin(y, [0, 1])):
        raise GP3BayesError("Outcome family `bernoulli` requires outcome values coded 0/1.")
    if outcome_family in {"poisson", "negative_binomial"} and not np.all(
        (y >= 0) & (np.floor(y) == y)
    ):
        raise GP3BayesError(
            f"Outcome family `{outcome_family}` requires non-negative integer counts."
        )
    if outcome_family == "ordinal":
        unique = np.unique(y)
        if len(unique) < 3 or not np.array_equal(unique, np.arange(unique.min(), unique.max() + 1)):
            raise GP3BayesError(
                "Ordinal outcomes must be contiguous integer categories with at least three levels."
            )


def _has_variation(values: pd.Series | np.ndarray, *, tolerance: float = 1e-12) -> bool:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    return bool(arr.size >= 2 and np.nanmax(arr) - np.nanmin(arr) > tolerance)


def _estimable_simple_paths(data: pd.DataFrame) -> tuple[str, ...]:
    paths: list[str] = []
    if _has_variation(data["X_within"]):
        paths.extend(["a_within", "cprime_within"])
    if _has_variation(data["X_between"]):
        paths.extend(["a_between", "cprime_between"])
    if _has_variation(data["M_within"]):
        paths.append("b_within")
    if _has_variation(data["M_between"]):
        paths.append("b_between")
    return tuple(paths)


def _validate_sampling_controls(
    *,
    chains: int,
    draws: int,
    tune: int,
    cores: int | None,
    seed: int,
    target_accept: float,
    max_treedepth: int,
) -> dict[str, int | float]:
    for value, name, minimum in [
        (chains, "chains", 2),
        (draws, "draws", 50),
        (tune, "tune", 0),
        (max_treedepth, "max_treedepth", 5),
    ]:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or int(value) < minimum
        ):
            raise GP3BayesError(f"`{name}` must be an integer >= {minimum}.")
    if cores is None:
        cores_value = min(int(chains), 2)
    elif isinstance(cores, bool) or not isinstance(cores, (int, np.integer)) or int(cores) < 1:
        raise GP3BayesError("`cores` must be a positive integer or None.")
    else:
        cores_value = int(cores)
    if cores_value > int(chains):
        raise GP3BayesError("`cores` cannot exceed `chains`.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise GP3BayesError("`seed` must be a non-negative integer.")
    if (
        isinstance(target_accept, bool)
        or not isinstance(target_accept, (int, float))
        or not math.isfinite(float(target_accept))
        or not 0 < float(target_accept) < 1
    ):
        raise GP3BayesError("`target_accept` must lie strictly between 0 and 1.")
    return {
        "chains": int(chains),
        "draws": int(draws),
        "tune": int(tune),
        "cores": cores_value,
        "seed": int(seed),
        "target_accept": float(target_accept),
        "max_treedepth": int(max_treedepth),
    }


def _validate_draw_seed(draws: int, seed: int) -> tuple[int, int]:
    if isinstance(draws, bool) or not isinstance(draws, (int, np.integer)) or int(draws) < 1:
        raise GP3BayesError("`draws` must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise GP3BayesError("`seed` must be a non-negative integer.")
    return int(draws), int(seed)


def specify_multilevel_gaze_mediation(
    prepared: Any,
    *,
    mediator_family: str = "gaussian",
    outcome_family: str = "bernoulli",
    priors: MediationPriorSpecification | None = None,
    random_slopes: Sequence[str] = ("mediator_x",),
    missingness_policy: str = "error",
) -> MultilevelMediationSpecification:
    """Create a backend-independent multilevel mediation specification.

    ``missingness_policy='error'`` is intentionally conservative. Explicit
    ``complete_case`` or ``quality_eligible`` policies are available, and every
    excluded input row is recorded in the returned specification.
    """
    data, columns, provenance = _prepared_contract(prepared)
    mediator_family = _normalize_family(mediator_family, _MEDIATOR_FAMILIES, "mediator_family")
    outcome_family = _normalize_family(outcome_family, _OUTCOME_FAMILIES, "outcome_family")
    if missingness_policy not in _MISSING_POLICIES:
        raise GP3BayesError(
            "`missingness_policy` must be one of: " + ", ".join(sorted(_MISSING_POLICIES)) + "."
        )
    slopes = tuple(random_slopes)
    invalid_slopes = sorted(set(slopes) - _RANDOM_SLOPES)
    if invalid_slopes:
        raise GP3BayesError("Unknown random slope(s): " + ", ".join(invalid_slopes) + ".")
    if len(set(slopes)) != len(slopes):
        raise GP3BayesError("`random_slopes` must not contain duplicates.")
    priors = priors or create_mediation_prior_specification()
    if not isinstance(priors, MediationPriorSpecification):
        raise GP3BayesError("`priors` must be a MediationPriorSpecification.")

    mediator_col = str(columns["mediator"])
    outcome_col = str(columns["outcome"])
    participant_col = str(columns["participant"])
    trial_col = str(columns["trial"])
    required = [
        participant_col,
        trial_col,
        mediator_col,
        outcome_col,
        "X_within",
        "X_between",
        "M_within",
        "M_between",
    ]
    missing_cols = [c for c in required if c not in data]
    if missing_cols:
        raise GP3BayesError(
            "Prepared data is missing model columns: " + ", ".join(missing_cols) + "."
        )

    complete = data[required].notna().all(axis=1)
    eligible = data["mediation_analysis_eligible"].fillna(False).astype(bool)
    if missingness_policy == "error":
        if not bool((complete & eligible).all()):
            n_problem = int((~(complete & eligible)).sum())
            raise GP3BayesError(
                f"{n_problem} trial row(s) are incomplete or quality-ineligible. "
                "Choose an explicit `missingness_policy` after reviewing the preparation audits."
            )
        keep = pd.Series(True, index=data.index)
    elif missingness_policy == "complete_case":
        keep = complete
    else:
        keep = complete & eligible

    excluded_positions = tuple(np.flatnonzero(~keep.to_numpy()).astype(int).tolist())
    analysis = data.loc[keep].copy()
    if analysis.empty:
        raise GP3BayesError("No rows remain under the requested missingness policy.")
    if analysis[participant_col].nunique() < 2:
        raise GP3BayesError("At least two participants are required for multilevel mediation.")
    _validate_family_data(analysis, mediator_col, outcome_col, mediator_family, outcome_family)
    estimable_paths = _estimable_simple_paths(analysis)
    required_indirect = {"a_within", "b_within"}
    if not required_indirect.issubset(estimable_paths):
        missing_paths = sorted(required_indirect - set(estimable_paths))
        raise GP3BayesError(
            "The within-participant indirect effect is not estimable from the observed design; "
            "missing variation for path(s): " + ", ".join(missing_paths) + "."
        )
    slope_path = {"mediator_x": "a_within", "outcome_x": "cprime_within", "outcome_m": "b_within"}
    unsupported_slopes = [name for name in slopes if slope_path[name] not in estimable_paths]
    if unsupported_slopes:
        raise GP3BayesError(
            "Requested random slope(s) are not estimable from observed within-participant variation: "
            + ", ".join(unsupported_slopes)
            + "."
        )

    # Trial-level random slopes need repeated observations within at least some groups.
    counts = analysis.groupby(participant_col)[trial_col].nunique()
    if slopes and int((counts >= 2).sum()) < 2:
        raise GP3BayesError("Random slopes require repeated trials for at least two participants.")

    model_provenance = {
        "preparation": dict(provenance),
        "model_specification": {
            "mediator_family": mediator_family,
            "outcome_family": outcome_family,
            "random_slopes": list(slopes),
            "missingness_policy": missingness_policy,
            "priors": priors.as_dict(),
            "estimand_scale": "linear_predictor_product",
            "estimable_paths": list(estimable_paths),
        },
    }
    return MultilevelMediationSpecification(
        data=analysis,
        participant_col=participant_col,
        trial_col=trial_col,
        outcome_col=outcome_col,
        mediator_col=mediator_col,
        mediator_family=mediator_family,
        outcome_family=outcome_family,
        priors=priors,
        random_slopes=slopes,
        missingness_policy=missingness_policy,
        input_rows=len(data),
        analysis_rows=len(analysis),
        excluded_row_positions=excluded_positions,
        provenance=model_provenance,
        estimable_paths=estimable_paths,
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not_installed"


def _load_pymc() -> Any:
    try:
        return import_module("pymc")
    except Exception as exc:
        raise BackendUnavailableError(
            "PyMC could not be imported for multilevel mediation. Install/repair the "
            "gp3bayespy Bayesian backend and its compatible ArviZ/xarray dependencies."
        ) from exc


def _normal_paths(pm: Any, sd: float) -> dict[str, Any]:
    return {
        name: pm.Normal(name, mu=0.0, sigma=sd)
        for name in [
            "a_within",
            "a_between",
            "b_within",
            "b_between",
            "cprime_within",
            "cprime_between",
        ]
    }


def _random_effect(
    pm: Any, name: str, n_participants: int, participant_idx: np.ndarray, scale: float
) -> Any:
    sd = pm.HalfNormal(f"sd_{name}", sigma=scale)
    z = pm.Normal(f"z_{name}", 0.0, 1.0, shape=n_participants)
    return sd * z[participant_idx]


def _random_slope(
    pm: Any,
    name: str,
    predictor: np.ndarray,
    n_participants: int,
    participant_idx: np.ndarray,
    scale: float,
) -> Any:
    sd = pm.HalfNormal(f"sd_{name}", sigma=scale)
    z = pm.Normal(f"z_{name}", 0.0, 1.0, shape=n_participants)
    return sd * z[participant_idx] * predictor


def _observe_family(
    pm: Any,
    name: str,
    family: str,
    eta: Any,
    observed: np.ndarray,
    priors: MediationPriorSpecification,
) -> None:
    if family == "gaussian":
        sigma = pm.HalfNormal(f"sigma_{name}", sigma=priors.residual_sd_scale)
        pm.Normal(name, mu=eta, sigma=sigma, observed=observed)
    elif family == "lognormal":
        sigma = pm.HalfNormal(f"sigma_{name}", sigma=priors.residual_sd_scale)
        pm.LogNormal(name, mu=eta, sigma=sigma, observed=observed)
    elif family == "gamma":
        shape = pm.Exponential(f"shape_{name}", lam=priors.dispersion_rate)
        mu = pm.math.exp(eta)
        pm.Gamma(name, alpha=shape, beta=shape / mu, observed=observed)
    elif family == "beta":
        precision = pm.Exponential(f"precision_{name}", lam=priors.beta_precision_rate)
        mu = pm.math.sigmoid(eta)
        pm.Beta(name, alpha=mu * precision, beta=(1.0 - mu) * precision, observed=observed)
    elif family == "bernoulli":
        pm.Bernoulli(name, logit_p=eta, observed=observed.astype(int))
    elif family == "poisson":
        pm.Poisson(name, mu=pm.math.exp(eta), observed=observed.astype(int))
    elif family == "negative_binomial":
        alpha = pm.Exponential(f"alpha_{name}", lam=priors.dispersion_rate)
        pm.NegativeBinomial(name, mu=pm.math.exp(eta), alpha=alpha, observed=observed.astype(int))
    elif family == "ordinal":
        categories = np.unique(observed.astype(int))
        n_cut = len(categories) - 1
        init = np.linspace(-1.0, 1.0, n_cut)
        cutpoints = pm.Normal(
            f"cutpoints_{name}",
            mu=init,
            sigma=1.5,
            shape=n_cut,
            transform=pm.distributions.transforms.ordered,
            initval=init,
        )
        pm.OrderedLogistic(name, eta=eta, cutpoints=cutpoints, observed=observed.astype(int))
    else:  # pragma: no cover - validation protects this
        raise GP3BayesError(f"Unsupported family `{family}`.")


def _build_pymc_model(spec: MultilevelMediationSpecification) -> Any:
    pm = _load_pymc()
    data = spec.data
    participant_idx, levels = pd.factorize(data[spec.participant_col], sort=False)
    n_participants = len(levels)
    xw = data["X_within"].to_numpy(dtype=float)
    xb = data["X_between"].to_numpy(dtype=float)
    mw = data["M_within"].to_numpy(dtype=float)
    mb = data["M_between"].to_numpy(dtype=float)
    mediator = pd.to_numeric(data[spec.mediator_col], errors="raise").to_numpy(dtype=float)
    outcome = pd.to_numeric(data[spec.outcome_col], errors="raise").to_numpy(dtype=float)
    priors = spec.priors
    estimable = set(spec.estimable_paths)

    with pm.Model() as model:
        paths = {name: pm.Normal(name, 0.0, priors.coefficient_sd) for name in estimable}
        alpha_m = pm.Normal("alpha_m", 0.0, priors.intercept_sd)
        alpha_y = pm.Normal("alpha_y", 0.0, priors.intercept_sd)
        eta_m = alpha_m + paths["a_within"] * xw
        if "a_between" in paths:
            eta_m = eta_m + paths["a_between"] * xb
        eta_m = eta_m + _random_effect(
            pm, "participant_m", n_participants, participant_idx, priors.group_sd_scale
        )
        participant_a_slope = None
        if "mediator_x" in spec.random_slopes:
            sd_a = pm.HalfNormal("sd_participant_a", sigma=priors.group_sd_scale)
            z_a = pm.Normal("z_participant_a", 0.0, 1.0, shape=n_participants)
            participant_a_slope = pm.Deterministic("participant_a_slope", sd_a * z_a)
            eta_m = eta_m + participant_a_slope[participant_idx] * xw
        _observe_family(pm, "M_obs", spec.mediator_family, eta_m, mediator, priors)

        eta_y = alpha_y + paths["cprime_within"] * xw + paths["b_within"] * mw
        for name, predictor in [("cprime_between", xb), ("b_between", mb)]:
            if name in paths:
                eta_y = eta_y + paths[name] * predictor
        eta_y = eta_y + _random_effect(
            pm, "participant_y", n_participants, participant_idx, priors.group_sd_scale
        )
        if "outcome_x" in spec.random_slopes:
            eta_y = eta_y + _random_slope(
                pm, "participant_c", xw, n_participants, participant_idx, priors.group_sd_scale
            )
        participant_b_slope = None
        if "outcome_m" in spec.random_slopes:
            sd_b = pm.HalfNormal("sd_participant_b", sigma=priors.group_sd_scale)
            z_b = pm.Normal("z_participant_b", 0.0, 1.0, shape=n_participants)
            participant_b_slope = pm.Deterministic("participant_b_slope", sd_b * z_b)
            eta_y = eta_y + participant_b_slope[participant_idx] * mw
        _observe_family(pm, "Y_obs", spec.outcome_family, eta_y, outcome, priors)

        pm.Deterministic("indirect_within", paths["a_within"] * paths["b_within"])
        if {"a_between", "b_between"}.issubset(paths):
            pm.Deterministic("indirect_between", paths["a_between"] * paths["b_between"])
        pm.Deterministic(
            "total_within", paths["cprime_within"] + paths["a_within"] * paths["b_within"]
        )
        if {"cprime_between", "a_between", "b_between"}.issubset(paths):
            pm.Deterministic(
                "total_between", paths["cprime_between"] + paths["a_between"] * paths["b_between"]
            )
        if participant_a_slope is not None and participant_b_slope is not None:
            pm.Deterministic(
                "participant_indirect_within",
                (paths["a_within"] + participant_a_slope)
                * (paths["b_within"] + participant_b_slope),
            )
    return model


def _flatten_posterior(idata: Any, names: Sequence[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    posterior = getattr(idata, "posterior", None)
    if posterior is None:
        return out
    for name in names:
        try:
            values = np.asarray(posterior[name])
        except Exception:
            continue
        if values.ndim >= 2:
            values = values.reshape((-1,) + values.shape[2:])
        out[name] = np.asarray(values)
    return out


def _sampler_diagnostics(idata: Any, *, max_treedepth: int) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "max_rhat": np.nan,
        "min_ess_bulk": np.nan,
        "divergences": 0,
        "treedepth_hits": 0,
    }
    try:
        az = import_module("arviz")
        rhat = az.rhat(idata)
        ess = az.ess(idata, method="bulk")
        rhat_values = [np.asarray(v).ravel() for v in rhat.data_vars.values()]
        ess_values = [np.asarray(v).ravel() for v in ess.data_vars.values()]
        if rhat_values:
            diagnostics["max_rhat"] = float(np.nanmax(np.concatenate(rhat_values)))
        if ess_values:
            diagnostics["min_ess_bulk"] = float(np.nanmin(np.concatenate(ess_values)))
    except Exception:
        pass
    stats = getattr(idata, "sample_stats", None)
    if stats is not None:
        with suppress(Exception):
            diagnostics["divergences"] = int(np.asarray(stats["diverging"]).sum())
        for candidate in ("tree_depth", "treedepth"):
            try:
                values = np.asarray(stats[candidate])
                diagnostics["treedepth_hits"] = int(np.sum(values >= max_treedepth))
                break
            except Exception:
                continue
    return diagnostics


def fit_multilevel_gaze_mediation(
    prepared: Any,
    *,
    mediator_family: str = "gaussian",
    outcome_family: str = "bernoulli",
    priors: MediationPriorSpecification | None = None,
    random_slopes: Sequence[str] = ("mediator_x",),
    missingness_policy: str = "error",
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    target_accept: float = 0.95,
    max_treedepth: int = 12,
) -> MultilevelMediationFit:
    """Fit a Bayesian trial-level multilevel mediation model with PyMC NUTS."""
    spec = specify_multilevel_gaze_mediation(
        prepared,
        mediator_family=mediator_family,
        outcome_family=outcome_family,
        priors=priors,
        random_slopes=random_slopes,
        missingness_policy=missingness_policy,
    )
    controls = _validate_sampling_controls(
        chains=chains,
        draws=draws,
        tune=tune,
        cores=cores,
        seed=seed,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
    )

    pm = _load_pymc()
    model = _build_pymc_model(spec)
    with model:
        idata = pm.sample(
            draws=controls["draws"],
            tune=controls["tune"],
            chains=controls["chains"],
            cores=controls["cores"],
            random_seed=controls["seed"],
            target_accept=controls["target_accept"],
            nuts={"max_treedepth": controls["max_treedepth"]},
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    names = [
        "a_within",
        "a_between",
        "b_within",
        "b_between",
        "cprime_within",
        "cprime_between",
        "indirect_within",
        "indirect_between",
        "total_within",
        "total_between",
        "participant_a_slope",
        "participant_b_slope",
        "participant_indirect_within",
    ]
    posterior = _flatten_posterior(idata, names)
    diagnostics = _sampler_diagnostics(idata, max_treedepth=int(controls["max_treedepth"]))
    fit = MultilevelMediationFit(
        specification=spec,
        posterior=posterior,
        sampler_diagnostics=diagnostics,
        backend_fit=idata,
        backend_model=model,
        package_versions={"pymc": _package_version("pymc"), "arviz": _package_version("arviz")},
    )
    convergence = check_mediation_convergence(fit)
    if not convergence.passed:
        warnings.warn(
            "Mediation fit failed one or more critical convergence checks. "
            "Indirect-effect extraction will fail by default until diagnostics are resolved.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fit


def check_mediation_convergence(
    fit: MultilevelMediationFit,
    *,
    rhat_max: float = 1.01,
    ess_bulk_min: float = 400.0,
    divergences_max: int = 0,
    treedepth_hits_max: int = 0,
) -> MediationConvergence:
    """Apply explicit convergence thresholds to a mediation fit."""
    if not isinstance(fit, MultilevelMediationFit):
        raise GP3BayesError("`fit` must be a MultilevelMediationFit.")
    d = fit.sampler_diagnostics
    max_rhat = float(d.get("max_rhat", np.nan))
    min_ess = float(d.get("min_ess_bulk", np.nan))
    divergences = int(d.get("divergences", 0))
    treedepth_hits = int(d.get("treedepth_hits", 0))
    issues: list[str] = []
    if not np.isfinite(max_rhat):
        issues.append("R-hat was not available")
    elif max_rhat > rhat_max:
        issues.append(f"max R-hat {max_rhat:.3f} exceeds {rhat_max:.3f}")
    if not np.isfinite(min_ess):
        issues.append("bulk ESS was not available")
    elif min_ess < ess_bulk_min:
        issues.append(f"minimum bulk ESS {min_ess:.0f} is below {ess_bulk_min:.0f}")
    if divergences > divergences_max:
        issues.append(f"{divergences} divergence(s) detected")
    if treedepth_hits > treedepth_hits_max:
        issues.append(f"{treedepth_hits} maximum-tree-depth hit(s) detected")
    passed = len(issues) == 0
    return MediationConvergence(
        status="pass" if passed else "fail",
        passed=passed,
        max_rhat=max_rhat,
        min_ess_bulk=min_ess,
        divergences=divergences,
        treedepth_hits=treedepth_hits,
        thresholds={
            "rhat_max": float(rhat_max),
            "ess_bulk_min": float(ess_bulk_min),
            "divergences_max": float(divergences_max),
            "treedepth_hits_max": float(treedepth_hits_max),
        },
        issues=tuple(issues),
    )


def _effect_from_draws(
    name: str, draws: np.ndarray, probability: float, scale: str
) -> MediationEffect:
    values = np.asarray(draws, dtype=float).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all():
        raise GP3BayesError(f"Posterior draws for `{name}` are missing or non-finite.")
    if not 0 < probability < 1:
        raise GP3BayesError("`probability` must lie strictly between 0 and 1.")
    alpha = (1.0 - probability) / 2.0
    lower, upper = np.quantile(values, [alpha, 1.0 - alpha])
    return MediationEffect(
        effect=name,
        draws=values,
        scale=scale,
        probability=probability,
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        sd=float(np.std(values, ddof=1)),
        lower=float(lower),
        upper=float(upper),
        probability_positive=float(np.mean(values > 0)),
        probability_negative=float(np.mean(values < 0)),
    )


def _require_acceptable_fit(fit: MultilevelMediationFit, require_convergence: bool) -> None:
    if require_convergence:
        result = check_mediation_convergence(fit)
        if not result.passed:
            raise GP3BayesError(
                "Critical convergence checks failed: " + "; ".join(result.issues) + ". "
                "Set `require_convergence=False` only for diagnostic inspection, not substantive reporting."
            )


def posterior_indirect_effect(
    fit: MultilevelMediationFit,
    *,
    level: str = "within",
    probability: float = 0.95,
    require_convergence: bool = True,
) -> MediationEffect:
    """Return posterior indirect-effect draws on the linear-predictor product scale."""
    _require_acceptable_fit(fit, require_convergence)
    if level not in {"within", "between"}:
        raise GP3BayesError("`level` must be `within` or `between`.")
    key = f"indirect_{level}"
    draws = fit.posterior.get(key)
    if draws is None:
        a = fit.posterior.get(f"a_{level}")
        b = fit.posterior.get(f"b_{level}")
        if a is None or b is None:
            raise GP3BayesError(f"Posterior path draws required for `{key}` are unavailable.")
        draws = np.asarray(a) * np.asarray(b)
    return _effect_from_draws(key, draws, probability, "linear_predictor_product")


def estimate_within_indirect_effect(fit: MultilevelMediationFit, **kwargs: Any) -> MediationEffect:
    return posterior_indirect_effect(fit, level="within", **kwargs)


def estimate_between_indirect_effect(fit: MultilevelMediationFit, **kwargs: Any) -> MediationEffect:
    return posterior_indirect_effect(fit, level="between", **kwargs)


def estimate_indirect_effect(
    fit: MultilevelMediationFit, *, level: str = "within", **kwargs: Any
) -> MediationEffect:
    return posterior_indirect_effect(fit, level=level, **kwargs)


def posterior_direct_effect(
    fit: MultilevelMediationFit,
    *,
    level: str = "within",
    probability: float = 0.95,
    require_convergence: bool = True,
) -> MediationEffect:
    _require_acceptable_fit(fit, require_convergence)
    if level not in {"within", "between"}:
        raise GP3BayesError("`level` must be `within` or `between`.")
    key = f"cprime_{level}"
    draws = fit.posterior.get(key)
    if draws is None:
        raise GP3BayesError(f"Posterior draws `{key}` are unavailable.")
    return _effect_from_draws(key, draws, probability, "linear_predictor")


def posterior_total_effect(
    fit: MultilevelMediationFit,
    *,
    level: str = "within",
    probability: float = 0.95,
    require_convergence: bool = True,
) -> MediationEffect:
    _require_acceptable_fit(fit, require_convergence)
    if level not in {"within", "between"}:
        raise GP3BayesError("`level` must be `within` or `between`.")
    key = f"total_{level}"
    draws = fit.posterior.get(key)
    if draws is None:
        c = fit.posterior.get(f"cprime_{level}")
        indirect = posterior_indirect_effect(
            fit, level=level, probability=probability, require_convergence=False
        ).draws
        if c is None:
            raise GP3BayesError(f"Posterior draws required for `{key}` are unavailable.")
        draws = np.asarray(c) + indirect
    return _effect_from_draws(key, draws, probability, "linear_predictor")


def summarise_multilevel_mediation(
    fit: MultilevelMediationFit,
    *,
    probability: float = 0.95,
    require_convergence: bool = True,
) -> pd.DataFrame:
    """Summarise within/between a, b, direct, indirect, and total effects."""
    _require_acceptable_fit(fit, require_convergence)
    rows: list[dict[str, Any]] = []
    for level in ("within", "between"):
        for name, scale in [
            (f"a_{level}", "linear_predictor"),
            (f"b_{level}", "linear_predictor"),
            (f"cprime_{level}", "linear_predictor"),
        ]:
            if name in fit.posterior:
                effect = _effect_from_draws(name, fit.posterior[name], probability, scale)
                rows.append(_effect_row(effect))
        with suppress(GP3BayesError):
            rows.append(
                _effect_row(
                    posterior_indirect_effect(
                        fit,
                        level=level,
                        probability=probability,
                        require_convergence=False,
                    )
                )
            )
        with suppress(GP3BayesError):
            rows.append(
                _effect_row(
                    posterior_total_effect(
                        fit,
                        level=level,
                        probability=probability,
                        require_convergence=False,
                    )
                )
            )
    return pd.DataFrame(rows)


def _effect_row(effect: MediationEffect) -> dict[str, Any]:
    return {
        "effect": effect.effect,
        "scale": effect.scale,
        "mean": effect.mean,
        "median": effect.median,
        "sd": effect.sd,
        "lower": effect.lower,
        "upper": effect.upper,
        "probability_positive": effect.probability_positive,
        "probability_negative": effect.probability_negative,
    }


def posterior_predictive_check_mediation(
    fit: MultilevelMediationFit,
    *,
    draws: int = 200,
    seed: int = 1,
) -> pd.DataFrame:
    """Run simple posterior-predictive checks for mediator and outcome."""
    draws, seed = _validate_draw_seed(draws, seed)
    if fit.backend_model is None or fit.backend_fit is None:
        raise GP3BayesError("Posterior predictive checks require a fitted backend model.")
    pm = _load_pymc()
    with fit.backend_model:
        ppc = pm.sample_posterior_predictive(
            fit.backend_fit,
            var_names=["M_obs", "Y_obs"],
            random_seed=seed,
            progressbar=False,
            predictions=False,
        )
    predictive = getattr(ppc, "posterior_predictive", None)
    if predictive is None:
        raise GP3BayesError("Backend did not return posterior predictive draws.")
    rows: list[dict[str, Any]] = []
    for var, observed_col in [
        ("M_obs", fit.specification.mediator_col),
        ("Y_obs", fit.specification.outcome_col),
    ]:
        sims = np.asarray(predictive[var]).reshape(-1, fit.specification.analysis_rows)
        if len(sims) > draws:
            sims = sims[:draws]
        observed = pd.to_numeric(fit.specification.data[observed_col], errors="raise").to_numpy(
            dtype=float
        )
        sim_means = sims.mean(axis=1)
        obs_mean = float(np.mean(observed))
        rows.append(
            {
                "variable": var,
                "observed_mean": obs_mean,
                "predictive_mean": float(np.mean(sim_means)),
                "predictive_mean_lower": float(np.quantile(sim_means, 0.025)),
                "predictive_mean_upper": float(np.quantile(sim_means, 0.975)),
                "tail_probability": float(np.mean(sim_means >= obs_mean)),
            }
        )
    return pd.DataFrame(rows)


def prior_predictive_check_mediation(
    specification: MultilevelMediationSpecification,
    *,
    draws: int = 200,
    seed: int = 1,
) -> pd.DataFrame:
    """Sample prior predictive mediator/outcome distributions from a specification."""
    if not isinstance(specification, MultilevelMediationSpecification):
        raise GP3BayesError("`specification` must be a MultilevelMediationSpecification.")
    draws, seed = _validate_draw_seed(draws, seed)
    pm = _load_pymc()
    model = _build_pymc_model(specification)
    with model:
        prior = pm.sample_prior_predictive(draws=draws, random_seed=seed)
    group = getattr(prior, "prior_predictive", None)
    if group is None:
        raise GP3BayesError("Backend did not return prior predictive draws.")
    rows = []
    for var in ("M_obs", "Y_obs"):
        sims = np.asarray(group[var]).reshape(-1, specification.analysis_rows)
        rows.append(
            {
                "variable": var,
                "draws": len(sims),
                "mean_of_means": float(np.mean(sims.mean(axis=1))),
                "minimum": float(np.min(sims)),
                "maximum": float(np.max(sims)),
            }
        )
    return pd.DataFrame(rows)


def _same_comparison_observations(
    reference: MultilevelMediationSpecification,
    candidate: MultilevelMediationSpecification,
) -> bool:
    """Return whether two specifications contain the same ordered responses."""
    if (
        reference.mediator_col != candidate.mediator_col
        or reference.outcome_col != candidate.outcome_col
    ):
        return False
    if reference.analysis_rows != candidate.analysis_rows:
        return False
    ref_keys = reference.data[[reference.participant_col, reference.trial_col]].reset_index(
        drop=True
    )
    cand_keys = candidate.data[[candidate.participant_col, candidate.trial_col]].reset_index(
        drop=True
    )
    if not ref_keys.set_axis(["participant", "trial"], axis=1).equals(
        cand_keys.set_axis(["participant", "trial"], axis=1)
    ):
        return False
    ref_values = reference.data[[reference.mediator_col, reference.outcome_col]].reset_index(
        drop=True
    )
    cand_values = candidate.data[[candidate.mediator_col, candidate.outcome_col]].reset_index(
        drop=True
    )
    return ref_values.set_axis(["mediator", "outcome"], axis=1).equals(
        cand_values.set_axis(["mediator", "outcome"], axis=1)
    )


def _joint_pointwise_log_likelihood(
    mediator_ll: Any, outcome_ll: Any, *, expected_rows: int
) -> Any:
    """Align mediator/outcome pointwise log likelihoods before summing them."""
    if hasattr(mediator_ll, "dims") and hasattr(outcome_ll, "dims"):
        sample_dims = {"chain", "draw", "sample"}
        mediator_obs_dims = [dim for dim in mediator_ll.dims if dim not in sample_dims]
        outcome_obs_dims = [dim for dim in outcome_ll.dims if dim not in sample_dims]
        if len(mediator_obs_dims) != 1 or len(outcome_obs_dims) != 1:
            raise GP3BayesError(
                "Mediator and outcome log-likelihood arrays must each have exactly one observation dimension."
            )
        mediator_dim = mediator_obs_dims[0]
        outcome_dim = outcome_obs_dims[0]
        if (
            int(mediator_ll.sizes[mediator_dim]) != expected_rows
            or int(outcome_ll.sizes[outcome_dim]) != expected_rows
        ):
            raise GP3BayesError(
                "Mediator/outcome log-likelihood observation dimensions do not match the analysis rows."
            )
        mediator_aligned = (
            mediator_ll
            if mediator_dim == "observation"
            else mediator_ll.rename({mediator_dim: "observation"})
        )
        outcome_aligned = (
            outcome_ll
            if outcome_dim == "observation"
            else outcome_ll.rename({outcome_dim: "observation"})
        )
        return mediator_aligned + outcome_aligned

    mediator_array = np.asarray(mediator_ll)
    outcome_array = np.asarray(outcome_ll)
    if (
        mediator_array.shape != outcome_array.shape
        or mediator_array.ndim < 1
        or mediator_array.shape[-1] != expected_rows
    ):
        raise GP3BayesError(
            "Mediator/outcome log-likelihood arrays must have matching shapes with the final dimension equal "
            "to the number of analysis rows."
        )
    return mediator_array + outcome_array


def compare_multilevel_mediation_models(fits: Mapping[str, MultilevelMediationFit]) -> pd.DataFrame:
    """Compare mediation fits using joint PSIS-LOO on identical observations only."""
    if not isinstance(fits, Mapping) or len(fits) < 2:
        raise GP3BayesError("`fits` must contain at least two named mediation fits.")
    fit_items = list(fits.items())
    for label, fit in fit_items:
        if not isinstance(fit, MultilevelMediationFit) or fit.backend_fit is None:
            raise GP3BayesError(f"Model `{label}` is not a fitted mediation object.")
    reference_label, reference_fit = fit_items[0]
    for label, fit in fit_items[1:]:
        if not _same_comparison_observations(reference_fit.specification, fit.specification):
            raise GP3BayesError(
                "PSIS-LOO comparison requires the same mediator/outcome observations in the same "
                f"participant-trial order; `{label}` does not match `{reference_label}`. "
                "Do not compare fits produced from different missingness/exclusion sets."
            )
    try:
        az = import_module("arviz")
    except Exception as exc:
        raise BackendUnavailableError("ArviZ is required for PSIS-LOO model comparison.") from exc
    rows: list[dict[str, Any]] = []
    for label, fit in fit_items:
        ll = getattr(fit.backend_fit, "log_likelihood", None)
        if ll is None or "M_obs" not in ll or "Y_obs" not in ll:
            raise GP3BayesError(f"Model `{label}` lacks mediator/outcome log-likelihood draws.")
        joint = _joint_pointwise_log_likelihood(
            ll["M_obs"], ll["Y_obs"], expected_rows=fit.specification.analysis_rows
        )
        temp = fit.backend_fit.copy()
        try:
            temp.log_likelihood["joint"] = joint
            loo = az.loo(temp, var_name="joint")
        except Exception as exc:
            raise GP3BayesError(f"Could not compute joint PSIS-LOO for `{label}`: {exc}") from exc
        rows.append(
            {
                "model": label,
                "elpd_loo": float(loo.elpd_loo),
                "se": float(loo.se),
                "p_loo": float(loo.p_loo),
                "warning": bool(getattr(loo, "warning", False)),
            }
        )
    return pd.DataFrame(rows).sort_values("elpd_loo", ascending=False).reset_index(drop=True)


def plot_indirect_effect_distribution(
    fit: MultilevelMediationFit,
    *,
    level: str = "within",
    probability: float = 0.95,
    require_convergence: bool = True,
    ax: Any = None,
) -> Any:
    """Plot the posterior distribution of an indirect effect."""
    effect = posterior_indirect_effect(
        fit, level=level, probability=probability, require_convergence=require_convergence
    )
    try:
        plt = import_module("matplotlib.pyplot")
    except Exception as exc:
        raise BackendUnavailableError("Matplotlib is required for mediation plots.") from exc
    if ax is None:
        _, ax = plt.subplots()
    ax.hist(effect.draws, bins=30, density=True)
    ax.axvline(0.0, linewidth=1)
    ax.axvline(effect.mean, linewidth=1.5)
    ax.set_xlabel(f"{level.title()} indirect effect ({effect.scale})")
    ax.set_ylabel("Posterior density")
    ax.set_title(f"Posterior {level} indirect effect")
    return ax


def plot_mediation_posteriors(
    fit: MultilevelMediationFit,
    *,
    probability: float = 0.95,
    require_convergence: bool = True,
    ax: Any = None,
) -> Any:
    """Plot posterior interval summaries for mediation paths and effects."""
    summary = summarise_multilevel_mediation(
        fit, probability=probability, require_convergence=require_convergence
    )
    try:
        plt = import_module("matplotlib.pyplot")
    except Exception as exc:
        raise BackendUnavailableError("Matplotlib is required for mediation plots.") from exc
    if ax is None:
        _, ax = plt.subplots()
    y = np.arange(len(summary))
    ax.errorbar(
        summary["mean"],
        y,
        xerr=[summary["mean"] - summary["lower"], summary["upper"] - summary["mean"]],
        fmt="o",
    )
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y, summary["effect"])
    ax.set_xlabel("Posterior effect")
    ax.set_title("Multilevel mediation posterior intervals")
    return ax


def plot_participant_mediation_effects(fit: MultilevelMediationFit, *, ax: Any = None) -> Any:
    """Plot participant random-effect scales when available.

    Participant-specific indirect effects are not manufactured when the fitted
    model did not include both participant-specific a and b paths.
    """
    keys = [k for k in fit.posterior if k.startswith("participant_indirect_")]
    if not keys:
        raise GP3BayesError(
            "Participant-specific indirect effects are unavailable because the model did not estimate "
            "participant-specific a and b paths jointly."
        )
    try:
        plt = import_module("matplotlib.pyplot")
    except Exception as exc:
        raise BackendUnavailableError("Matplotlib is required for mediation plots.") from exc
    if ax is None:
        _, ax = plt.subplots()
    values = np.asarray(fit.posterior[keys[0]])
    means = values.mean(axis=0)
    ax.plot(np.arange(len(means)), means, "o")
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Posterior mean indirect effect")
    return ax


def report_multilevel_gaze_mediation(
    fit: MultilevelMediationFit,
    *,
    probability: float = 0.95,
    require_convergence: bool = True,
) -> str:
    """Create a compact, conservative reporting paragraph."""
    convergence = check_mediation_convergence(fit)
    if require_convergence and not convergence.passed:
        raise GP3BayesError("A substantive report is blocked because convergence checks failed.")
    within = posterior_indirect_effect(
        fit, level="within", probability=probability, require_convergence=False
    )
    try:
        between = posterior_indirect_effect(
            fit, level="between", probability=probability, require_convergence=False
        )
        between_text = f"{between.mean:.3f} [{between.lower:.3f}, {between.upper:.3f}]"
    except GP3BayesError:
        between_text = "not estimable from the observed design"
    spec = fit.specification
    excluded = len(spec.excluded_row_positions)
    return (
        f"A Bayesian trial-level multilevel mediation model was fit to {spec.analysis_rows} observations "
        f"from {spec.data[spec.participant_col].nunique()} participants, using a {spec.mediator_family} "
        f"mediator model and {spec.outcome_family} outcome model. The explicit missingness policy was "
        f"'{spec.missingness_policy}' ({excluded} input rows excluded). The within-participant indirect "
        f"effect on the linear-predictor product scale was {within.mean:.3f} "
        f"[{within.lower:.3f}, {within.upper:.3f}], and the between-participant indirect effect was "
        f"{between_text}. Sampler diagnostics status: "
        f"{convergence.status}. For nonlinear families these coefficient products should not be interpreted "
        f"as probability-scale natural indirect effects."
    )


def simulate_multilevel_gaze_mediation(
    *,
    n_participants: int = 100,
    trials_per_participant: int = 20,
    a_within: float = 0.7,
    b_within: float = 0.8,
    cprime_within: float = 0.2,
    a_between: float = 0.2,
    b_between: float = 0.3,
    seed: int = 2026,
) -> pd.DataFrame:
    """Simulate the repeated AI-advice example used by tests and documentation."""
    if n_participants < 2 or trials_per_participant < 2:
        raise GP3BayesError(
            "Simulation requires at least two participants and two trials per participant."
        )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    participant_intercept_m = rng.normal(0, 0.5, n_participants)
    participant_intercept_y = rng.normal(0, 0.5, n_participants)
    participant_x_propensity = rng.beta(4, 4, n_participants)
    for i in range(n_participants):
        x = rng.binomial(1, participant_x_propensity[i], trials_per_participant)
        x_between = float(np.mean(x))
        x_within = x - x_between
        mediator = (
            1.5
            + a_within * x_within
            + a_between * x_between
            + participant_intercept_m[i]
            + rng.normal(0, 0.6, trials_per_participant)
        )
        mediator = np.maximum(mediator, 0.02)
        m_between = float(np.mean(mediator))
        m_within = mediator - m_between
        eta = (
            -0.5
            + cprime_within * x_within
            + b_within * m_within
            + b_between * m_between
            + participant_intercept_y[i]
        )
        p = 1.0 / (1.0 + np.exp(-eta))
        outcome = rng.binomial(1, p)
        for j in range(trials_per_participant):
            rows.append(
                {
                    "participant_id": f"p{i + 1:03d}",
                    "trial_id": j + 1,
                    "ai_correct": int(x[j]),
                    "source_dwell": float(mediator[j]),
                    "correct_override": int(outcome[j]),
                    "valid_fraction": 1.0,
                }
            )
    return pd.DataFrame(rows)


# Serial and moderated extensions intentionally operate only on already-prepared
# canonical columns; decomposition stays in eyeprocess/eyeprocesspy.
def _select_analysis_rows(
    data: pd.DataFrame,
    *,
    required: Sequence[str],
    missingness_policy: str,
) -> tuple[pd.DataFrame, tuple[int, ...]]:
    if missingness_policy not in _MISSING_POLICIES:
        raise GP3BayesError(
            "`missingness_policy` must be one of: " + ", ".join(sorted(_MISSING_POLICIES)) + "."
        )
    complete = data[list(required)].notna().all(axis=1)
    eligible = data["mediation_analysis_eligible"].fillna(False).astype(bool)
    if missingness_policy == "error":
        keep = complete & eligible
        if not bool(keep.all()):
            n_problem = int((~keep).sum())
            raise GP3BayesError(
                f"{n_problem} trial row(s) are incomplete or quality-ineligible. "
                "Choose an explicit `missingness_policy` after reviewing the preparation audits."
            )
        keep = pd.Series(True, index=data.index)
    elif missingness_policy == "complete_case":
        keep = complete
    else:
        keep = complete & eligible
    excluded = tuple(np.flatnonzero(~keep.to_numpy()).astype(int).tolist())
    analysis = data.loc[keep].copy()
    if analysis.empty:
        raise GP3BayesError("No rows remain under the requested missingness policy.")
    return analysis, excluded


def specify_multilevel_serial_gaze_mediation(
    prepared: Any,
    *,
    mediator2_family: str = "gaussian",
    mediator_family: str = "gaussian",
    outcome_family: str = "bernoulli",
    priors: MediationPriorSpecification | None = None,
    random_slopes: Sequence[str] = (),
    missingness_policy: str = "error",
) -> MultilevelMediationSpecification:
    """Specify X -> M1 -> M2 -> Y using eyeprocess-prepared components.

    Serial models currently use participant random intercepts. Non-empty
    ``random_slopes`` are rejected explicitly rather than ignored.
    """
    data, columns, provenance = _prepared_contract(prepared)
    for key in ("mediator2", "mediator2_within", "mediator2_between"):
        if key not in columns:
            raise GP3BayesError(
                "Serial mediation requires eyeprocess-prepared `mediator2`, "
                "`mediator2_within`, and `mediator2_between` semantic columns."
            )
    slopes = tuple(random_slopes)
    if slopes:
        raise GP3BayesError(
            "Serial mediation random slopes are not yet implemented; pass `random_slopes=()` "
            "instead of requesting slopes that would be silently ignored."
        )
    m1_family = _normalize_family(mediator_family, _MEDIATOR_FAMILIES, "mediator_family")
    m2_family = _normalize_family(mediator2_family, _MEDIATOR_FAMILIES, "mediator2_family")
    y_family = _normalize_family(outcome_family, _OUTCOME_FAMILIES, "outcome_family")
    priors = priors or create_mediation_prior_specification()
    if not isinstance(priors, MediationPriorSpecification):
        raise GP3BayesError("`priors` must be a MediationPriorSpecification.")

    participant_col = str(columns["participant"])
    trial_col = str(columns["trial"])
    mediator_col = str(columns["mediator"])
    outcome_col = str(columns["outcome"])
    mediator2_col = str(columns["mediator2"])
    m2w = str(columns["mediator2_within"])
    m2b = str(columns["mediator2_between"])
    required = [
        participant_col,
        trial_col,
        mediator_col,
        mediator2_col,
        outcome_col,
        "X_within",
        "X_between",
        "M_within",
        "M_between",
        m2w,
        m2b,
    ]
    missing_cols = [c for c in required if c not in data]
    if missing_cols:
        raise GP3BayesError(
            "Prepared serial data is missing model columns: " + ", ".join(missing_cols) + "."
        )
    analysis, excluded = _select_analysis_rows(
        data, required=required, missingness_policy=missingness_policy
    )
    if analysis[participant_col].nunique() < 2:
        raise GP3BayesError(
            "At least two participants are required for serial multilevel mediation."
        )
    _validate_family_data(analysis, mediator_col, outcome_col, m1_family, y_family)
    temp = analysis.rename(columns={mediator2_col: "__m2__", outcome_col: "__y__"})
    _validate_family_data(temp, "__m2__", "__y__", m2_family, y_family)

    estimable: list[str] = []
    if _has_variation(analysis["X_within"]):
        estimable += ["a1_within", "a2_within", "cprime_within"]
    if _has_variation(analysis["X_between"]):
        estimable += ["a1_between", "a2_between", "cprime_between"]
    if _has_variation(analysis["M_within"]):
        estimable += ["d_within", "b1_within"]
    if _has_variation(analysis["M_between"]):
        estimable += ["d_between", "b1_between"]
    if _has_variation(analysis[m2w]):
        estimable.append("b2_within")
    if _has_variation(analysis[m2b]):
        estimable.append("b2_between")
    estimable_paths = tuple(estimable)
    required_serial = {"a1_within", "d_within", "b2_within"}
    if not required_serial.issubset(estimable_paths):
        raise GP3BayesError(
            "The within-participant serial indirect effect is not estimable from the observed design; "
            "missing path(s): " + ", ".join(sorted(required_serial - set(estimable_paths))) + "."
        )

    model_provenance = {
        "preparation": dict(provenance),
        "model_specification": {
            "model_kind": "serial",
            "mediator_family": m1_family,
            "mediator2_family": m2_family,
            "outcome_family": y_family,
            "random_slopes": [],
            "missingness_policy": missingness_policy,
            "priors": priors.as_dict(),
            "estimand_scale": "linear_predictor_product",
            "estimable_paths": list(estimable_paths),
            "random_effects_scope": "participant_random_intercepts",
        },
    }
    return MultilevelMediationSpecification(
        data=analysis,
        participant_col=participant_col,
        trial_col=trial_col,
        outcome_col=outcome_col,
        mediator_col=mediator_col,
        mediator_family=m1_family,
        outcome_family=y_family,
        priors=priors,
        random_slopes=(),
        missingness_policy=missingness_policy,
        input_rows=len(data),
        analysis_rows=len(analysis),
        excluded_row_positions=excluded,
        provenance=model_provenance,
        model_kind="serial",
        extra_columns={
            "mediator2": mediator2_col,
            "mediator2_within": m2w,
            "mediator2_between": m2b,
            "mediator2_family": m2_family,
        },
        serial=True,
        estimable_paths=estimable_paths,
    )


def _build_serial_pymc_model(spec: MultilevelMediationSpecification) -> Any:
    pm = _load_pymc()
    data = spec.data
    idx, levels = pd.factorize(data[spec.participant_col], sort=False)
    n_participants = len(levels)
    predictors = {
        "xw": data["X_within"].to_numpy(dtype=float),
        "xb": data["X_between"].to_numpy(dtype=float),
        "m1w": data["M_within"].to_numpy(dtype=float),
        "m1b": data["M_between"].to_numpy(dtype=float),
        "m2w": data[spec.extra_columns["mediator2_within"]].to_numpy(dtype=float),
        "m2b": data[spec.extra_columns["mediator2_between"]].to_numpy(dtype=float),
    }
    m1 = pd.to_numeric(data[spec.mediator_col], errors="raise").to_numpy(dtype=float)
    m2 = pd.to_numeric(data[spec.extra_columns["mediator2"]], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(data[spec.outcome_col], errors="raise").to_numpy(dtype=float)
    p = spec.priors
    estimable = set(spec.estimable_paths)
    with pm.Model() as model:
        path = {name: pm.Normal(name, 0, p.coefficient_sd) for name in estimable}
        eta_m1 = pm.Normal("alpha_m1", 0, p.intercept_sd) + path["a1_within"] * predictors["xw"]
        if "a1_between" in path:
            eta_m1 = eta_m1 + path["a1_between"] * predictors["xb"]
        eta_m1 += _random_effect(pm, "participant_m1", n_participants, idx, p.group_sd_scale)
        _observe_family(pm, "M1_obs", spec.mediator_family, eta_m1, m1, p)

        eta_m2 = (
            pm.Normal("alpha_m2", 0, p.intercept_sd)
            + path["a2_within"] * predictors["xw"]
            + path["d_within"] * predictors["m1w"]
        )
        for name, pred in [("a2_between", "xb"), ("d_between", "m1b")]:
            if name in path:
                eta_m2 = eta_m2 + path[name] * predictors[pred]
        eta_m2 += _random_effect(pm, "participant_m2", n_participants, idx, p.group_sd_scale)
        _observe_family(pm, "M2_obs", spec.extra_columns["mediator2_family"], eta_m2, m2, p)

        eta_y = (
            pm.Normal("alpha_y", 0, p.intercept_sd)
            + path["cprime_within"] * predictors["xw"]
            + path["b1_within"] * predictors["m1w"]
            + path["b2_within"] * predictors["m2w"]
        )
        for name, pred in [("cprime_between", "xb"), ("b1_between", "m1b"), ("b2_between", "m2b")]:
            if name in path:
                eta_y = eta_y + path[name] * predictors[pred]
        eta_y += _random_effect(pm, "participant_y", n_participants, idx, p.group_sd_scale)
        _observe_family(pm, "Y_obs", spec.outcome_family, eta_y, y, p)

        pm.Deterministic(
            "serial_indirect_within", path["a1_within"] * path["d_within"] * path["b2_within"]
        )
        if {"a1_between", "d_between", "b2_between"}.issubset(path):
            pm.Deterministic(
                "serial_indirect_between",
                path["a1_between"] * path["d_between"] * path["b2_between"],
            )
        pm.Deterministic("m1_indirect_within", path["a1_within"] * path["b1_within"])
        pm.Deterministic("m2_indirect_within", path["a2_within"] * path["b2_within"])
        total_ind = (
            path["a1_within"] * path["b1_within"]
            + path["a2_within"] * path["b2_within"]
            + path["a1_within"] * path["d_within"] * path["b2_within"]
        )
        pm.Deterministic("total_indirect_within", total_ind)
        pm.Deterministic("total_within", path["cprime_within"] + total_ind)
    return model


def fit_multilevel_serial_gaze_mediation(
    prepared: Any,
    *,
    mediator2_family: str = "gaussian",
    mediator_family: str = "gaussian",
    outcome_family: str = "bernoulli",
    priors: MediationPriorSpecification | None = None,
    random_slopes: Sequence[str] = (),
    missingness_policy: str = "error",
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    target_accept: float = 0.95,
    max_treedepth: int = 12,
) -> MultilevelMediationFit:
    """Fit X -> M1 -> M2 -> Y without re-decomposing variables in gp3bayespy."""
    spec = specify_multilevel_serial_gaze_mediation(
        prepared,
        mediator2_family=mediator2_family,
        mediator_family=mediator_family,
        outcome_family=outcome_family,
        priors=priors,
        random_slopes=random_slopes,
        missingness_policy=missingness_policy,
    )
    controls = _validate_sampling_controls(
        chains=chains,
        draws=draws,
        tune=tune,
        cores=cores,
        seed=seed,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
    )
    pm = _load_pymc()
    model = _build_serial_pymc_model(spec)
    with model:
        idata = pm.sample(
            draws=controls["draws"],
            tune=controls["tune"],
            chains=controls["chains"],
            cores=controls["cores"],
            random_seed=controls["seed"],
            target_accept=controls["target_accept"],
            nuts={"max_treedepth": controls["max_treedepth"]},
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    names = [
        "a1_within",
        "a1_between",
        "a2_within",
        "a2_between",
        "d_within",
        "d_between",
        "b1_within",
        "b1_between",
        "b2_within",
        "b2_between",
        "cprime_within",
        "cprime_between",
        "serial_indirect_within",
        "serial_indirect_between",
        "m1_indirect_within",
        "m2_indirect_within",
        "total_indirect_within",
        "total_within",
    ]
    fit = MultilevelMediationFit(
        specification=spec,
        posterior=_flatten_posterior(idata, names),
        sampler_diagnostics=_sampler_diagnostics(
            idata, max_treedepth=int(controls["max_treedepth"])
        ),
        backend_fit=idata,
        backend_model=model,
        package_versions={"pymc": _package_version("pymc"), "arviz": _package_version("arviz")},
    )
    if not check_mediation_convergence(fit).passed:
        warnings.warn(
            "Serial mediation fit failed critical convergence checks.", RuntimeWarning, stacklevel=2
        )
    return fit


def posterior_serial_indirect_effect(
    fit: MultilevelMediationFit,
    *,
    level: str = "within",
    probability: float = 0.95,
    require_convergence: bool = True,
) -> MediationEffect:
    if fit.specification.model_kind != "serial":
        raise GP3BayesError("`fit` is not a serial mediation model.")
    _require_acceptable_fit(fit, require_convergence)
    if level not in {"within", "between"}:
        raise GP3BayesError("`level` must be `within` or `between`.")
    key = f"serial_indirect_{level}"
    if key not in fit.posterior:
        raise GP3BayesError(f"Posterior draws `{key}` are unavailable.")
    return _effect_from_draws(key, fit.posterior[key], probability, "linear_predictor_product")


def specify_multilevel_moderated_gaze_mediation(
    prepared: Any,
    *,
    moderation_path: str = "a",
    moderator_component: str = "within",
    mediator_family: str = "gaussian",
    outcome_family: str = "bernoulli",
    priors: MediationPriorSpecification | None = None,
    random_slopes: Sequence[str] = (),
    missingness_policy: str = "error",
) -> MultilevelMediationSpecification:
    """Specify moderation of the within a or b path with a prepared moderator.

    Moderated extensions currently use participant random intercepts. Requested
    random slopes are rejected explicitly until the extension implements them.
    """
    data, columns, _ = _prepared_contract(prepared)
    if moderation_path not in {"a", "b"}:
        raise GP3BayesError("`moderation_path` must be `a` or `b`.")
    if moderator_component not in {"within", "between"}:
        raise GP3BayesError("`moderator_component` must be `within` or `between`.")
    if tuple(random_slopes):
        raise GP3BayesError(
            "Moderated mediation random slopes are not yet implemented; pass `random_slopes=()` "
            "instead of requesting slopes that would be silently ignored."
        )
    key = f"moderator_{moderator_component}"
    if key not in columns:
        raise GP3BayesError(f"Moderated mediation requires eyeprocess-prepared `{key}` semantics.")
    base = specify_multilevel_gaze_mediation(
        prepared,
        mediator_family=mediator_family,
        outcome_family=outcome_family,
        priors=priors,
        random_slopes=(),
        missingness_policy=missingness_policy,
    )
    zcol = str(columns[key])
    if zcol not in data:
        raise GP3BayesError(f"Prepared moderator column `{zcol}` is missing.")
    z_missing = data[zcol].isna()
    if bool(z_missing.any()) and missingness_policy == "error":
        raise GP3BayesError(
            "Moderator contains missing values; choose an explicit missingness policy."
        )
    analysis = base.data.loc[base.data[zcol].notna()].copy()
    excluded_extra = tuple(np.flatnonzero(z_missing.to_numpy()).astype(int).tolist())
    excluded = tuple(sorted(set(base.excluded_row_positions) | set(excluded_extra)))
    if analysis.empty:
        raise GP3BayesError("No rows remain after applying moderator missingness rules.")
    if not _has_variation(analysis[zcol]):
        raise GP3BayesError("The selected moderator component has no observed variation.")
    interaction = (
        analysis["X_within"] * pd.to_numeric(analysis[zcol], errors="raise")
        if moderation_path == "a"
        else analysis["M_within"] * pd.to_numeric(analysis[zcol], errors="raise")
    )
    if not _has_variation(interaction):
        raise GP3BayesError("The requested moderated path interaction has no observed variation.")
    estimable_paths = _estimable_simple_paths(analysis)
    if not {"a_within", "b_within"}.issubset(estimable_paths):
        raise GP3BayesError(
            "The within-participant indirect effect is not estimable after moderator filtering."
        )
    provenance2 = dict(base.provenance)
    provenance2["model_specification"] = dict(provenance2["model_specification"])
    provenance2["model_specification"].update(
        {
            "model_kind": "moderated",
            "moderation_path": moderation_path,
            "moderator_component": moderator_component,
            "random_effects_scope": "participant_random_intercepts",
            "estimable_paths": list(estimable_paths),
        }
    )
    return MultilevelMediationSpecification(
        data=analysis,
        participant_col=base.participant_col,
        trial_col=base.trial_col,
        outcome_col=base.outcome_col,
        mediator_col=base.mediator_col,
        mediator_family=base.mediator_family,
        outcome_family=base.outcome_family,
        priors=base.priors,
        random_slopes=(),
        missingness_policy=base.missingness_policy,
        input_rows=base.input_rows,
        analysis_rows=len(analysis),
        excluded_row_positions=excluded,
        provenance=provenance2,
        model_kind="moderated",
        extra_columns={"moderator": zcol},
        moderation={"path": moderation_path, "component": moderator_component},
        moderated=True,
        estimable_paths=estimable_paths,
    )


def _build_moderated_pymc_model(spec: MultilevelMediationSpecification) -> Any:
    pm = _load_pymc()
    data = spec.data
    idx, levels = pd.factorize(data[spec.participant_col], sort=False)
    n_participants = len(levels)
    xw = data["X_within"].to_numpy(dtype=float)
    xb = data["X_between"].to_numpy(dtype=float)
    mw = data["M_within"].to_numpy(dtype=float)
    mb = data["M_between"].to_numpy(dtype=float)
    z = pd.to_numeric(data[spec.extra_columns["moderator"]], errors="raise").to_numpy(dtype=float)
    m = pd.to_numeric(data[spec.mediator_col], errors="raise").to_numpy(dtype=float)
    y = pd.to_numeric(data[spec.outcome_col], errors="raise").to_numpy(dtype=float)
    p = spec.priors
    estimable = set(spec.estimable_paths)
    with pm.Model() as model:
        path = {name: pm.Normal(name, 0, p.coefficient_sd) for name in estimable}
        moderation = pm.Normal("moderation", 0, p.coefficient_sd)
        eta_m = pm.Normal("alpha_m", 0, p.intercept_sd) + path["a_within"] * xw
        if "a_between" in path:
            eta_m = eta_m + path["a_between"] * xb
        if spec.moderation["path"] == "a":
            eta_m = eta_m + moderation * xw * z
        eta_m += _random_effect(pm, "participant_m", n_participants, idx, p.group_sd_scale)
        _observe_family(pm, "M_obs", spec.mediator_family, eta_m, m, p)

        eta_y = (
            pm.Normal("alpha_y", 0, p.intercept_sd)
            + path["cprime_within"] * xw
            + path["b_within"] * mw
        )
        for name, predictor in [("cprime_between", xb), ("b_between", mb)]:
            if name in path:
                eta_y = eta_y + path[name] * predictor
        if spec.moderation["path"] == "b":
            eta_y = eta_y + moderation * mw * z
        eta_y += _random_effect(pm, "participant_y", n_participants, idx, p.group_sd_scale)
        _observe_family(pm, "Y_obs", spec.outcome_family, eta_y, y, p)
        pm.Deterministic("indirect_within", path["a_within"] * path["b_within"])
        if {"a_between", "b_between"}.issubset(path):
            pm.Deterministic("indirect_between", path["a_between"] * path["b_between"])
    return model


def fit_multilevel_moderated_gaze_mediation(
    prepared: Any,
    *,
    moderation_path: str = "a",
    moderator_component: str = "within",
    mediator_family: str = "gaussian",
    outcome_family: str = "bernoulli",
    priors: MediationPriorSpecification | None = None,
    random_slopes: Sequence[str] = (),
    missingness_policy: str = "error",
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    cores: int | None = None,
    seed: int = 1,
    target_accept: float = 0.95,
    max_treedepth: int = 12,
) -> MultilevelMediationFit:
    """Fit moderation of the within a or b path using an explicitly prepared moderator."""
    spec = specify_multilevel_moderated_gaze_mediation(
        prepared,
        moderation_path=moderation_path,
        moderator_component=moderator_component,
        mediator_family=mediator_family,
        outcome_family=outcome_family,
        priors=priors,
        random_slopes=random_slopes,
        missingness_policy=missingness_policy,
    )
    controls = _validate_sampling_controls(
        chains=chains,
        draws=draws,
        tune=tune,
        cores=cores,
        seed=seed,
        target_accept=target_accept,
        max_treedepth=max_treedepth,
    )
    pm = _load_pymc()
    model = _build_moderated_pymc_model(spec)
    with model:
        idata = pm.sample(
            draws=controls["draws"],
            tune=controls["tune"],
            chains=controls["chains"],
            cores=controls["cores"],
            random_seed=controls["seed"],
            target_accept=controls["target_accept"],
            nuts={"max_treedepth": controls["max_treedepth"]},
            progressbar=False,
            compute_convergence_checks=False,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )
    names = [
        "a_within",
        "a_between",
        "b_within",
        "b_between",
        "cprime_within",
        "cprime_between",
        "moderation",
        "indirect_within",
        "indirect_between",
    ]
    fit = MultilevelMediationFit(
        specification=spec,
        posterior=_flatten_posterior(idata, names),
        sampler_diagnostics=_sampler_diagnostics(
            idata, max_treedepth=int(controls["max_treedepth"])
        ),
        backend_fit=idata,
        backend_model=model,
        package_versions={"pymc": _package_version("pymc"), "arviz": _package_version("arviz")},
    )
    if not check_mediation_convergence(fit).passed:
        warnings.warn(
            "Moderated mediation fit failed critical convergence checks.",
            RuntimeWarning,
            stacklevel=2,
        )
    return fit


def posterior_conditional_indirect_effect(
    fit: MultilevelMediationFit,
    *,
    moderator_value: float,
    probability: float = 0.95,
    require_convergence: bool = True,
) -> MediationEffect:
    """Compute the within-person indirect effect at a specified moderator value."""
    if fit.specification.model_kind != "moderated":
        raise GP3BayesError("`fit` is not a moderated mediation model.")
    _require_acceptable_fit(fit, require_convergence)
    if (
        isinstance(moderator_value, bool)
        or not isinstance(moderator_value, (int, float))
        or not math.isfinite(float(moderator_value))
    ):
        raise GP3BayesError("`moderator_value` must be one finite numeric value.")
    z = float(moderator_value)
    raw_a = fit.posterior.get("a_within")
    raw_b = fit.posterior.get("b_within")
    raw_mod = fit.posterior.get("moderation")
    if raw_a is None or raw_b is None or raw_mod is None:
        raise GP3BayesError("Required moderated path posterior draws are unavailable.")
    a = np.asarray(raw_a)
    b = np.asarray(raw_b)
    mod = np.asarray(raw_mod)
    draws = (a + mod * z) * b if fit.specification.moderation["path"] == "a" else a * (b + mod * z)
    return _effect_from_draws(
        f"conditional_indirect_within_z_{z:g}", draws, probability, "linear_predictor_product"
    )


__all__ = [
    "MediationPriorSpecification",
    "MultilevelMediationSpecification",
    "MultilevelMediationFit",
    "MediationConvergence",
    "MediationEffect",
    "create_mediation_prior_specification",
    "specify_multilevel_gaze_mediation",
    "fit_multilevel_gaze_mediation",
    "specify_multilevel_serial_gaze_mediation",
    "fit_multilevel_serial_gaze_mediation",
    "posterior_serial_indirect_effect",
    "specify_multilevel_moderated_gaze_mediation",
    "fit_multilevel_moderated_gaze_mediation",
    "posterior_conditional_indirect_effect",
    "estimate_indirect_effect",
    "estimate_within_indirect_effect",
    "estimate_between_indirect_effect",
    "posterior_indirect_effect",
    "posterior_direct_effect",
    "posterior_total_effect",
    "summarise_multilevel_mediation",
    "compare_multilevel_mediation_models",
    "check_mediation_convergence",
    "posterior_predictive_check_mediation",
    "prior_predictive_check_mediation",
    "plot_mediation_posteriors",
    "plot_indirect_effect_distribution",
    "plot_participant_mediation_effects",
    "report_multilevel_gaze_mediation",
    "simulate_multilevel_gaze_mediation",
]
