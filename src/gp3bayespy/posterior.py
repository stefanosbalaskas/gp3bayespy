"""Shared posterior draw, summary, and sampling-diagnostic infrastructure.

The frozen R reference uses the posterior package over brms/rstan fits.  The
Python port preserves the same restricted posterior contracts while adapting
storage and numerical diagnostics to PyMC InferenceData and ArviZ.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import numpy as np
import pandas as pd

from .exceptions import BackendUnavailableError, GP3BayesError


@dataclass(frozen=True, slots=True)
class _SamplingDiagnosticsResult:
    diagnostic_version: str
    family: str
    status: str
    component_table: pd.DataFrame
    parameter_table: pd.DataFrame
    chain_table: pd.DataFrame
    thresholds: Mapping[str, float]
    diagnostics_assessed: bool = True
    convergence_claim: bool = False
    posterior_adequacy_established: bool = False
    interpretation: str = (
        "The status reports whether prespecified numerical sampling thresholds "
        "were met. It is not an automatic declaration of convergence or posterior adequacy."
    )

    def __repr__(self) -> str:
        return "\n".join(
            [
                "<gp3bayes_sampling_diagnostics>",
                f"  Family: {self.family}",
                f"  Threshold status: {self.status}",
                "  Diagnostics assessed: TRUE",
                "  Automatic convergence claim: FALSE",
                "  Posterior adequacy established: FALSE",
            ]
        )


@dataclass(frozen=True, slots=True)
class _PosteriorSummaryResult:
    summary_version: str
    family: str
    probability: float
    table: pd.DataFrame
    interpretation_scale: Mapping[str, str]
    outcome_unit: str | None = None
    posterior_summarised: bool = True
    convergence_claim: bool = False
    posterior_adequacy_established: bool = False

    def __repr__(self) -> str:
        lines = [
            f"<gp3bayes_{self.family}_posterior_summary>",
            f"  Probability: {self.probability:g}",
            f"  Parameters: {len(self.table)}",
            "  Posterior summarised: TRUE",
            "  Automatic convergence claim: FALSE",
            "  Posterior adequacy established: FALSE",
        ]
        if self.outcome_unit is not None:
            lines.insert(2, f"  Outcome unit: {self.outcome_unit}")
        return "\n".join(lines)


def _numeric_scalar(
    value: object,
    name: str,
    *,
    lower: float = -math.inf,
    upper: float = math.inf,
    lower_open: bool = False,
    upper_open: bool = False,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Real)
        or not math.isfinite(float(value))
    ):
        raise GP3BayesError(f"`{name}` must be one finite numeric value.")
    number = float(cast(Any, value))
    lower_ok = number > lower if lower_open else number >= lower
    upper_ok = number < upper if upper_open else number <= upper
    if not lower_ok or not upper_ok:
        left = "(" if lower_open else "["
        right = ")" if upper_open else "]"
        raise GP3BayesError(
            f"`{name}` must lie in {left}{lower:g}, {upper:g}{right}."
        )
    return number


def _validate_probability(value: object, name: str, *, open: bool = False) -> float:
    return _numeric_scalar(
        value,
        name,
        lower=0,
        upper=1,
        lower_open=open,
        upper_open=open,
    )


def _validate_fit_like(fit: object, family: str | None = None) -> Any:
    if (
        getattr(fit, "fit_performed", False) is not True
        or getattr(fit, "backend_fit", None) is None
    ):
        raise GP3BayesError("`fit` must inherit from `gp3bayes_fit`.")
    observed_family = getattr(fit, "family", None)
    if observed_family not in {"binary", "duration"}:
        raise GP3BayesError("The fit does not use an approved gp3bayes family.")
    if family is not None and observed_family != family:
        raise GP3BayesError(f"`fit` must use the approved `{family}` family.")
    validated = cast(Any, fit)
    backend_fit = validated.backend_fit
    if getattr(backend_fit, "posterior", None) is None:
        raise GP3BayesError("`fit.backend_fit` must contain posterior draws.")
    return validated


def _array_values(value: Any) -> np.ndarray:
    if hasattr(value, "values"):
        return np.asarray(value.values, dtype=float)
    return np.asarray(value, dtype=float)


def _posterior_data_vars(fit: Any) -> Mapping[str, Any]:
    posterior = fit.backend_fit.posterior
    data_vars = getattr(posterior, "data_vars", None)
    if data_vars is not None:
        return cast(Mapping[str, Any], data_vars)
    if isinstance(posterior, Mapping):
        return cast(Mapping[str, Any], posterior)
    raise GP3BayesError("The fitted posterior uses an unsupported storage structure.")


def _prepared_model_columns(fit: Any) -> tuple[str, ...]:
    specification = getattr(fit, "specification", None)
    prepared = getattr(specification, "prepared", None)
    columns = getattr(prepared, "model_matrix_columns", ())
    return tuple(str(value) for value in columns)


def _mapping_name(fit: Any, key: str) -> str | None:
    specification = getattr(fit, "specification", None)
    contract = getattr(specification, "contract", None)
    if contract is None:
        prepared = getattr(specification, "prepared", None)
        contract = getattr(prepared, "contract", None)
    mappings = getattr(contract, "mappings", {}) if contract is not None else {}
    value = mappings.get(key) if isinstance(mappings, Mapping) else None
    return None if value is None else str(value)


def _component_name(raw: str, index: tuple[int, ...]) -> str:
    one_based = ",".join(str(value + 1) for value in index)
    return f"{raw}[{one_based}]"


def _posterior_components(fit: Any) -> dict[str, np.ndarray]:
    fit = _validate_fit_like(fit)
    data_vars = _posterior_data_vars(fit)
    components: dict[str, np.ndarray] = {}
    participant = _mapping_name(fit, "participant") or "participant"
    item = _mapping_name(fit, "item")
    condition = _mapping_name(fit, "condition") or "condition"
    fixed_columns = _prepared_model_columns(fit)
    beta_names = fixed_columns[1:] if len(fixed_columns) > 1 else ()

    for raw, value in data_vars.items():
        array = _array_values(value)
        if array.ndim < 2:
            raise GP3BayesError(
                f"Posterior variable `{raw}` must retain chain and draw dimensions."
            )
        extra_shape = array.shape[2:]

        if raw == "b" and len(extra_shape) == 1 and len(beta_names) == extra_shape[0]:
            for index, term in enumerate(beta_names):
                components[f"b_{term}"] = array[:, :, index]
            continue
        if raw == "sd_participant" and not extra_shape:
            components[f"sd_{participant}__Intercept"] = array
            continue
        if raw == "sd_item" and not extra_shape and item is not None:
            components[f"sd_{item}__Intercept"] = array
            continue
        if raw == "participant_chol_stds" and extra_shape == (2,):
            components[f"sd_{participant}__Intercept"] = array[:, :, 0]
            components[f"sd_{participant}__{condition}"] = array[:, :, 1]
            continue
        if raw == "participant_chol_corr" and extra_shape == (2, 2):
            components[f"cor_{participant}__Intercept__{condition}"] = array[:, :, 0, 1]
            continue

        if not extra_shape:
            components[str(raw)] = array
            continue
        flattened_extra = array.reshape(array.shape[0], array.shape[1], -1)
        for flat_index in range(flattened_extra.shape[2]):
            remainder = flat_index
            reverse_index: list[int] = []
            for size in reversed(extra_shape):
                size_int = int(size)
                reverse_index.append(remainder % size_int)
                remainder //= size_int
            index = tuple(reversed(reverse_index))
            components[_component_name(str(raw), index)] = flattened_extra[:, :, flat_index]

    if not components:
        raise GP3BayesError("No posterior variables were found.")
    return components


def _select_components(
    fit: Any,
    *,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    parameters_only: bool = False,
) -> dict[str, np.ndarray]:
    components = _posterior_components(fit)
    if parameters_only:
        components = {
            name: values
            for name, values in components.items()
            if name.startswith(("b_", "sd_", "cor_")) or name == "sigma"
        }
        if not components:
            raise GP3BayesError("No supported posterior parameters were found.")

    selected_names = list(components)
    if variables is not None:
        requested = [variables] if isinstance(variables, str) else list(variables)
        if not requested or any(not isinstance(value, str) or not value for value in requested):
            raise GP3BayesError("`variables` must be a non-empty character vector.")
        missing = [value for value in requested if value not in components]
        if missing:
            if parameters_only:
                raise GP3BayesError(
                    "Requested posterior variables were not found: "
                    + ", ".join(missing)
                    + "."
                )
            raise GP3BayesError("Unknown posterior variables: " + ", ".join(missing) + ".")
        requested_set = set(requested)
        selected_names = [name for name in selected_names if name in requested_set]

    if regex is not None:
        if not isinstance(regex, str):
            raise GP3BayesError("`regex` must be one non-missing character string.")
        import re

        try:
            pattern = re.compile(regex)
        except re.error as error:
            raise GP3BayesError(f"Invalid `regex`: {error}.") from error
        selected_names = [name for name in selected_names if pattern.search(name)]

    if not selected_names:
        raise GP3BayesError("No posterior variables remain after selection.")
    return {name: components[name] for name in selected_names}


def extract_draws(
    fit: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    format: str = "array",
) -> Any:
    if format not in {"array", "matrix", "df", "rvars"}:
        raise GP3BayesError("`format` must be one of: array, matrix, df, rvars.")
    selected = _select_components(fit, variables=variables, regex=regex)
    names = list(selected)
    arrays = [np.asarray(selected[name], dtype=float) for name in names]
    shape = arrays[0].shape
    if any(values.shape != shape or values.ndim != 2 for values in arrays):
        raise GP3BayesError("Posterior components must share chain and draw dimensions.")
    chains, draws = shape
    cube = np.stack(arrays, axis=2)

    if format == "array":
        try:
            xr = import_module("xarray")
        except Exception as error:
            raise BackendUnavailableError(
                "Optional package `xarray` is required for `format='array'`."
            ) from error
        return xr.DataArray(
            cube,
            dims=("chain", "draw", "variable"),
            coords={
                "chain": np.arange(1, chains + 1),
                "draw": np.arange(1, draws + 1),
                "variable": names,
            },
            name="posterior_draws",
        )
    rows = cube.reshape(chains * draws, len(names))
    if format == "matrix":
        return pd.DataFrame(rows, columns=names)
    if format == "rvars":
        return {name: selected[name].copy() for name in names}

    frame = pd.DataFrame(rows, columns=names)
    frame[".chain"] = np.repeat(np.arange(1, chains + 1), draws)
    frame[".iteration"] = np.tile(np.arange(1, draws + 1), chains)
    frame[".draw"] = np.arange(1, chains * draws + 1)
    return frame


def _arviz() -> Any:
    try:
        return import_module("arviz")
    except Exception as error:
        raise BackendUnavailableError(
            "Optional package `arviz` is required for posterior diagnostics and summaries."
        ) from error


def _dataset_scalar(dataset: Any, name: str) -> float:
    try:
        value = dataset[name]
    except Exception:
        return math.nan
    array = _array_values(value).reshape(-1)
    return math.nan if array.size == 0 else float(array[0])


def _diagnostic_metrics(selected: Mapping[str, np.ndarray]) -> pd.DataFrame:
    az = _arviz()
    posterior = {name: np.asarray(values, dtype=float) for name, values in selected.items()}
    rhat = az.rhat(posterior, method="rank", chain_axis=0, draw_axis=1)
    ess_bulk = az.ess(
        posterior, method="bulk", relative=False, chain_axis=0, draw_axis=1
    )
    ess_tail = az.ess(
        posterior, method="tail", relative=False, chain_axis=0, draw_axis=1
    )
    return pd.DataFrame(
        {
            "variable": list(selected),
            "rhat": [_dataset_scalar(rhat, name) for name in selected],
            "ess_bulk": [_dataset_scalar(ess_bulk, name) for name in selected],
            "ess_tail": [_dataset_scalar(ess_tail, name) for name in selected],
        }
    )


def _classify_upper(value: float, pass_value: float, review: float) -> str:
    if not math.isfinite(value):
        return "not_assessed"
    if value <= pass_value:
        return "pass"
    if value <= review:
        return "review"
    return "fail"


def _classify_lower(value: float, pass_value: float, review: float) -> str:
    if not math.isfinite(value):
        return "not_assessed"
    if value >= pass_value:
        return "pass"
    if value >= review:
        return "review"
    return "fail"


def _worst_status(statuses: Sequence[str]) -> str:
    values = [value for value in statuses if value]
    if not values:
        return "review"
    if "fail" in values:
        return "fail"
    if any(value in {"review", "not_assessed", "not_applicable"} for value in values):
        return "review"
    return "pass"


def _sample_stats_array(sample_stats: Any, names: Sequence[str]) -> np.ndarray | None:
    data_vars = getattr(sample_stats, "data_vars", None)
    mapping: Mapping[str, Any]
    if data_vars is not None:
        mapping = cast(Mapping[str, Any], data_vars)
    elif isinstance(sample_stats, Mapping):
        mapping = cast(Mapping[str, Any], sample_stats)
    else:
        return None
    for name in names:
        if name in mapping:
            array = _array_values(mapping[name])
            if array.ndim >= 2:
                return array
    return None


def _chain_table(fit: Any, max_treedepth: int) -> pd.DataFrame:
    sample_stats = getattr(fit.backend_fit, "sample_stats", None)
    posterior_components = _posterior_components(fit)
    first = next(iter(posterior_components.values()))
    chains, draws = first.shape
    divergence = _sample_stats_array(sample_stats, ("diverging", "divergence"))
    tree_depth = _sample_stats_array(sample_stats, ("tree_depth", "treedepth"))
    energy = _sample_stats_array(sample_stats, ("energy",))

    rows: list[dict[str, int | float]] = []
    for chain in range(chains):
        div_values = None if divergence is None else divergence[chain]
        tree_values = None if tree_depth is None else tree_depth[chain]
        energy_values = None if energy is None else energy[chain]
        ebfmi = math.nan
        if energy_values is not None and len(energy_values) >= 3:
            variance = float(np.var(energy_values, ddof=1))
            if variance > 0 and math.isfinite(variance):
                ebfmi = float(np.mean(np.diff(energy_values) ** 2) / variance)
        rows.append(
            {
                "chain": chain + 1,
                "iterations": draws,
                "divergences": (
                    math.nan if div_values is None else int(np.sum(div_values > 0))
                ),
                "treedepth_hits": (
                    math.nan
                    if tree_values is None
                    else int(np.sum(tree_values >= max_treedepth))
                ),
                "treedepth_hit_fraction": (
                    math.nan
                    if tree_values is None
                    else float(np.mean(tree_values >= max_treedepth))
                ),
                "ebfmi": ebfmi,
            }
        )
    return pd.DataFrame(rows)


def diagnose_fit(
    fit: Any,
    *,
    family: str,
    rhat_pass: float = 1.01,
    rhat_fail: float = 1.05,
    ess_per_chain_pass: float = 100,
    ess_per_chain_fail: float = 50,
    maximum_treedepth_fraction: float = 0.01,
    ebfmi_pass: float = 0.30,
    ebfmi_fail: float = 0.20,
) -> _SamplingDiagnosticsResult:
    fit = _validate_fit_like(fit, family=family)
    rhat_pass = _numeric_scalar(rhat_pass, "rhat_pass", lower=1)
    rhat_fail = _numeric_scalar(rhat_fail, "rhat_fail", lower=rhat_pass)
    ess_per_chain_pass = _numeric_scalar(
        ess_per_chain_pass, "ess_per_chain_pass", lower=1
    )
    ess_per_chain_fail = _numeric_scalar(
        ess_per_chain_fail,
        "ess_per_chain_fail",
        lower=1,
        upper=ess_per_chain_pass,
    )
    maximum_treedepth_fraction = _validate_probability(
        maximum_treedepth_fraction, "maximum_treedepth_fraction"
    )
    ebfmi_pass = _numeric_scalar(ebfmi_pass, "ebfmi_pass", lower=0)
    ebfmi_fail = _numeric_scalar(ebfmi_fail, "ebfmi_fail", lower=0, upper=ebfmi_pass)

    selected = _select_components(fit, parameters_only=True)
    parameter_table = _diagnostic_metrics(selected)
    chain_count = next(iter(selected.values())).shape[0]
    parameter_table["ess_bulk_per_chain"] = parameter_table["ess_bulk"] / chain_count
    parameter_table["ess_tail_per_chain"] = parameter_table["ess_tail"] / chain_count
    parameter_table["rhat_status"] = [
        _classify_upper(float(value), rhat_pass, rhat_fail)
        for value in parameter_table["rhat"]
    ]
    parameter_table["ess_bulk_status"] = [
        _classify_lower(float(value), ess_per_chain_pass, ess_per_chain_fail)
        for value in parameter_table["ess_bulk_per_chain"]
    ]
    parameter_table["ess_tail_status"] = [
        _classify_lower(float(value), ess_per_chain_pass, ess_per_chain_fail)
        for value in parameter_table["ess_tail_per_chain"]
    ]

    max_tree = int(getattr(fit, "sampling", {}).get("max_treedepth", 10))
    chain_table = _chain_table(fit, max_tree)

    finite_rhat = parameter_table["rhat"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_bulk = parameter_table["ess_bulk_per_chain"].replace([np.inf, -np.inf], np.nan).dropna()
    finite_tail = parameter_table["ess_tail_per_chain"].replace([np.inf, -np.inf], np.nan).dropna()
    max_rhat = math.nan if finite_rhat.empty else float(finite_rhat.max())
    min_bulk = math.nan if finite_bulk.empty else float(finite_bulk.min())
    min_tail = math.nan if finite_tail.empty else float(finite_tail.min())

    divergences = pd.to_numeric(chain_table["divergences"], errors="coerce")
    tree_fraction = pd.to_numeric(chain_table["treedepth_hit_fraction"], errors="coerce")
    ebfmi = pd.to_numeric(chain_table["ebfmi"], errors="coerce")
    total_divergences = math.nan if divergences.isna().all() else float(divergences.sum())
    maximum_tree_fraction = math.nan if tree_fraction.isna().all() else float(tree_fraction.max())
    minimum_ebfmi = math.nan if ebfmi.isna().all() else float(ebfmi.min())

    observed = [
        max_rhat,
        min_bulk,
        min_tail,
        total_divergences,
        maximum_tree_fraction,
        minimum_ebfmi,
    ]
    statuses = [
        _classify_upper(max_rhat, rhat_pass, rhat_fail),
        _classify_lower(min_bulk, ess_per_chain_pass, ess_per_chain_fail),
        _classify_lower(min_tail, ess_per_chain_pass, ess_per_chain_fail),
        (
            "not_assessed"
            if not math.isfinite(total_divergences)
            else "pass" if total_divergences == 0 else "fail"
        ),
        (
            "not_assessed"
            if not math.isfinite(maximum_tree_fraction)
            else "pass"
            if maximum_tree_fraction == 0
            else "review"
            if maximum_tree_fraction <= maximum_treedepth_fraction
            else "fail"
        ),
        _classify_lower(minimum_ebfmi, ebfmi_pass, ebfmi_fail),
    ]
    component_table = pd.DataFrame(
        {
            "component": [
                "rhat",
                "bulk_ess_per_chain",
                "tail_ess_per_chain",
                "divergences",
                "treedepth_saturation",
                "energy_ebfmi",
            ],
            "observed": observed,
            "pass_rule": [
                f"<={rhat_pass:g}",
                f">={ess_per_chain_pass:g}",
                f">={ess_per_chain_pass:g}",
                "0",
                "0",
                f">={ebfmi_pass:g}",
            ],
            "review_rule": [
                f">{rhat_pass:g} and <={rhat_fail:g}",
                f">={ess_per_chain_fail:g} and <{ess_per_chain_pass:g}",
                f">={ess_per_chain_fail:g} and <{ess_per_chain_pass:g}",
                "not used",
                f">0 and <={maximum_treedepth_fraction:g}",
                f">={ebfmi_fail:g} and <{ebfmi_pass:g}",
            ],
            "status": statuses,
        }
    )
    return _SamplingDiagnosticsResult(
        diagnostic_version="0.1",
        family=family,
        status=_worst_status(statuses),
        component_table=component_table,
        parameter_table=parameter_table,
        chain_table=chain_table,
        thresholds={
            "rhat_pass": rhat_pass,
            "rhat_fail": rhat_fail,
            "ess_per_chain_pass": ess_per_chain_pass,
            "ess_per_chain_fail": ess_per_chain_fail,
            "maximum_treedepth_fraction": maximum_treedepth_fraction,
            "ebfmi_pass": ebfmi_pass,
            "ebfmi_fail": ebfmi_fail,
        },
    )


def _summary_table(
    fit: Any,
    probability: float,
    variables: Sequence[str] | str | None,
) -> pd.DataFrame:
    probability = _validate_probability(probability, "probability", open=True)
    selected = _select_components(fit, variables=variables, parameters_only=True)
    metrics = _diagnostic_metrics(selected).set_index("variable")
    alpha = 1 - probability
    rows: list[dict[str, float | str]] = []
    for name, values in selected.items():
        flattened = np.asarray(values, dtype=float).reshape(-1)
        rows.append(
            {
                "variable": name,
                "mean": float(np.mean(flattened)),
                "median": float(np.median(flattened)),
                "sd": float(np.std(flattened, ddof=1)),
                "lower": float(
                    np.quantile(flattened, alpha / 2, method="median_unbiased")
                ),
                "upper": float(
                    np.quantile(flattened, 1 - alpha / 2, method="median_unbiased")
                ),
                "probability_positive": float(np.mean(flattened > 0)),
                "rhat": float(cast(Any, metrics.loc[name, "rhat"])),
                "ess_bulk": float(cast(Any, metrics.loc[name, "ess_bulk"])),
                "ess_tail": float(cast(Any, metrics.loc[name, "ess_tail"])),
            }
        )
    return pd.DataFrame(rows)


def summarise_binary(
    fit: Any,
    probability: float = 0.95,
    variables: Sequence[str] | str | None = None,
) -> _PosteriorSummaryResult:
    fit = _validate_fit_like(fit, family="binary")
    probability = _validate_probability(probability, "probability", open=True)
    table = _summary_table(fit, probability, variables)
    population = table["variable"].str.startswith("b_")
    table["odds_ratio_median"] = np.nan
    table["odds_ratio_lower"] = np.nan
    table["odds_ratio_upper"] = np.nan
    table.loc[population, "odds_ratio_median"] = np.exp(table.loc[population, "median"])
    table.loc[population, "odds_ratio_lower"] = np.exp(table.loc[population, "lower"])
    table.loc[population, "odds_ratio_upper"] = np.exp(table.loc[population, "upper"])
    return _PosteriorSummaryResult(
        summary_version="0.1",
        family="binary",
        probability=probability,
        table=table,
        interpretation_scale={
            "population_coefficients": "log-odds and odds ratio",
            "group_standard_deviations": "log-odds standard deviation",
            "correlations": "correlation",
        },
    )


def summarise_duration(
    fit: Any,
    probability: float = 0.95,
    variables: Sequence[str] | str | None = None,
) -> _PosteriorSummaryResult:
    fit = _validate_fit_like(fit, family="duration")
    probability = _validate_probability(probability, "probability", open=True)
    table = _summary_table(fit, probability, variables)
    population = table["variable"].str.startswith("b_")
    table["median_ratio"] = np.nan
    table["ratio_lower"] = np.nan
    table["ratio_upper"] = np.nan
    table.loc[population, "median_ratio"] = np.exp(table.loc[population, "median"])
    table.loc[population, "ratio_lower"] = np.exp(table.loc[population, "lower"])
    table.loc[population, "ratio_upper"] = np.exp(table.loc[population, "upper"])
    return _PosteriorSummaryResult(
        summary_version="0.1",
        family="duration",
        outcome_unit=str(getattr(fit, "outcome_unit", "unknown")),
        probability=probability,
        table=table,
        interpretation_scale={
            "population_coefficients": "log duration and median ratio",
            "group_standard_deviations": "log-duration standard deviation",
            "sigma": "lognormal residual standard deviation",
            "correlations": "correlation",
        },
    )
