"""Backend-neutral posterior exploration and diagnostic tables.

The module preserves gp3bayes 0.5.0's conservative interpretation: numerical
MCMC/LOO flags request review and never establish convergence, adequacy, or
causal validity automatically.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .exceptions import GP3BayesError
from .posterior import _posterior_components, extract_draws


def extract_posterior_draws(
    fit: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    format: str = "array",
) -> Any:
    """Extract posterior draws from an approved gp3bayespy fit."""
    aliases = {
        "dataframe": "df",
        "df": "df",
        "matrix": "matrix",
        "rvars": "rvars",
        "array": "array",
    }
    fmt = aliases.get(format, format)
    return extract_draws(fit, variables=variables, regex=regex, format=fmt)


def _draw_matrix(
    x: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
) -> pd.DataFrame:
    if getattr(x, "fit_performed", False):
        components = _posterior_components(x)
        data = {name: np.asarray(value, float).reshape(-1) for name, value in components.items()}
        frame = pd.DataFrame(data)
    elif isinstance(x, pd.DataFrame):
        frame = x.select_dtypes(include=[np.number]).copy()
    elif isinstance(x, Mapping):
        frame = pd.DataFrame({str(k): np.asarray(v, float).reshape(-1) for k, v in x.items()})
    else:
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise GP3BayesError("Posterior draws must be a fit, mapping, or numeric 2-D object.")
        names = getattr(x, "columns", None)
        if names is None:
            names = [f"variable_{i + 1}" for i in range(arr.shape[1])]
        frame = pd.DataFrame(arr, columns=[str(v) for v in names])
    selected = list(frame.columns)
    if variables is not None:
        requested = [variables] if isinstance(variables, str) else list(variables)
        missing = [v for v in requested if v not in frame.columns]
        if missing:
            raise GP3BayesError("Unknown posterior variables: " + ", ".join(map(str, missing)))
        selected = [v for v in selected if v in requested]
    if regex is not None:
        try:
            pattern = re.compile(regex)
        except re.error as exc:
            raise GP3BayesError("`regex` must be a valid regular expression.") from exc
        selected = [v for v in selected if pattern.search(str(v))]
    if not selected:
        raise GP3BayesError("No posterior variables remain after selection.")
    out = frame.loc[:, selected].copy()
    if not np.isfinite(out.to_numpy(float)).all():
        raise GP3BayesError("Posterior draws must be finite.")
    return out


def _probs(probs: Sequence[float], *, three: bool = True) -> tuple[float, ...]:
    values = tuple(float(v) for v in probs)
    if (three and len(values) != 3) or (not three and len(values) != 2):
        raise GP3BayesError("`probs` has the wrong length.")
    if any(not np.isfinite(v) or v < 0 or v > 1 for v in values) or any(
        a >= b for a, b in zip(values, values[1:], strict=False)
    ):
        raise GP3BayesError("`probs` must contain increasing probabilities in [0, 1].")
    return values


def posterior_interval_table(
    x: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    draws = _draw_matrix(x, variables, regex)
    p = _probs(probs)
    rows = []
    for name in draws.columns:
        z = draws[name].to_numpy(float)
        q = np.quantile(z, p, method="linear")
        rows.append(
            {
                "variable": name,
                "mean": float(z.mean()),
                "median": float(q[1]),
                "sd": float(z.std(ddof=1)) if len(z) > 1 else 0.0,
                "lower": float(q[0]),
                "upper": float(q[2]),
            }
        )
    return pd.DataFrame(rows)


def posterior_probability_table(
    x: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    rope: Sequence[float] | None = None,
) -> pd.DataFrame:
    draws = _draw_matrix(x, variables, regex)
    rope_values: tuple[float, float] | None = None
    if rope is not None:
        r = tuple(float(v) for v in rope)
        if len(r) != 2 or not r[0] < r[1] or not all(np.isfinite(r)):
            raise GP3BayesError("`rope` must be NULL or two increasing finite numbers.")
        rope_values = (r[0], r[1])
    rows = []
    for name in draws.columns:
        z = draws[name].to_numpy(float)
        row = {
            "variable": name,
            "probability_gt_zero": float(np.mean(z > 0)),
            "probability_lt_zero": float(np.mean(z < 0)),
        }
        if rope_values is not None:
            row.update(
                {
                    "probability_in_rope": float(
                        np.mean((z >= rope_values[0]) & (z <= rope_values[1]))
                    ),
                    "rope_lower": rope_values[0],
                    "rope_upper": rope_values[1],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def posterior_correlation_table(
    x: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
    method: str = "pearson",
) -> pd.DataFrame:
    if method not in {"pearson", "spearman"}:
        raise GP3BayesError("`method` must be 'pearson' or 'spearman'.")
    draws = _draw_matrix(x, variables, regex)
    if draws.shape[1] < 2:
        raise GP3BayesError("At least two posterior variables are required.")
    values = draws.rank(method="average") if method == "spearman" else draws
    corr = values.corr(method="pearson")
    rows = []
    names = list(corr.columns)
    for i in range(1, len(names)):
        for j in range(i):
            rows.append(
                {
                    "variable_1": names[i],
                    "variable_2": names[j],
                    "correlation": float(corr.iloc[i, j]),  # type: ignore[arg-type]
                    "method": method,
                }
            )
    return pd.DataFrame(rows)


def _split_rhat(arr: np.ndarray) -> float:
    if arr.ndim != 2 or arr.shape[0] < 2 or arr.shape[1] < 4:
        return float("nan")
    m, n = arr.shape
    half = n // 2
    chains = np.concatenate([arr[:, :half], arr[:, -half:]], axis=0)
    n = half
    chain_means = chains.mean(axis=1)
    w = np.mean(np.var(chains, axis=1, ddof=1))
    if not np.isfinite(w) or w <= 0:
        return 1.0 if np.allclose(chains, chains.flat[0]) else float("nan")
    b = n * np.var(chain_means, ddof=1)
    var_hat = ((n - 1) / n) * w + b / n
    return float(np.sqrt(var_hat / w))


def _ess_1d(z: np.ndarray) -> float:
    z = np.asarray(z, float)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 4:
        return float(n)
    z = z - z.mean()
    var = np.dot(z, z) / n
    if var <= np.finfo(float).eps:
        return float(n)
    # Initial positive sequence approximation.
    max_lag = min(n - 1, 1000)
    rho_sum = 0.0
    for lag in range(1, max_lag, 2):
        r1 = np.dot(z[:-lag], z[lag:]) / ((n - lag) * var)
        if lag + 1 <= max_lag:
            r2 = np.dot(z[: -(lag + 1)], z[lag + 1 :]) / ((n - lag - 1) * var)
        else:
            r2 = 0.0
        pair = r1 + r2
        if not np.isfinite(pair) or pair < 0:
            break
        rho_sum += pair
    return float(min(n, n / max(1.0 + 2.0 * rho_sum, np.finfo(float).eps)))


def mcmc_diagnostic_table(
    fit: Any,
    variables: Sequence[str] | str | None = None,
    regex: str | None = None,
) -> pd.DataFrame:
    components = _posterior_components(fit)
    selected = list(components)
    if variables is not None:
        requested = [variables] if isinstance(variables, str) else list(variables)
        missing = [v for v in requested if v not in components]
        if missing:
            raise GP3BayesError("Unknown posterior variables: " + ", ".join(missing))
        selected = [v for v in selected if v in requested]
    if regex is not None:
        pat = re.compile(regex)
        selected = [v for v in selected if pat.search(v)]
    rows = []
    for name in selected:
        arr = np.asarray(components[name], float)
        flat = arr.reshape(-1)
        sd = float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
        ess = _ess_1d(flat)
        rows.append(
            {
                "variable": name,
                "mean": float(np.mean(flat)),
                "median": float(np.median(flat)),
                "sd": sd,
                "rhat": _split_rhat(arr),
                "ess_bulk": ess,
                "ess_tail": ess,
                "mcse_mean": sd / math.sqrt(ess) if ess > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def identify_mcmc_issues(
    x: Any,
    rhat_threshold: float = 1.01,
    min_bulk_ess: float = 400,
    min_tail_ess: float = 400,
    max_mcse_fraction: float = 0.10,
) -> pd.DataFrame:
    if rhat_threshold <= 1:
        raise GP3BayesError("`rhat_threshold` must be greater than 1.")
    d = mcmc_diagnostic_table(x) if getattr(x, "fit_performed", False) else x
    if not isinstance(d, pd.DataFrame):
        raise GP3BayesError("`x` must be a fit or diagnostic table.")
    required = {"variable", "sd", "rhat", "ess_bulk", "ess_tail", "mcse_mean"}
    if not required.issubset(d.columns):
        raise GP3BayesError("The diagnostic table is missing required columns.")
    sd = pd.to_numeric(d["sd"], errors="coerce").to_numpy(float)
    mcse = pd.to_numeric(d["mcse_mean"], errors="coerce").to_numpy(float)
    safe = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
    frac = np.abs(mcse) / safe
    rhat = pd.to_numeric(d["rhat"], errors="coerce").to_numpy(float)
    bulk = pd.to_numeric(d["ess_bulk"], errors="coerce").to_numpy(float)
    tail = pd.to_numeric(d["ess_tail"], errors="coerce").to_numpy(float)
    out = pd.DataFrame(
        {
            "variable": d["variable"].astype(str).to_numpy(),
            "rhat": rhat,
            "ess_bulk": bulk,
            "ess_tail": tail,
            "mcse_fraction": frac,
            "rhat_flag": ~np.isfinite(rhat) | (rhat > rhat_threshold),
            "bulk_ess_flag": ~np.isfinite(bulk) | (bulk < min_bulk_ess),
            "tail_ess_flag": ~np.isfinite(tail) | (tail < min_tail_ess),
            "mcse_flag": ~np.isfinite(frac) | (frac > max_mcse_fraction),
        }
    )
    out["flagged"] = out[["rhat_flag", "bulk_ess_flag", "tail_ess_flag", "mcse_flag"]].any(axis=1)
    return out


def extract_sampler_diagnostics(fit: Any) -> pd.DataFrame:
    backend = getattr(fit, "backend_fit", None)
    stats = getattr(backend, "sample_stats", None)
    if stats is None:
        raise GP3BayesError("Sampler diagnostics are unavailable for this fit.")
    rows: list[pd.DataFrame] = []
    for raw_name in getattr(stats, "data_vars", {}):
        try:
            arr = np.asarray(stats[raw_name], dtype=float)
        except Exception:
            continue
        if arr.ndim < 2:
            continue
        chain_count, draw_count = arr.shape[:2]
        flat = arr.reshape(chain_count, draw_count, -1)
        # Scalar sampler statistics are the intended path; for any extra
        # dimensions retain each component deterministically.
        for component in range(flat.shape[2]):
            values = flat[:, :, component]
            rows.append(
                pd.DataFrame(
                    {
                        "Chain": np.repeat(np.arange(1, chain_count + 1), draw_count),
                        "Iteration": np.tile(np.arange(1, draw_count + 1), chain_count),
                        "Parameter": str(raw_name),
                        "Value": values.reshape(-1),
                    }
                )
            )
    if not rows:
        return pd.DataFrame(columns=["Chain", "Iteration", "Parameter", "Value"])
    return pd.concat(rows, ignore_index=True)


def sampler_diagnostic_table(fit: Any) -> pd.DataFrame:
    np_table = extract_sampler_diagnostics(fit)
    if np_table.empty:
        return pd.DataFrame(columns=["metric", "value", "threshold", "flagged"])
    names = np_table["Parameter"].str.lower()
    values = pd.to_numeric(np_table["Value"], errors="coerce").to_numpy(float)
    divergence_mask = names.isin(["diverging", "divergent__"]).to_numpy()
    depth_mask = names.isin(["tree_depth", "treedepth__"]).to_numpy()
    max_td = float(getattr(fit, "sampling", {}).get("max_treedepth", 12))
    divergence = int(np.sum(values[divergence_mask] > 0))
    treedepth = int(np.sum(values[depth_mask] >= max_td))
    rows: list[dict[str, Any]] = [
        {
            "metric": "divergent_transitions",
            "value": divergence,
            "threshold": 0.0,
            "flagged": divergence > 0,
        },
        {
            "metric": "max_treedepth_hits",
            "value": treedepth,
            "threshold": 0.0,
            "flagged": treedepth > 0,
        },
    ]
    energy_mask = names.isin(["energy", "energy__"]).to_numpy()
    if energy_mask.any():
        energy_frame = np_table.loc[energy_mask, ["Chain", "Value"]]
        for chain, frame in energy_frame.groupby("Chain", sort=True):
            z = pd.to_numeric(frame["Value"], errors="coerce").dropna().to_numpy(float)
            bfmi = (
                float(np.mean(np.diff(z) ** 2) / np.var(z, ddof=1))
                if len(z) >= 3 and np.var(z, ddof=1) > 0
                else float("nan")
            )
            rows.append(
                {
                    "metric": f"ebfmi_chain_{chain}",
                    "value": bfmi,
                    "threshold": 0.30,
                    "flagged": not np.isfinite(bfmi) or bfmi < 0.30,
                }
            )
    return pd.DataFrame(rows)


@dataclass(slots=True)
class MCMCQuality:
    family: str
    parameters: pd.DataFrame
    issues: pd.DataFrame
    sampler: pd.DataFrame
    flagged_parameters: int
    flagged_sampler_metrics: int
    interpretation: str = "Flags identify diagnostics requiring inspection. Absence of flags does not establish model adequacy."

    def to_frame(self) -> pd.DataFrame:
        return self.issues.copy()


def summarise_mcmc_quality(
    fit: Any,
    rhat_threshold: float = 1.01,
    min_bulk_ess: float = 400,
    min_tail_ess: float = 400,
    max_mcse_fraction: float = 0.1,
) -> MCMCQuality:
    parameters = mcmc_diagnostic_table(fit)
    issues = identify_mcmc_issues(
        parameters, rhat_threshold, min_bulk_ess, min_tail_ess, max_mcse_fraction
    )
    try:
        sampler = sampler_diagnostic_table(fit)
    except GP3BayesError:
        sampler = pd.DataFrame(columns=["metric", "value", "threshold", "flagged"])
    return MCMCQuality(
        str(getattr(fit, "family", "unknown")),
        parameters,
        issues,
        sampler,
        int(issues["flagged"].sum()),
        int(sampler["flagged"].sum()) if "flagged" in sampler else 0,
    )


def extract_log_likelihood(
    fit: Any,
    newdata: pd.DataFrame | None = None,
    include_group_effects: bool = True,
    ndraws: int | None = None,
) -> np.ndarray:
    if newdata is not None:
        raise GP3BayesError(
            "Python backend log-likelihood extraction currently supports fitted-data rows only."
        )
    backend = getattr(fit, "backend_fit", None)
    group = getattr(backend, "log_likelihood", None)
    if group is None:
        raise GP3BayesError(
            "Pointwise log likelihood was not stored for this fit. Refit with log-likelihood storage enabled."
        )
    data_vars = getattr(group, "data_vars", None)
    if not data_vars:
        raise GP3BayesError("The stored log-likelihood group contains no variables.")
    name = list(data_vars)[0]
    arr = np.asarray(group[name], dtype=float)
    if arr.ndim < 2:
        raise GP3BayesError("The stored log-likelihood array has an unsupported shape.")
    arr = arr.reshape(arr.shape[0] * arr.shape[1], -1)
    if ndraws is not None:
        if not isinstance(ndraws, int) or ndraws < 1:
            raise GP3BayesError("`ndraws` must be NULL or one positive integer.")
        arr = arr[:ndraws]
    if not np.isfinite(arr).all():
        raise GP3BayesError("The extracted log-likelihood matrix is not finite.")
    return arr


def group_effect_table(
    fit: Any,
    groups: Sequence[str] | str | None = None,
    probs: Sequence[float] = (0.025, 0.975),
) -> pd.DataFrame:
    p = _probs(probs, three=False)
    components = _posterior_components(fit)
    spec = getattr(fit, "specification", None)
    prepared = getattr(spec, "prepared", None)
    data = getattr(prepared, "data", None)
    contract = getattr(spec, "contract", None)
    if data is None or contract is None:
        raise GP3BayesError("The fit must retain prepared data and its model contract.")
    available_groups: dict[str, tuple[str, str]] = {}
    participant = contract.mappings.get("participant")
    item = contract.mappings.get("item")
    if isinstance(participant, str) and participant in data:
        available_groups["participant"] = (participant, "participant")
    if isinstance(item, str) and item in data:
        available_groups["item"] = (item, "item")
    requested = (
        list(available_groups)
        if groups is None
        else ([groups] if isinstance(groups, str) else list(groups))
    )
    missing = [g for g in requested if g not in available_groups]
    if missing:
        raise GP3BayesError("Unknown grouping factors: " + ", ".join(missing))
    rows: list[dict[str, Any]] = []
    for group_name in requested:
        column, stem = available_groups[group_name]
        levels = pd.unique(data[column].dropna()).tolist()
        sd_name = f"sd_{stem}"
        z_name = f"{stem}_z"
        if sd_name in components and z_name in components:
            sd = np.asarray(components[sd_name], float)
            z = np.asarray(components[z_name], float)
            # z may have been flattened into component names by _posterior_components.
            for idx, level in enumerate(levels):
                component = f"{z_name}[{idx + 1}]"
                if component in components:
                    draws = sd * np.asarray(components[component], float)
                elif z.ndim >= 3 and idx < z.shape[2]:
                    draws = sd * z[:, :, idx]
                else:
                    continue
                flat = draws.reshape(-1)
                q = np.quantile(flat, p, method="linear")
                rows.append(
                    {
                        "group": group_name,
                        "level": str(level),
                        "coefficient": "Intercept",
                        "estimate": float(np.mean(flat)),
                        "se": float(np.std(flat, ddof=1)),
                        "lower": float(q[0]),
                        "upper": float(q[1]),
                    }
                )
        else:
            # Canonical flattened r_* components from other backends.
            prefix = f"r_{stem}["
            for name, arr in components.items():
                if not name.startswith(prefix):
                    continue
                flat = np.asarray(arr, float).reshape(-1)
                q = np.quantile(flat, p, method="linear")
                rows.append(
                    {
                        "group": group_name,
                        "level": name[len(prefix) :].split(",", 1)[0].rstrip("]"),
                        "coefficient": "Intercept",
                        "estimate": float(flat.mean()),
                        "se": float(flat.std(ddof=1)),
                        "lower": float(q[0]),
                        "upper": float(q[1]),
                    }
                )
    if not rows:
        raise GP3BayesError("No group-level effects could be extracted.")
    return pd.DataFrame(rows)


def variance_component_table(
    fit: Any,
    probs: Sequence[float] = (0.025, 0.5, 0.975),
) -> pd.DataFrame:
    return posterior_interval_table(fit, regex=r"^(sd_|cor_|sigma$)", probs=probs)


def loo_diagnostic_table(x: Any) -> pd.DataFrame:
    k = getattr(x, "pareto_k", None)
    if k is None and isinstance(x, Mapping):
        k = x.get("pareto_k")
    if k is None:
        raise GP3BayesError("`x` must contain Pareto-k diagnostics.")
    values = np.asarray(k, float).reshape(-1)
    category = np.select(
        [values < 0.5, values < 0.7, values < 1.0],
        ["good", "okay", "review"],
        default="severe",
    )
    return pd.DataFrame(
        {
            "observation": np.arange(1, len(values) + 1),
            "pareto_k": values,
            "category": category,
            "flagged": ~np.isfinite(values) | (values >= 0.7),
        }
    )


def loo_summary_table(x: Any) -> pd.DataFrame:
    estimates = getattr(x, "estimates", None)
    if estimates is None and hasattr(x, "raw"):
        estimates = getattr(x.raw, "estimates", None)
    if isinstance(estimates, pd.DataFrame):
        out = estimates.copy()
        if "quantity" not in out:
            out = out.reset_index().rename(columns={"index": "quantity"})
        return out
    # Our LOOResult uses scalar fields.
    rows = []
    for quantity, attr, se_attr in [
        ("elpd_loo", "elpd_loo", "se_elpd_loo"),
        ("p_loo", "p_loo", "se_p_loo"),
        ("looic", "looic", "se_looic"),
    ]:
        if hasattr(x, attr):
            rows.append(
                {
                    "quantity": quantity,
                    "estimate": float(getattr(x, attr)),
                    "se": float(getattr(x, se_attr, np.nan)),
                }
            )
    if not rows:
        raise GP3BayesError("`x` does not contain LOO summary estimates.")
    return pd.DataFrame(rows)


def model_comparison_table(x: Any) -> pd.DataFrame:
    m = getattr(x, "comparison", x)
    if isinstance(m, pd.DataFrame):
        required = {"elpd_diff", "se_diff"}
        if not required.issubset(m.columns):
            raise GP3BayesError("The comparison object does not contain ELPD-difference columns.")
        out = m.copy()
        if "model" not in out:
            out = out.reset_index().rename(columns={"index": "model"})
        out["automatic_selection"] = False
        return out[["model", "elpd_diff", "se_diff", "automatic_selection"]]
    raise GP3BayesError("`x` must contain a model-comparison table.")


def model_weights_table(x: Any) -> pd.DataFrame:
    w = getattr(x, "weights", x)
    if isinstance(w, Mapping):
        names = list(map(str, w.keys()))
        values = np.asarray(list(w.values()), float)
    elif isinstance(w, pd.Series):
        names = w.index.astype(str).tolist()
        values = w.to_numpy(float)
    else:
        values = np.asarray(w, float).reshape(-1)
        names = [f"model_{i + 1}" for i in range(len(values))]
    if not len(values) or not np.isfinite(values).all():
        raise GP3BayesError("`x` must contain finite numeric model weights.")
    return pd.DataFrame({"model": names, "weight": values, "automatic_selection": False})


__all__ = [
    "MCMCQuality",
    "extract_log_likelihood",
    "extract_posterior_draws",
    "extract_sampler_diagnostics",
    "group_effect_table",
    "identify_mcmc_issues",
    "loo_diagnostic_table",
    "loo_summary_table",
    "mcmc_diagnostic_table",
    "model_comparison_table",
    "model_weights_table",
    "posterior_correlation_table",
    "posterior_interval_table",
    "posterior_probability_table",
    "sampler_diagnostic_table",
    "summarise_mcmc_quality",
    "variance_component_table",
]
