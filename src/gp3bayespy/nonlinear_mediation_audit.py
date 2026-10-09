"""Experimental frequentist functional-form check for clustered mediation.

This diagnostic is not a Bayesian posterior, not automated causal discovery,
and does not identify natural effects without strong assumptions.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _estimate(x: np.ndarray, m: np.ndarray, y: np.ndarray, degree: int) -> float:
    a = np.column_stack([np.ones(len(x)), x])
    ab, _, rank, _ = np.linalg.lstsq(a, m, rcond=None)
    if rank < 2:
        raise ValueError("treatment groups are not both represented")
    mu0 = np.full(len(x), ab[0])
    mu1 = mu0 + ab[1]
    residual = m - (a @ ab)
    m0, m1 = mu0 + residual, mu1 + residual
    def features(v: np.ndarray, treatment: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(v)), treatment, v, treatment * v]
        if degree == 2:
            cols.extend([v * v, treatment * v * v])
        return np.column_stack(cols)
    design = features(m, x)
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < design.shape[1]:
        raise ValueError("outcome model has insufficient design rank")
    treated = np.ones(len(x))
    # Interventional model-based mediator shift at a held treatment value.
    return float(np.mean(features(m1, treated) @ coef - features(m0, treated) @ coef))


def audit_mediation_functional_form(
    data: pd.DataFrame,
    *,
    treatment_col: str,
    mediator_col: str,
    outcome_col: str,
    participant_col: str = "participant_id",
    bootstrap_replicates: int = 199,
    seed: int = 2026,
) -> dict:
    """Compare linear versus quadratic mediator-outcome models.

    Binary treatment, continuous mediator/outcome, subject-cluster resampling.
    Both models include treatment-mediator interactions. Bootstraps are refits,
    and failed replicates are retained in the evidence ledger, not discarded.
    """
    names = [treatment_col, mediator_col, outcome_col, participant_col]
    if not isinstance(data, pd.DataFrame) or data.empty or len(set(names)) != 4:
        raise ValueError("data and distinct treatment/mediator/outcome/participant columns required")
    if not set(names).issubset(data.columns):
        raise ValueError("required columns missing")
    if data[participant_col].isna().any():
        raise ValueError("participant identity cannot be missing")
    if not isinstance(bootstrap_replicates, int) or not 0 <= bootstrap_replicates <= 10000:
        raise ValueError("bootstrap_replicates must be an integer from 0 through 10000")
    x = pd.to_numeric(data[treatment_col], errors="coerce").to_numpy(dtype=float)
    m = pd.to_numeric(data[mediator_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(data[outcome_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(np.column_stack([x, m, y])).all() or not np.isin(x, [0, 1]).all():
        raise ValueError("all analysis values must be finite and treatment must be 0/1")
    groups = pd.factorize(data[participant_col], sort=False)[0]
    n_groups = int(groups.max() + 1)
    if n_groups < 4:
        raise ValueError("at least four independent participants required")
    estimates = {}
    for label, degree in (("linear", 1), ("quadratic", 2)):
        estimates[label] = _estimate(x, m, y, degree)
    rng = np.random.default_rng(seed)
    indices = [np.flatnonzero(groups == i) for i in range(n_groups)]
    draws = []
    for replicate in range(bootstrap_replicates):
        chosen = rng.integers(0, n_groups, size=n_groups)
        idx = np.concatenate([indices[i] for i in chosen])
        row = {"replicate": replicate}
        for label, degree in (("linear", 1), ("quadratic", 2)):
            try:
                row[label] = _estimate(x[idx], m[idx], y[idx], degree)
                row[label + "_error"] = None
            except (ValueError, np.linalg.LinAlgError) as exc:
                row[label] = np.nan
                row[label + "_error"] = type(exc).__name__ + ": " + str(exc)
        draws.append(row)
    boot = pd.DataFrame(
        draws, columns=["replicate", "linear", "linear_error", "quadratic", "quadratic_error"]
    )
    rows = []
    for label in ("linear", "quadratic"):
        successes = boot[label].dropna().to_numpy(dtype=float) if bootstrap_replicates else np.array([])
        rows.append({
            "model": label,
            "model_based_indirect_contrast": estimates[label],
            "bootstrap_successes": int(len(successes)),
            "bootstrap_failures": bootstrap_replicates - len(successes),
            "bootstrap_percentile_lower": float(np.quantile(successes, .025)) if len(successes) else np.nan,
            "bootstrap_percentile_upper": float(np.quantile(successes, .975)) if len(successes) else np.nan,
        })
    return {
        "estimates": pd.DataFrame(rows),
        "bootstrap_ledger": boot,
        "n_participants": n_groups,
        "n_rows": len(x),
        "estimand": "model_based_indirect_mediator_shift_at_treatment_1",
        "claim_boundary": (
            "Functional-form and participant-cluster bootstrap sensitivity only. "
            "Not a Bayesian posterior or identified causal natural indirect effect; "
            "requires independently defended mediator-outcome assumptions."
        ),
    }
