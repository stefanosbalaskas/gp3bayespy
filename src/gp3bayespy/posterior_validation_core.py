"""Posterior-validation plotting helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .exceptions import GP3BayesError
from .posterior import _posterior_components, _validate_fit_like
from .postfit_exploration import extract_sampler_diagnostics


def _mpl():
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise GP3BayesError(
            "Matplotlib is required for plotting; install gp3bayespy[plots]."
        ) from exc
    return plt


def plot_sampling_diagnostics(
    fit: Any,
    type: str = "trace",
    variables: Sequence[str] | str | None = None,
):
    """Plot trace, energy, treedepth, or divergence sampling diagnostics.

    The returned Matplotlib figure is interpretive only; it does not establish
    convergence or substantive model adequacy.
    """
    fit = _validate_fit_like(fit)
    kind = str(type)
    allowed = {"trace", "energy", "treedepth", "divergence"}
    if kind not in allowed:
        raise GP3BayesError("`type` must be one of: trace, energy, treedepth, divergence.")
    plt = _mpl()
    if kind == "trace":
        components = _posterior_components(fit)
        names = list(components)
        if variables is not None:
            requested = [variables] if isinstance(variables, str) else list(variables)
            missing = [name for name in requested if name not in components]
            if missing:
                raise GP3BayesError(
                    "Requested posterior variables were not found: " + ", ".join(missing) + "."
                )
            names = [str(name) for name in requested]
        elif len(names) > 8:
            names = names[:8]
        fig, axes = plt.subplots(
            len(names), 1, squeeze=False, figsize=(8, max(2.4, 1.8 * len(names)))
        )
        for row, name in enumerate(names):
            values = np.asarray(components[name], float)
            for chain in range(values.shape[0]):
                axes[row, 0].plot(values[chain].reshape(-1), alpha=0.8, label=f"chain {chain + 1}")
            axes[row, 0].set_ylabel(name)
        axes[-1, 0].set_xlabel("Iteration")
        if names and components[names[0]].shape[0] <= 6:
            axes[0, 0].legend()
        fig.suptitle("Posterior trace diagnostics")
        fig.tight_layout()
        return fig

    sampler = extract_sampler_diagnostics(fit)
    if sampler.empty:
        raise GP3BayesError("Sampler diagnostics were not available for this fitted backend.")
    fig, ax = plt.subplots()
    if kind == "energy":
        d = sampler[sampler["Parameter"].astype(str).str.lower().isin({"energy", "energy__"})]
        if d.empty:
            raise GP3BayesError("Energy diagnostics were not available for this fitted backend.")
        for chain, frame in d.groupby("Chain", observed=False):  # type: ignore[assignment]
            ax.hist(frame["Value"], bins=30, histtype="step", density=True, label=f"chain {chain}")
        ax.set_xlabel("Energy")
        ax.set_ylabel("Density")
        ax.set_title("NUTS energy diagnostic")
        ax.legend()
    elif kind == "treedepth":
        d = sampler[
            sampler["Parameter"]
            .astype(str)
            .str.lower()
            .isin({"tree_depth", "treedepth", "treedepth__"})
        ]
        if d.empty:
            raise GP3BayesError(
                "Tree-depth diagnostics were not available for this fitted backend."
            )
        counts = d["Value"].value_counts().sort_index()
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_xlabel("Tree depth")
        ax.set_ylabel("Iterations")
        ax.set_title("NUTS tree-depth diagnostic")
    else:
        d = sampler[
            sampler["Parameter"]
            .astype(str)
            .str.lower()
            .isin({"diverging", "divergent", "divergent__"})
        ]
        if d.empty:
            raise GP3BayesError(
                "Divergence diagnostics were not available for this fitted backend."
            )
        divergent = d[d["Value"] > 0]
        ax.scatter(d["Iteration"], d["Chain"], alpha=0.15, label="all")
        if not divergent.empty:
            ax.scatter(divergent["Iteration"], divergent["Chain"], marker="x", label="divergent")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Chain")
        ax.set_title("NUTS divergence diagnostic")
        ax.legend()
    fig.tight_layout()
    return fig


__all__ = ["plot_sampling_diagnostics"]
