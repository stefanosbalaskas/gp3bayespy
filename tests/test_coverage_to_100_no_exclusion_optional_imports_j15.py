from __future__ import annotations

import builtins

import pytest

import gp3bayespy.hierarchical_effects_advanced as hierarchical
import gp3bayespy.posterior_validation_core as validation
import gp3bayespy.predictive as predictive
import gp3bayespy.reporting as reporting
import gp3bayespy.sensitivity as sensitivity
from gp3bayespy.exceptions import GP3BayesError


@pytest.mark.parametrize(
    ("helper", "message"),
    [
        (hierarchical._mpl, "Matplotlib is required for plotting"),
        (validation._mpl, "Matplotlib is required for plotting"),
        (predictive._plt, "Matplotlib is required for plotting"),
        (reporting._plt, "Matplotlib is required for publication graphics"),
        (sensitivity._mpl, "Matplotlib is required for sensitivity plots"),
    ],
)
def test_optional_matplotlib_import_guards_without_coverage_exclusions(
    monkeypatch,
    helper,
    message,
):
    original_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("synthetic optional-dependency failure")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(GP3BayesError, match=message):
        helper()
