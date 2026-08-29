from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pytest

import gp3bayespy.evidence_graphics_gg as eg
import gp3bayespy.sensitivity as sensitivity
from gp3bayespy.exceptions import GP3BayesError

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_evidence_graphics_fallback_and_status_plot_matrix(monkeypatch):
    frame = pd.DataFrame(
        {
            "component": ["a", "b"],
            "status": ["pass", "review"],
            "value": [1.0, 2.0],
        }
    )

    assert eg._field(frame).equals(frame)
    assert eg._field({"table": frame}, "table").equals(frame)
    with pytest.raises(GP3BayesError):
        eg._field(object(), "table")

    monkeypatch.setattr(
        sensitivity,
        "summarise_sensitivity_suite",
        lambda x: (_ for _ in ()).throw(RuntimeError("fallback")),
    )
    assert eg.sensitivity_suite_table({"table": frame}).equals(frame)

    assert eg._status_plot(
        frame,
        "label+status",
        "component",
    ).axes
    assert eg._status_plot(
        frame[["status"]],
        "status only",
    ).axes
    assert eg._status_plot(
        pd.DataFrame({"name": ["a", "b"]}),
        "text only",
    ).axes
    assert eg._status_plot(
        pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}),
        "numeric",
    ).axes

    pair = pd.DataFrame(
        {
            "rstan_mean": [0.1, 0.2],
            "cmdstanr_mean": [0.11, 0.19],
        }
    )
    assert eg.plot_backend_parity_gg(pair).axes
    assert eg.plot_backend_parity_gg(frame).axes
