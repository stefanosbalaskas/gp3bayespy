from pathlib import Path

import pandas as pd

from gp3bayespy.reporting import (
    create_complete_evidence_inventory,
    create_diagnostic_dashboard,
    create_figure_set,
    create_publication_registry,
    diagnostic_dashboard_table,
    evidence_inventory_table,
    plot_binary_calibration,
    plot_predictive_coverage,
    plot_reporting_checklist,
    publication_registry_table,
    register_publication_figure,
    register_publication_table,
    save_figure_set,
    validate_publication_registry,
)


def test_publication_registry_is_explicit_and_valid(tmp_path: Path):
    registry = create_publication_registry("example")
    registry = register_publication_table(
        registry,
        "posterior",
        pd.DataFrame({"variable": ["b_x"], "median": [0.2]}),
        caption="Posterior summary",
        source="example",
    )
    assert len(publication_registry_table(registry)) == 1
    assert validate_publication_registry(registry).valid
    assert validate_publication_registry(registry).automatic_writing is False
    fig = plot_binary_calibration(
        pd.DataFrame(
            {"mean_predicted_probability": [0.2, 0.5, 0.8], "observed_rate": [0.25, 0.45, 0.75]}
        )
    )
    registry = register_publication_figure(registry, "figure1", fig)
    fs = create_figure_set({"example": fig})
    out = save_figure_set(fs, tmp_path)
    assert Path(out.loc[0, "file"]).exists()


def test_inventory_and_dashboard_are_descriptive():
    inventory = create_complete_evidence_inventory(
        {"diagnostic": {"status": "review"}, "table": pd.DataFrame({"x": [1]})}
    )
    table = evidence_inventory_table(inventory)
    assert len(table) == 2
    assert not table["automatic_decision"].any()
    dashboard = create_diagnostic_dashboard(model_card={"example": True}, label="example")
    d = diagnostic_dashboard_table(dashboard)
    assert bool(d.loc[d["component"].eq("model_card"), "available"].iloc[0])
    assert dashboard.automatic_decision is False


def test_reporting_checklist_and_calibration_plots():
    class Card:
        pass

    # DataFrame checklists are accepted directly by the plotting adapter.
    checklist = pd.DataFrame(
        {"item": ["formula_recorded"], "available": [True], "automatic_requirement": [False]}
    )
    fig = plot_reporting_checklist(checklist)
    assert hasattr(fig, "savefig")
    coverage = pd.DataFrame(
        {
            "nominal_coverage": [0.5, 0.8, 0.95],
            "empirical_coverage": [0.52, 0.79, 0.93],
            "mean_interval_width": [1, 2, 3],
        }
    )
    assert hasattr(plot_predictive_coverage(coverage), "savefig")
