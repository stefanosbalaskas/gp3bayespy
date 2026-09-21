import csv
import inspect
import json
from pathlib import Path

import gp3bayespy as gp

ROOT = Path(__file__).resolve().parents[1]


def test_all_458_frozen_exports_are_root_importable_and_promoted():
    rows = list(csv.DictReader((ROOT / "dev/parity/function_map.csv").open()))
    assert len(rows) == 458
    assert all(row["status"] == "implemented" for row in rows)
    missing = [row["python_name"] for row in rows if not hasattr(gp, row["python_name"])]
    assert missing == []
    assert gp.parity_counts() == {
        "implemented": 458,
        "implemented_initial": 0,
        "mapped_not_implemented": 0,
    }


def test_frozen_exports_do_not_use_public_kwargs_catchalls():
    rows = list(csv.DictReader((ROOT / "dev/parity/function_map.csv").open()))
    offenders = []
    for row in rows:
        signature = inspect.signature(getattr(gp, row["python_name"]))
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
            offenders.append(row["python_name"])
    assert offenders == []


def test_all_59_articles_are_materialized_and_promoted():
    mapping = json.loads((ROOT / "dev/parity/articles.json").read_text())
    assert len(mapping) == 59
    assert all(article["status"] == "ported" for article in mapping)
    missing = [
        article["python_article"]
        for article in mapping
        if not (ROOT / "docs/articles" / article["python_article"]).is_file()
    ]
    assert missing == []


def test_canonical_repository_examples_are_preserved_and_mediation_examples_exist():
    canonical = {
        "backend_status.py",
        "binary_workflow.py",
        "duration_workflow.py",
        "loo_model_comparison.py",
        "predictive_diagnostics.py",
        "pupil_workflow.py",
        "reproducibility_workflow.py",
        "sensitivity_workflow.py",
    }

    mediation = {
        "multilevel_gaze_mediation_contract.py",
        "multilevel_gaze_mediation_failure_cases.py",
        "multilevel_gaze_mediation_family_contracts.py",
        "multilevel_gaze_mediation_prior_sensitivity_contract.py",
        "multilevel_gaze_mediation_simulation_recovery_contract.py",
    }

    discovered = {path.name for path in (ROOT / "examples").glob("*.py")}

    # Backward compatibility: all eight frozen examples remain available.
    assert canonical <= discovered

    # New method tranche: all five mediation examples are shipped.
    assert mediation <= discovered
