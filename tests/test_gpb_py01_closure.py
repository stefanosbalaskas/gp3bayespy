import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from gp3bayespy import (
    audit_model_readiness,
    build_model_formula,
    create_model_contract,
    create_model_specification,
    create_prior_specification,
    parity_counts,
    read_parity_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (ROOT / "dev/parity/core_reference_cases_0.5.0.json").read_text(encoding="utf-8")
)


def _balanced_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [f"p{i}" for i in range(1, 5) for _ in range(4)],
            "stimulus_id": [f"s{i}" for _ in range(4) for i in range(1, 5)],
            "trial_id": list(range(1, 5)) * 4,
            "condition": pd.Categorical(
                ["control", "treatment"] * 8,
                categories=["control", "treatment"],
            ),
            "trial_order": list(range(1, 5)) * 4,
            "age_z": [-1.0] * 4 + [-0.5] * 4 + [0.5] * 4 + [1.0] * 4,
            "selected": [0, 1, 1, 0] * 4,
        }
    )


def _binary_contract():
    expected = FIXTURE["binary_contract"]
    return create_model_contract(
        family=expected["family"],
        outcome_col=expected["outcome"],
        participant_col=expected["participant"],
        item_col=expected["item"],
        trial_col=expected["trial"],
        condition_col=expected["condition"],
        time_col=expected["time"],
        predictors=expected["predictors"],
        interaction=expected["interaction"],
        random_slope=expected["random_slope"],
    )


def test_core_fixture_provenance_is_explicit():
    assert FIXTURE["reference_package"] == "gp3bayes"
    assert FIXTURE["reference_version"] == "0.5.0"
    assert FIXTURE["provenance"]["runtime_captured_from_R"] is False
    assert len(FIXTURE["provenance"]["sources"]) == 6


def test_core_formula_matches_r_derived_fixture():
    assert build_model_formula(_binary_contract()) == FIXTURE["binary_formula"]
    nonsyntactic = create_model_contract(
        "binary",
        "selected outcome",
        "participant id",
        predictors=["age score", "if"],
    )
    assert build_model_formula(nonsyntactic) == FIXTURE["non_syntactic_formula"]


def test_core_prior_matches_r_derived_fixture():
    expected = FIXTURE["binary_prior"]
    priors = create_prior_specification(
        _binary_contract(),
        baseline=expected["baseline"],
        intercept_scale=1.25,
        coefficient_scale=0.75,
        group_sd_scale=0.8,
        correlation_eta=3,
        student_df=4,
    )
    assert priors.transformed_baseline == pytest.approx(
        expected["transformed_baseline"], abs=1e-15
    )
    assert priors.table["parameter_class"].tolist() == expected["parameter_classes"]
    assert priors.table["distribution"].tolist() == expected["distributions"]
    assert priors.backend == expected["backend"]
    assert priors.executable is expected["executable"]


def test_core_readiness_matches_r_derived_fixture():
    expected = FIXTURE["balanced_readiness"]
    audit = audit_model_readiness(_balanced_data(), _binary_contract())
    assert audit.ready is expected["ready"]
    assert audit.status == expected["status"]
    assert audit.status_counts == expected["status_counts"]
    assert audit.checks["check_id"].tolist() == expected["check_ids"]


def test_core_model_specification_closes_from_fixture_objects():
    contract = _binary_contract()
    audit = audit_model_readiness(_balanced_data(), contract)
    priors = create_prior_specification(contract)
    specification = create_model_specification(contract, audit, priors)
    assert specification.formula_text == FIXTURE["binary_formula"]
    assert specification.readiness_status == "ready"
    assert specification.warning_count == 0
    assert specification.backend == "none"
    assert specification.fit_performed is False


def test_dev_and_packaged_ledgers_are_identical():
    with (ROOT / "dev/parity/function_map.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        dev_rows = list(csv.DictReader(handle))
    assert dev_rows == read_parity_manifest()


def test_gpb_py01_promotions_are_truthful_and_counts_are_exact():
    rows = {row["r_export"]: row for row in read_parity_manifest()}
    for name in FIXTURE["promoted_exports"]:
        assert rows[name]["status"] == "implemented"
    assert rows["backend_capabilities"]["status"] == "implemented_initial"
    assert parity_counts() == {
        "mapped_not_implemented": 451,
        "implemented": 6,
        "implemented_initial": 1,
    }
