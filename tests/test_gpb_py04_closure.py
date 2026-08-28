import csv
import json
import tomllib
from pathlib import Path

from gp3bayespy import parity_counts, read_parity_manifest

ROOT = Path(__file__).resolve().parents[1]

GPB_PY04_EXPORTS = {
    "translate_binary_model_to_brms",
    "fit_binary_model",
    "translate_duration_model_to_brms",
    "fit_duration_model",
}


def _normalize(rows):
    return [
        {
            key: (value.replace("\r\n", "\n").replace("\r", "\n") if value else value)
            for key, value in row.items()
        }
        for row in rows
    ]


def test_gpb_py04_fitting_exports_are_promoted_exactly():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    assert all(status_by_export[name] == "implemented" for name in GPB_PY04_EXPORTS)
    assert status_by_export["backend_capabilities"] in {"implemented_initial", "implemented"}


def test_gpb_py04_ledger_still_covers_all_exports():
    counts = parity_counts()
    assert counts["implemented"] >= 18
    assert counts["implemented_initial"] in {0, 1}
    assert sum(counts.values()) == 458


def test_gpb_py04_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(newline="", encoding="utf-8") as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py04_runtime_backend_evidence_is_frozen():
    path = ROOT / "dev/parity/fitting_backend_validation_0.1.0.dev0.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["kind"] == "runtime-captured Python validation evidence"
    assert evidence["validated_commit"] == "accc8f1"
    assert evidence["python_version"] == "3.13.15"
    assert evidence["quality_gates"]["pytest"] == {
        "status": "pass",
        "tests": 124,
    }
    assert evidence["quality_gates"]["ruff"]["status"] == "pass"
    assert evidence["quality_gates"]["mypy"]["status"] == "pass"
    assert evidence["quality_gates"]["build"]["status"] == "pass"
    assert evidence["backend_smoke"]["binary_fit"] == "pass"
    assert evidence["backend_smoke"]["duration_fit"] == "pass"
    assert evidence["backend_smoke"]["diagnostic_claims_permitted"] is False
    assert evidence["backend_smoke"]["posterior_adequacy_established"] is False


def test_gpb_py04_dependency_matrix_is_constrained():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    optional = pyproject["project"]["optional-dependencies"]
    bayes = set(optional["bayes"])
    dev = set(optional["dev"])
    assert "numpy>=2.0,<2.5; python_version >= '3.12'" in bayes
    assert "numba>=0.64,<=0.66.0; python_version >= '3.12'" in bayes
    assert "pymc>=6.3.1; python_version >= '3.12'" in bayes
    assert "pandas-stubs>=2.2" in dev
