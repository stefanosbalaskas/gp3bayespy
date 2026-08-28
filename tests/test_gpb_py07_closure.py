import csv
import json
from pathlib import Path

from gp3bayespy import parity_counts, read_parity_manifest

ROOT = Path(__file__).resolve().parents[1]

GPB_PY07_EXPORTS = {
    "check_binary_posterior_predictive",
    "check_duration_posterior_predictive",
}


def _normalize(rows):
    return [
        {
            key: (value.replace("\r\n", "\n").replace("\r", "\n") if value else value)
            for key, value in row.items()
        }
        for row in rows
    ]


def test_gpb_py07_ppc_exports_are_promoted_exactly():
    manifest = read_parity_manifest()
    status_by_export = {row["r_export"]: row["status"] for row in manifest}
    assert all(status_by_export[name] == "implemented" for name in GPB_PY07_EXPORTS)
    assert status_by_export["backend_capabilities"] in {"implemented_initial", "implemented"}


def test_gpb_py07_ledger_still_covers_all_exports():
    counts = parity_counts()
    assert counts["implemented"] >= 35
    assert counts["implemented_initial"] in {0, 1}
    assert counts["mapped_not_implemented"] <= 422
    assert sum(counts.values()) == 458


def test_gpb_py07_dev_and_packaged_ledgers_match_semantically():
    with (ROOT / "dev/parity/function_map.csv").open(newline="", encoding="utf-8") as handle:
        dev_rows = list(csv.DictReader(handle))
    assert _normalize(dev_rows) == _normalize(read_parity_manifest())


def test_gpb_py07_reference_fixture_and_runtime_evidence_are_closed():
    fixture_path = ROOT / "dev/parity/posterior_predictive_reference_0.5.0.json"
    evidence_path = ROOT / "dev/parity/posterior_predictive_backend_validation_0.1.0.dev0.json"
    assert fixture_path.is_file()
    assert evidence_path.is_file()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["ledger_promoted"] is True


def test_gpb_py07_runtime_backend_evidence_is_conservative():
    path = ROOT / "dev/parity/posterior_predictive_backend_validation_0.1.0.dev0.json"
    evidence = json.loads(path.read_text(encoding="utf-8"))
    assert evidence["validated_commit"] == "fd046c2"
    assert evidence["validated_commit_sha"] == ("fd046c267c5b3cc8bc4f778b58dcf6713ebae5e6")
    assert evidence["quality_gates"]["ruff"]["status"] == "pass"
    assert evidence["quality_gates"]["mypy"] == {
        "status": "pass",
        "source_files": 15,
    }
    assert evidence["quality_gates"]["pytest"] == {
        "status": "pass",
        "tests": 217,
    }
    assert evidence["quality_gates"]["build"]["status"] == "pass"
    smoke = evidence["backend_smoke"]
    assert smoke["status"] == "pass"
    assert smoke["binary_ppc_path"] == "pass"
    assert smoke["duration_ppc_path"] == "pass"
    assert smoke["binary_ppc_status"] == "review"
    assert smoke["duration_ppc_status"] == "pass"
    assert smoke["global_adequacy_established"] is False
    governance = evidence["governance"]
    assert governance["ppc_status_is_global_adequacy_claim"] is False
    assert governance["binary_adequacy_established"] is False
    assert governance["duration_adequacy_established"] is False
