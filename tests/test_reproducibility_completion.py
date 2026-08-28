from pathlib import Path

from gp3bayespy import (
    create_model_contract,
    prepare_hierarchical_binary_data,
    simulate_hierarchical_binary_data,
    specify_binary_model,
)
from gp3bayespy.reproducibility import (
    compare_analysis_manifests,
    create_analysis_manifest,
    freeze_analysis_manifest,
    read_analysis_manifest,
    validate_analysis_manifest,
    write_reproducibility_report,
)


def _spec(seed=201):
    sim = simulate_hierarchical_binary_data(
        n_participants=8, trials_per_participant=6, n_items=4, random_slope_sd=0, seed=seed
    )
    contract = create_model_contract(
        family="binary",
        outcome_col="selected",
        participant_col="participant_id",
        item_col="item_id",
        trial_col="trial_id",
        condition_col="condition",
    )
    prepared = prepare_hierarchical_binary_data(
        sim.data, contract, condition_levels=("control", "treatment")
    )
    return specify_binary_model(prepared, baseline=0.35)


def test_manifest_fingerprints_and_validates():
    manifest = create_analysis_manifest(
        specification=_spec(),
        estimands=("standardized_probability_contrast",),
        seed=2026,
        label="synthetic binary analysis",
    )
    assert manifest.family == "binary"
    assert manifest.data["available"] is True
    assert len(manifest.data["hash"]) == 32
    assert len(manifest.specification["hash"]) == 32
    assert manifest.frozen is False
    assert validate_analysis_manifest(manifest).status == "pass"


def test_manifest_freeze_roundtrip_compare_and_report(tmp_path: Path):
    a = create_analysis_manifest(_spec(202), seed=1)
    b = create_analysis_manifest(_spec(202), seed=2)
    comparison = compare_analysis_manifests(a, b)
    assert comparison.identical is False
    assert "seed" in comparison.changed_components
    path = tmp_path / "manifest.bin"
    frozen = freeze_analysis_manifest(a, path)
    assert frozen.frozen and len(frozen.manifest_hash) == 32
    restored = read_analysis_manifest(path)
    assert restored.frozen
    report = tmp_path / "report.md"
    result = write_reproducibility_report(restored, report)
    assert Path(result).exists()
    text = report.read_text(encoding="utf-8")
    assert "Data fingerprint" in text
    assert "Interpretation boundary" in text
