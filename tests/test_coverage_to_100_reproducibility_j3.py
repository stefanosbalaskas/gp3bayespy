from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3bayespy.reproducibility as r
from gp3bayespy.exceptions import GP3BayesError


class SlotOnly:
    __slots__ = ()


def test_repro_stable_fingerprint_signature_and_versions(monkeypatch):
    assert r._stable(float("nan")) is None
    assert r._stable(np.int64(3)) == 3
    assert r._stable(np.array([1, 2])) == [1, 2]
    assert r._stable({"b": 2, "a": 1}) == {"a": 1, "b": 2}
    assert r._stable((1, 2)) == [1, 2]
    assert isinstance(r._stable(SlotOnly()), str)

    frame = pd.DataFrame({"x": [1, 2]})
    fingerprint = r._data_fingerprint(frame)
    assert fingerprint["available"]
    assert r._data_fingerprint(None)["available"] is False
    with pytest.raises(GP3BayesError):
        r._data_fingerprint([1, 2])  # type: ignore[arg-type]

    assert r._signature(None)["available"] is False
    assert r._signature({"x": 1})["available"] is True

    original = r.version

    def missing(package: str):
        if package == "gp3bayespy":
            raise r.PackageNotFoundError(package)
        return original(package)

    monkeypatch.setattr(r, "version", missing)
    versions = r._versions()
    assert "gp3bayespy" not in versions


def test_manifest_validation_freeze_compare_and_files(tmp_path):
    with pytest.raises(GP3BayesError):
        r.create_analysis_manifest(fit=SimpleNamespace(family="pupil"))
    with pytest.raises(GP3BayesError):
        r.create_analysis_manifest(specification=SimpleNamespace(family="pupil"))
    with pytest.raises(GP3BayesError):
        r.create_analysis_manifest(seed=-1)
    with pytest.raises(GP3BayesError):
        r.create_analysis_manifest(label="")
    with pytest.raises(GP3BayesError):
        r.create_analysis_manifest(notes=("",))

    m1 = r.create_analysis_manifest(
        data=pd.DataFrame({"x": [1, 2]}),
        estimands=("a",),
        seed=1,
        label="one",
        notes=("note",),
    )
    assert r.validate_analysis_manifest(m1).status == "pass"
    assert r.validate_analysis_manifest(object()).status == "fail"  # type: ignore[arg-type]
    with pytest.raises(GP3BayesError):
        r.validate_analysis_manifest(object(), strict=True)  # type: ignore[arg-type]

    broken = r.create_analysis_manifest(data=pd.DataFrame({"x": [1]}))
    broken.data = {**dict(broken.data), "hash": None}
    assert r.validate_analysis_manifest(broken).status == "fail"
    with pytest.raises(GP3BayesError):
        r.validate_analysis_manifest(broken, strict=True)

    family_bad = r.create_analysis_manifest()
    family_bad.family = "mystery"
    assert r.validate_analysis_manifest(family_bad).status == "fail"

    path = tmp_path / "manifest.pkl"
    frozen = r.freeze_analysis_manifest(m1, path)
    assert frozen.frozen and frozen.manifest_hash
    with pytest.raises(GP3BayesError):
        r.freeze_analysis_manifest(m1, path)
    r.freeze_analysis_manifest(m1, path, overwrite=True)
    loaded = r.read_analysis_manifest(path)
    assert loaded.manifest_hash == m1.manifest_hash
    with pytest.raises(GP3BayesError):
        r.read_analysis_manifest(tmp_path / "missing.pkl")
    with pytest.raises(GP3BayesError):
        r.freeze_analysis_manifest(
            r.create_analysis_manifest(),
            tmp_path / "missing-dir" / "m.pkl",
        )

    m2 = r.create_analysis_manifest(
        data=pd.DataFrame({"x": [2, 1]}),
        estimands=("b",),
        seed=2,
        label="two",
    )
    comparison = r.compare_analysis_manifests(m1, m2)
    assert not comparison.identical
    assert comparison.changed_components
    assert len(r.analysis_manifest_table(m1)) == 5

    report = tmp_path / "report.md"
    returned = r.write_reproducibility_report(m1, report)
    assert returned == str(report.resolve())
    assert "gp3bayes reproducibility report" in report.read_text(encoding="utf-8")
    with pytest.raises(GP3BayesError):
        r.write_reproducibility_report(m1, report)
    r.write_reproducibility_report(m1, report, overwrite=True)
    with pytest.raises(GP3BayesError):
        r.write_reproducibility_report(
            m1,
            tmp_path / "missing-report-dir" / "report.md",
        )
