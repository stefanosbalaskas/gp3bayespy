import gp3bayespy as g


def test_packaged_reference_manifest():
    rows = g.read_parity_manifest()
    assert len(rows) == 458
    assert rows[0]["r_export"] == "advanced_pupil_trajectory_table"
    assert g.reference_metadata()["source_sha256"] == g.__r_reference_sha256__
    counts = g.parity_counts()
    assert sum(counts.values()) == 458
    assert counts["implemented"] >= 5
