"""Helpers for inspecting the frozen R-to-Python parity ledger."""

from __future__ import annotations

import csv
import io
import json
from importlib.resources import files


def read_parity_manifest() -> list[dict[str, str]]:
    """Return all 458 frozen R exports and their Python port status."""
    resource = files("gp3bayespy._reference").joinpath("function_map.csv")
    text = resource.read_text(encoding="utf-8")
    with io.StringIO(text, newline="") as file_handle:
        return list(csv.DictReader(file_handle))


def reference_metadata() -> dict[str, object]:
    """Return immutable metadata for the gp3bayes 0.5.0 source reference."""
    resource = files("gp3bayespy._reference").joinpath("r_reference_metadata.json")
    with resource.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def parity_counts() -> dict[str, int]:
    """Summarise current statuses in the 458-export parity ledger."""
    counts: dict[str, int] = {}
    for row in read_parity_manifest():
        status = row["status"]
        counts[status] = counts.get(status, 0) + 1
    return counts
