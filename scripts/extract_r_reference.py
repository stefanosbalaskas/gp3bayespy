"""Regenerate gp3bayes 0.5.0 reference metadata from a CRAN source tree.

The committed parity ledger is authoritative for this port. This script is a
small audit helper; changes to the ledger require explicit review.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def exports_from_namespace(path: Path) -> list[str]:
    """Extract exported names from an R NAMESPACE file."""
    exports: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("export("):
            exports.append(line[7:-1])
    return exports


def main() -> None:
    """Validate the frozen export count for an extracted gp3bayes source tree."""
    parser = argparse.ArgumentParser()
    parser.add_argument("source_tree", type=Path)
    args = parser.parse_args()
    exports = exports_from_namespace(args.source_tree / "NAMESPACE")
    print(f"exports={len(exports)}")
    if len(exports) != 458:
        raise SystemExit("Reference drift: expected 458 exports for gp3bayes 0.5.0")


if __name__ == "__main__":
    main()
