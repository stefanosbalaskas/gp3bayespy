# Release validation — gp3bayespy 0.5.0

Reference: frozen `gp3bayes` 0.5.0 CRAN source archive.

## Frozen inventory

- 458 exported public functions
- 230 S3 registrations
- 60 R source files
- 465 Rd documentation files
- 59 R Markdown vignettes
- 54 `tests/testthat/test-*.R` case files
- source SHA-256: `537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a`

## Final closure state

- parity ledger: 458 `implemented`, 0 `implemented_initial`, 0 `mapped_not_implemented`
- root namespace: all 458 frozen exports importable
- public frozen API: no unrestricted `**kwargs` catchalls
- Python-facing articles: 59/59 materialized
- executable examples: 8/8 materialized and exercised
- local regression suite: 321/321 PASS
- `compileall`: PASS
- wheel build: PASS
- source distribution build: PASS
- isolated wheel-target import/example smoke: PASS

## Static and cross-platform gates

The frozen source and metadata commits passed the authoritative GitHub Actions matrix before the release was finalized:

- Ruff
- mypy
- pytest on Linux/Windows/macOS and Python 3.11/3.12/3.13
- sdist/wheel build
- eight executable examples
- an Ubuntu/Python 3.13 `.[all]` installation
- strict MkDocs build

The final deep-freeze record is `dev/parity/final_deep_freeze_0.5.0.json`. The frozen source commit is `d9fa3cdaa0ac261cd5b3cb39175d3896d9c8a1c7`, and the freeze metadata commit is `791704a0897d3677b8c6d7eb34f8f0aa284a85db`.

## Governance

No diagnostic, predictive score, sensitivity summary, ranking, pupil model, or comparison automatically establishes convergence, adequacy, robustness, causality, exclusion, preferred-model status, or psychological interpretation.
