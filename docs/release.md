---
title: gp3bayespy 0.6.1 release
description: Certified release record for gp3bayespy 0.6.1.
---

# gp3bayespy 0.6.1

`gp3bayespy 0.6.1` is the corrected contract-first Bayesian multilevel
gaze-mediation release. It is published on
[GitHub](https://github.com/stefanosbalaskas/gp3bayespy/releases/tag/v0.6.1)
and [PyPI](https://pypi.org/project/gp3bayespy/0.6.1/).

The GitHub-only `v0.6.0` attempt failed its exact-wheel mediation API smoke
test before PyPI publication. Its tag is intentionally preserved unchanged as
an audit record and was never published to PyPI.

## Certified release record

The released `v0.6.1` tree passed the following gates:

- **731 tests passed, 1 skipped** in the release validation suite;
- **12,872 / 12,872 statements covered**;
- **4,210 / 4,210 branches covered**;
- **100.00% statement and branch-aware coverage**;
- **0 missing statements, 0 missing branches, and 0 coverage exclusions**;
- Ruff and mypy;
- Ubuntu, macOS, and Windows CI across Python 3.11, 3.12, and 3.13;
- all-extras validation;
- all 13 canonical and mediation examples;
- strict MkDocs build and site audit;
- wheel and sdist build plus Twine validation;
- independent clean wheel and clean sdist installation;
- installed multilevel-mediation API verification;
- PyPI Trusted Publishing;
- exact GitHub/PyPI SHA-256 identity for the wheel and sdist;
- fresh name-based installation of `gp3bayespy==0.6.1` from the official PyPI index.

Release commit:
`6f69aa5c51b6c75dd1c5fe25cf6350a84783e1a7`.

## Version and archive provenance

The frozen R-parity baseline remains **gp3bayes 0.5.0** with **458 / 458**
canonical exports and **59 / 59** canonical articles represented in the Python
port. Version 0.6.1 adds **25 public mediation functions** beyond that frozen
parity ledger.

The DOI [10.5281/zenodo.22150746](https://doi.org/10.5281/zenodo.22150746)
identifies the archived **gp3bayespy 0.5.0** software record. It must not be
interpreted as a DOI for v0.6.1.

See [Citing gp3bayespy](citation.md) for version-aware citation guidance.
