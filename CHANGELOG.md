# Changelog

## 0.6.1 — 2026-09-21

- Publish the corrected multilevel gaze-mediation release after the withdrawn GitHub-only v0.6.0 attempt.
- Include the complete PR #14 mediation implementation, top-level API exports, documentation, examples, and scientific contracts.
- Preserve repository-wide 100% branch-aware test coverage with zero missing statements, zero missing branches, and zero coverage exclusions.
- Preserve explicit missingness, estimability, convergence, provenance, family-selection, sensitivity, simulation/recovery, and reporting contracts.
- Validate all canonical and mediation examples in the release environment.
- Validate the exact wheel and sdist independently in clean environments before tagging.
- Retain the immutable v0.6.0 tag as the withdrawn audit record; v0.6.0 was never published to PyPI.

## 0.6.0 — 2026-09-21 (withdrawn GitHub-only release)

- Add contract-first Bayesian multilevel gaze mediation.
- Separate within- and between-participant pathways with explicit estimability checks.
- Add likelihood-family, prior, missingness, convergence, serial, moderated, posterior-effect, sensitivity, PSIS-LOO, simulation/recovery, plotting, and reporting contracts.
- Preserve the frozen gp3bayes 0.5.0 parity baseline while extending the Python API beyond frozen parity.
- Require repository-wide 100% branch-aware coverage with zero exclusions.
- Harden exact release-artifact validation and PyPI Trusted Publishing.
- Replace the placeholder license file with the complete MIT License.

## 0.5.0 — 2026-08-28

- First public Python release aligned with the frozen `gp3bayes` 0.5.0 reference.
- Final deep-freeze validation: 458/458 exports implemented, 59/59 articles ported, 321/321 tests passing, and no public unrestricted `**kwargs`.
- Completion-candidate parity closure: 458/458 frozen exports materialized and promoted.
- Materialize 59/59 Python-facing articles and eight executable workflow examples.
- Add advanced predictive, LOO, sensitivity, reproducibility/reporting and full pupillometry families.
- Add final closure tests, flattened `all` extra, cross-platform Ruff/mypy/test/build CI and strict docs build.

- Freeze gp3bayes 0.5.0 as the initial R parity reference.
- Extract a complete 458-export function/signature/help/source ledger and 230 S3 registrations.
- Add backend-independent Python contract, readiness, formula, prior, and model-specification foundation.
- Add backend capability discovery, tests, packaging metadata, CI, docs scaffold, and citation metadata.
