# Bootstrap validation

Reference: `gp3bayes` 0.5.0 CRAN source archive.

## Frozen inventory

- 458 exported public functions
- 230 S3 registrations
- 60 R source files
- 465 Rd documentation files
- 59 R Markdown vignettes
- 54 `tests/testthat/test-*.R` case files
- 55 R test files when `tests/testthat.R` is included
- source SHA-256: `537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a`

## Python bootstrap validation

- Python: 3.13.5 in the build environment
- `compileall`: PASS
- pytest: 14 passed
- wheel build: PASS
- wheel reinstall/import smoke test: PASS
- packaged parity ledger: 458/458 entries readable after installation
- source distribution build: PASS
- core remains backend-independent
- optional backend discovery in this environment: PyMC available, ArviZ available, CmdStanPy absent, NumPyro absent

The readiness audit is intentionally tagged `implemented_initial`: blocking core checks are present, while exact warning-level edge-case parity with the 1,841-line R readiness implementation remains an open tranche. No unimplemented export is represented as complete.
