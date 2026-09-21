# gp3bayespy 0.6.1

`gp3bayespy 0.6.1` is the corrected contract-first Bayesian
multilevel gaze-mediation release.

The GitHub-only `v0.6.0` attempt failed its exact-wheel mediation
API smoke test before PyPI publication. Its tag is intentionally
preserved unchanged as an audit record.

## Release gates

The 0.6.1 release requires:

- repository-wide 100% statement and branch-aware coverage;
- zero missing statements;
- zero missing branches;
- zero coverage exclusions;
- Ruff and mypy;
- Ubuntu, macOS, and Windows CI across supported Python versions;
- all-extras validation;
- all canonical and mediation examples;
- strict MkDocs/site validation;
- wheel and sdist build plus Twine validation;
- independent clean wheel and sdist installation;
- installed multilevel-mediation API verification;
- exact GitHub/PyPI artifact hash identity after publication.

The DOI `10.5281/zenodo.22150746` remains specific to the archived
gp3bayespy 0.5.0 release.
