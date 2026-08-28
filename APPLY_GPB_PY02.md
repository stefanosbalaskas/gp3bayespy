# Apply GPB-PY-02 binary foundation patch

This patch starts GPB-PY-02 from the closed GPB-PY-01 state.

It adds the Python-native port of the frozen gp3bayes 0.5.0 binary workflow foundation:

- `simulate_hierarchical_binary_data`
- `prepare_hierarchical_binary_data`
- `specify_binary_model`
- `check_binary_prior_predictive`

It also adds R-derived structural/signature fixtures and tests.

The parity ledger is deliberately **not** promoted by this patch. Promote the four exports only after the patch passes the user's full Windows quality gate.

Recommended branch sequence from a clean `feature/gpb-py-01-core-parity` at `fba2811`:

```powershell
git switch main
git merge --ff-only feature/gpb-py-01-core-parity
git push origin main
git switch -c feature/gpb-py-02-binary-foundation
```

Then copy the patch files over the repository and run:

```powershell
uv run ruff check .
uv run mypy src/gp3bayespy
uv run pytest
uv run python -m build
```

Expected test count from the assembled patch: 74 tests.

The development container used to assemble this patch had pytest but not the `build`, `ruff`, or `mypy` executables, so those three gates must be confirmed in the user's standard Windows/uv environment before any ledger promotion.
