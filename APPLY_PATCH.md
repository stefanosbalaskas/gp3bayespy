# GPB-PY-01 readiness typing/compatibility fix

This is a replacement for the previous GPB-PY-01 readiness parity patch.

It preserves the readiness semantics and parity tests while fixing:
- Ruff UP035 (`Callable` moved to `collections.abc`)
- pandas-stubs DataFrame/Series ambiguity via explicit DataFrame/Series casts
- `Mapping` vs `dict` annotation mismatch with `ModelContract.mappings`
- optional mapping narrowing for required outcome/participant mappings
- deprecated `is_categorical_dtype` calls, replaced with `CategoricalDtype` checks

Apply the two files over the repository and run:

```powershell
uv run ruff check .
uv run mypy src/gp3bayespy
uv run pytest
uv run python -m build
```

Do not promote the parity ledger until all four gates pass locally.
