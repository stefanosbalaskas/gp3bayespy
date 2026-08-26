# gp3bayespy final mypy cast patch

Replace this file in your project:

- `src/gp3bayespy/readiness.py`

Then run:

```powershell
uv run mypy src/gp3bayespy
uv run ruff check .
uv run pytest
uv run python -m build
```

This patch only makes the DataFrame type explicit for pandas-stubs. Runtime behavior is unchanged.
