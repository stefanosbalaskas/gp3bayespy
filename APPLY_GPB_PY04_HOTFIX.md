# GPB-PY-04 Windows integration hotfix

This hotfix addresses the first real PyMC/NUTS Windows smoke findings:

- Ruff UP033: use `functools.cache`.
- Mypy: avoid static optional-PyMC imports and do not force a Python 3.11 target while running under Python 3.13.
- SciPy optional typing: local ignore for `scipy.special`.
- PyTensor/Numba/NumPy runtime: cap the Bayesian extra at NumPy `<2.5` and Numba `0.64..0.66` for Python 3.12+, matching PyTensor 3.3.0's supported Numba line and avoiding the NumPy 2.5 removal of `np.row_stack`.
- Real smoke now prints the resolved backend matrix before sampling.

After copying the files, run `uv sync --extra bayes --extra dev`, then Ruff, mypy, pytest, build, and the real smoke. Commit `uv.lock` together with the hotfix because the corrected dependency matrix intentionally changes the lock.
