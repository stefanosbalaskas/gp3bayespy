# Ecosystem update — September 2026

## Related crossed location–scale development

The related **gpbiometricspy** package now includes a fully exact-main-certified crossed participant–item Gaussian hierarchical location–scale model with **one location random slope for each crossed factor**.

This is a complementary modelling path, not a change to `gp3bayespy`. `gp3bayespy` retains its own Bayesian model families, frozen R-parity record, diagnostics, predictive checks and release evidence. The gpbiometricspy implementation is a separate frequentist/Laplace-approximation method for crossed conditional location and residual-scale heterogeneity.

### What was certified

PR #129 is pinned to gpbiometricspy merge SHA `d078e0366ace49c3ebeb2f6800bad6394d70631e`:

- 14/14 exact-main push workflow families green;
- 12/12 Ubuntu/macOS/Windows × Python 3.11–3.14 test lanes green;
- 782/782 tests;
- 14,015/14,015 statements;
- 6,757/6,776 raw branches = 99.7196%;
- 19 unchanged audited structural arcs, with 0 unexpected, 0 stale and 0 unaudited debt;
- frozen `gpbiometrics 2.0.0` parity unchanged at 406/406.

[Read the crossed participant–item random-slope guide](https://github.com/stefanosbalaskas/gpbiometricspy/blob/d078e0366ace49c3ebeb2f6800bad6394d70631e/docs/methods/crossed-random-slopes-location-scale.md)

[Open gpbiometricspy PR #129](https://github.com/stefanosbalaskas/gpbiometricspy/pull/129)

This note does not imply Bayesian equivalence, coefficient identity, or replacement of `gp3bayespy` workflows.
