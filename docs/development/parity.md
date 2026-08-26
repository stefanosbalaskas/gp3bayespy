# R → Python parity

The frozen R reference is gp3bayes 0.5.0. `dev/parity/function_map.csv` contains every exported R function, its exact raw R signature, source file, source line, help file, proposed Python module, and current implementation status.

Parity is classified by exact structural parity, numerical tolerance parity, stochastic/distributional parity, semantic parity, or documented intentional Python divergence. Identical MCMC draws are not required across independent backends.
