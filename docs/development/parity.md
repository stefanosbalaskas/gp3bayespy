# R → Python parity

The frozen R reference is gp3bayes 0.5.0. `dev/parity/function_map.csv` contains every exported R function, its exact raw R signature, source file, source line, help file, proposed Python module, and current implementation status.

Parity is classified by exact structural parity, numerical tolerance parity, stochastic/distributional parity, semantic parity, or documented intentional Python divergence. Identical MCMC draws are not required across independent backends.

## GPB-PY-01 core parity checkpoint

The contract/readiness/specification core is frozen against R gp3bayes 0.5.0 structural expectations. Six exports are now `implemented`: `create_model_contract`, `audit_model_readiness`, `build_model_formula`, `create_prior_specification`, `validate_prior_specification`, and `create_model_specification`. `backend_capabilities` remains `implemented_initial` until backend/environment-specific parity is tested.

Current ledger counts: 6 `implemented`, 1 `implemented_initial`, 451 `mapped_not_implemented` = 458 total exports. The fixture `dev/parity/core_reference_cases_0.5.0.json` records its R-derived, non-runtime-captured provenance explicitly.
