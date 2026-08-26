# R → Python parity

The frozen R reference is gp3bayes 0.5.0. `dev/parity/function_map.csv` contains every exported R function, its exact raw R signature, source file, source line, help file, proposed Python module, and current implementation status.

Parity is classified by exact structural parity, numerical tolerance parity, stochastic/distributional parity, semantic parity, or documented intentional Python divergence. Identical MCMC draws are not required across independent backends.

## GPB-PY-01 core parity checkpoint

The contract/readiness/specification core is frozen against R gp3bayes 0.5.0 structural expectations. Six exports are now `implemented`: `create_model_contract`, `audit_model_readiness`, `build_model_formula`, `create_prior_specification`, `validate_prior_specification`, and `create_model_specification`. `backend_capabilities` remains `implemented_initial` until backend/environment-specific parity is tested.

Current ledger counts: 6 `implemented`, 1 `implemented_initial`, 451 `mapped_not_implemented` = 458 total exports. The fixture `dev/parity/core_reference_cases_0.5.0.json` records its R-derived, non-runtime-captured provenance explicitly.


## GPB-PY-02 binary foundation checkpoint

The backend-independent binary workflow foundation is frozen against R gp3bayes 0.5.0 source and test expectations. Four additional exports are now `implemented`: `simulate_hierarchical_binary_data`, `prepare_hierarchical_binary_data`, `specify_binary_model`, and `check_binary_prior_predictive`.

Simulation and prior-predictive parity are stochastic/semantic rather than bit-identical because R and NumPy use different random-number generators. Preparation and model-specification parity are structural/deterministic where the R contract permits.

Current ledger counts: 10 `implemented`, 1 `implemented_initial`, 447 `mapped_not_implemented` = 458 total exports. `backend_capabilities` remains `implemented_initial`.

The fixture `dev/parity/binary_foundation_reference_0.5.0.json` records the frozen signatures, reference provenance, and parity classification for this tranche.

## GPB-PY-03 duration foundation checkpoint

The backend-independent duration workflow foundation is frozen against R gp3bayes 0.5.0 source and test expectations. Four additional exports are now `implemented`: `simulate_hierarchical_duration_data`, `prepare_hierarchical_duration_data`, `specify_duration_model`, and `check_duration_prior_predictive`.

Duration preparation preserves explicit source/analysis-unit provenance, strictly positive finite outcomes, one-based dropped-row provenance, and R-style sample-SD predictor scaling. Simulation and prior-predictive parity are stochastic/semantic rather than bit-identical because R and NumPy use different random-number generators.

Current ledger counts: 14 `implemented`, 1 `implemented_initial`, 443 `mapped_not_implemented` = 458 total exports. `backend_capabilities` remains `implemented_initial`.

The fixture `dev/parity/duration_foundation_reference_0.5.0.json` records the frozen signatures, reference provenance, and parity classification for this tranche.

