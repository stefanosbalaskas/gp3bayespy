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

## GPB-PY-04 restricted fitting checkpoint

The restricted binary and duration fitting layer is frozen against the R gp3bayes 0.5.0 fitting contracts. Four additional exports are now `implemented`: `translate_binary_model_to_brms`, `fit_binary_model`, `translate_duration_model_to_brms`, and `fit_duration_model`.

The public compatibility names are retained from R, while Python execution is an intentional backend adaptation to PyMC/NUTS. Fit objects report the actual Python backend truthfully and retain the R `brms`/`rstan` pathway only as source-provenance metadata. The API exposes only governed sampling controls and does not provide unrestricted formula, family, prior, algorithm, backend, or arbitrary keyword escape hatches.

A real Windows/Python 3.13.15 backend smoke completed both binary and duration NUTS fits. This demonstrates executable backend integration only; short smoke chains do **not** establish convergence, posterior adequacy, predictive validity, robustness, or substantive validity. Runtime evidence is frozen in `dev/parity/fitting_backend_validation_0.1.0.dev0.json`.

Current ledger counts: 18 `implemented`, 1 `implemented_initial`, 439 `mapped_not_implemented` = 458 total exports. `backend_capabilities` remains `implemented_initial` because the broader backend-capability contract has not yet been closed.

## GPB-PY-05 posterior foundation checkpoint

The backend-neutral posterior extraction, posterior summary, and sampling-diagnostic layer is frozen against the R gp3bayes 0.5.0 posterior contracts. Five additional exports are now `implemented`: `extract_posterior_draws`, `diagnose_binary_fit`, `summarise_binary_posterior`, `diagnose_duration_fit`, and `summarise_duration_posterior`.

The Python port preserves the R-facing posterior parameter naming convention while adapting storage to PyMC `InferenceData`, xarray, pandas, and NumPy. Rank-normalized R-hat, bulk ESS, tail ESS, divergences, treedepth saturation, and E-BFMI are assessed through the governed ArviZ/PyMC pathway. Posterior summaries remain descriptive and do not automatically establish convergence, posterior adequacy, predictive validity, robustness, or substantive validity.

A real Windows/Python 3.13.15 PyMC/ArviZ smoke completed both binary and duration posterior paths. The deliberately short two-chain, 50-draw smoke failed the prespecified diagnostic thresholds for both families; this is expected and confirms conservative reporting rather than a backend failure. Runtime evidence is frozen in `dev/parity/posterior_backend_validation_0.1.0.dev0.json`.

Current ledger counts: 23 `implemented`, 1 `implemented_initial`, 434 `mapped_not_implemented` = 458 total exports. `backend_capabilities` remains `implemented_initial` until its broader capability contract is separately closed.

## GPB-PY-06 predictive foundation checkpoint

The governed prediction layer is frozen against the R gp3bayes 0.5.0 `prediction-support.R` contracts. Ten additional exports are now `implemented`: `create_prediction_grid`, `audit_prediction_support`, `prediction_support_table`, `predict_model`, `prediction_table`, `extract_expected_predictions`, `extract_posterior_predictions`, `extract_linear_predictions`, `predict_binary_probability`, and `predict_duration`.

Expected-response, posterior-predictive, linear-predictor, and duration-median quantities remain distinct. Population-level predictions omit fitted grouping effects by default; grouping effects require explicit inclusion, and unseen grouping levels require explicit permission. Support auditing is advisory and never removes or rejects rows automatically. Prediction remains descriptive under the fitted model and does **not** establish causal effects or out-of-sample adequacy.

A real Windows/Python 3.13.15 PyMC/NUTS smoke completed binary and duration predictive paths with 25 draws over two prediction rows for each family. Runtime evidence is frozen against implementation commit `a8feb83` in `dev/parity/predictive_backend_validation_0.1.0.dev0.json`.

Current ledger counts: 33 `implemented`, 1 `implemented_initial`, 424 `mapped_not_implemented` = 458 total exports. `backend_capabilities` remains `implemented_initial` until its broader capability contract is separately closed.

