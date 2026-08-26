"""gp3bayespy: contract-first Bayesian workflows in Python.

Frozen R reference: gp3bayes 0.5.0.
"""
from .backends import backend_capabilities
from .binary import (
    BinaryBackendSpecification,
    BinaryFit,
    BinaryModelSpecification,
    BinaryPrepared,
    BinaryPriorPredictiveCheck,
    BinarySimulation,
    check_binary_prior_predictive,
    diagnose_binary_fit,
    fit_binary_model,
    prepare_hierarchical_binary_data,
    simulate_hierarchical_binary_data,
    specify_binary_model,
    summarise_binary_posterior,
    translate_binary_model_to_brms,
)
from .contracts import ModelContract, create_model_contract
from .duration import (
    DurationBackendSpecification,
    DurationFit,
    DurationModelSpecification,
    DurationPrepared,
    DurationPriorPredictiveCheck,
    DurationSimulation,
    check_duration_prior_predictive,
    diagnose_duration_fit,
    fit_duration_model,
    prepare_hierarchical_duration_data,
    simulate_hierarchical_duration_data,
    specify_duration_model,
    summarise_duration_posterior,
    translate_duration_model_to_brms,
)
from .exceptions import BackendUnavailableError, GP3BayesError
from .parity import parity_counts, read_parity_manifest, reference_metadata
from .postfit_exploration import extract_posterior_draws
from .predictive import (
    Prediction,
    PredictionSupport,
    audit_prediction_support,
    create_prediction_grid,
    extract_expected_predictions,
    extract_linear_predictions,
    extract_posterior_predictions,
    predict_binary_probability,
    predict_duration,
    predict_model,
    prediction_support_table,
    prediction_table,
)
from .readiness import AuditCheck, ReadinessAudit, audit_model_readiness
from .specification import (
    ModelSpecification,
    PriorSpecification,
    build_model_formula,
    create_model_specification,
    create_prior_specification,
    validate_prior_specification,
)

__version__ = "0.1.0.dev0"
__r_reference_version__ = "0.5.0"
__r_reference_sha256__ = "537eb05f949de1bcc1d6f8234066f064597951ecfa9cbbdf938d0a895ce5dd8a"

__all__ = [
    "AuditCheck",
    "BackendUnavailableError",
    "BinaryBackendSpecification",
    "BinaryFit",
    "BinaryModelSpecification",
    "BinaryPrepared",
    "BinaryPriorPredictiveCheck",
    "BinarySimulation",
    "DurationBackendSpecification",
    "DurationFit",
    "DurationModelSpecification",
    "DurationPrepared",
    "DurationPriorPredictiveCheck",
    "DurationSimulation",
    "GP3BayesError",
    "ModelContract",
    "ModelSpecification",
    "PriorSpecification",
    "Prediction",
    "PredictionSupport",
    "ReadinessAudit",
    "audit_model_readiness",
    "backend_capabilities",
    "build_model_formula",
    "check_binary_prior_predictive",
    "check_duration_prior_predictive",
    "diagnose_binary_fit",
    "diagnose_duration_fit",
    "create_model_contract",
    "create_model_specification",
    "create_prior_specification",
    "fit_binary_model",
    "extract_posterior_draws",
    "fit_duration_model",
    "parity_counts",
    "prepare_hierarchical_binary_data",
    "prepare_hierarchical_duration_data",
    "audit_prediction_support",
    "create_prediction_grid",
    "extract_expected_predictions",
    "extract_linear_predictions",
    "extract_posterior_predictions",
    "predict_binary_probability",
    "predict_duration",
    "predict_model",
    "prediction_support_table",
    "prediction_table",
    "read_parity_manifest",
    "reference_metadata",
    "simulate_hierarchical_binary_data",
    "simulate_hierarchical_duration_data",
    "specify_binary_model",
    "specify_duration_model",
    "summarise_binary_posterior",
    "summarise_duration_posterior",
    "translate_binary_model_to_brms",
    "translate_duration_model_to_brms",
    "validate_prior_specification",
]
