"""Package-specific exceptions."""


class GP3BayesError(ValueError):
    """Base exception for contract or governance violations."""


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested optional backend is unavailable."""
