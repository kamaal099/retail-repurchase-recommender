"""Project-specific exceptions so failures are explicit and catchable."""


class RetailMLError(Exception):
    """Base class for all project errors."""


class DataNotFoundError(RetailMLError):
    """Raised when no dataset file is found in the data directory."""


class DataValidationError(RetailMLError):
    """Raised when the data is missing columns or is unusable."""


class ModelTrainingError(RetailMLError):
    """Raised when a model fails to train."""


class NotFittedError(RetailMLError):
    """Raised when a model is used before fit()."""
