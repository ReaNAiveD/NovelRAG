class NovelRagError(Exception):
    """Base exception for NovelRag errors"""
    pass

class ShellError(NovelRagError):
    """Base exception for shell errors"""
    pass

class AspectNotFoundError(ShellError):
    """Raised when aspect is not found"""
    def __init__(self, aspect_name: str, available_aspects: list[str]):
        self.aspect_name = aspect_name
        self.available_aspects = available_aspects
        super().__init__(
            f"Aspect '{aspect_name}' not found.\n"
            f"Available aspects: {', '.join(f'@{name}' for name in available_aspects)}"
        )

class ActionNotFoundError(ShellError):
    """Raised when action is not found"""
    def __init__(self, action_name: str, aspect_name: str):
        self.action_name = action_name
        self.aspect_name = aspect_name
        super().__init__(f"Action '{action_name}' not found in aspect '{aspect_name}'")

class NoAspectSelectedError(ShellError):
    """Raised when trying to perform action without selecting an aspect"""
    def __init__(self):
        super().__init__("No aspect selected. Please select an aspect first using @aspect")

class ActionError(NovelRagError):
    """Base exception for action-related errors"""
    pass

class InvalidIndexError(ActionError):
    """Raised when an index is invalid for any indexed aspect data"""
    def __init__(self, idx: int, max_idx: int, aspect_name: str):
        self.idx = idx
        self.max_idx = max_idx
        self.aspect_name = aspect_name
        super().__init__(
            f"Invalid {aspect_name} index {idx}. Index must be between 0 and {max_idx}"
        )

class InvalidMessageFormatError(ActionError):
    """Raised when action message format is invalid"""
    def __init__(self, action: str, aspect: str, message: str, expected_format: str):
        self.action = action
        self.aspect = aspect
        self.message = message
        self.expected_format = expected_format
        super().__init__(
            f"Invalid message format for {aspect}.{action}: {message}\n"
            f"Expected format: {expected_format}"
        )

class ActionNotSupportedError(ActionError):
    """Raised when an action doesn't support certain operations"""
    def __init__(self, action: str, aspect: str, operation: str):
        self.action = action
        self.aspect = aspect
        self.operation = operation
        super().__init__(
            f"Operation '{operation}' not supported for {aspect}.{action}"
        )

class DataValidationError(ActionError):
    """Raised when aspect data fails validation"""
    def __init__(self, aspect: str, details: str):
        self.aspect = aspect
        self.details = details
        super().__init__(
            f"Data validation failed for {aspect}: {details}"
        )

class UnrecognizedResultError(ActionError):
    """Raised when an action result type is not recognized"""
    def __init__(self, action: str | None, aspect: str | None, result_type: str):
        self.action = action
        self.aspect = aspect
        self.result_type = result_type
        context = f" in {aspect}.{action}" if aspect and action else ""
        super().__init__(
            f"Unrecognized action result type{context}: {result_type}"
        )
  