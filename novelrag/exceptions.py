class NovelRagError(Exception):
    """Base exception for NovelRag errors"""
    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self):
        return self.msg

class LLMError(NovelRagError):
    pass

class NoChatLLMConfigError(LLMError):
    def __init__(self, msg: str | None = None):
        super().__init__(msg or "Can not find available Chat LLM Config")

class NoEmbeddingConfigError(LLMError):
    def __init__(self, msg: str | None = None):
        super().__init__(msg or "Can not find available Embedding LLM Config")

class IntentError(NovelRagError):
    pass

class SessionError(NovelRagError):
    """Base exception for shell errors"""
    pass

class AspectError(NovelRagError):
    pass

class OperationError(NovelRagError):
    pass


class SessionQuitError(SessionError):
    def __init__(self):
        super().__init__("Quit Current Session.")

class AspectNotFoundError(SessionError):
    """Raised when aspect is not found"""
    def __init__(self, aspect_name: str, available_aspects: list[str]):
        self.aspect_name = aspect_name
        self.available_aspects = available_aspects
        super().__init__(
            f"Aspect '{aspect_name}' not found.\n"
            f"Available aspects: {', '.join(f'@{name}' for name in available_aspects)}"
        )

class ActionNotFoundError(SessionError):
    """Raised when action is not found"""
    def __init__(self, action_name: str, aspect_name: str):
        self.action_name = action_name
        self.aspect_name = aspect_name
        super().__init__(f"Action '{action_name}' not found in aspect '{aspect_name}'")

class NoAspectSelectedError(SessionError):
    """Raised when trying to perform action without selecting an aspect"""
    def __init__(self):
        super().__init__("No aspect selected. Please select an aspect first using @aspect")

class ActionError(NovelRagError):
    """Base exception for action-related errors"""
    pass


class NoItemToSubmitError(ActionError):
    def __init__(self):
        super().__init__("There is no Pending Update Item Available")


class NoItemToUndoError(ActionError):
    def __init__(self):
        super().__init__("There is no Undo Action Available")


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
    def __init__(self, action: str, aspect: str, message: str, expected_format: str | None):
        self.action = action
        self.aspect = aspect
        self.message = message
        self.expected_format = expected_format
        super().__init__(
            f"Invalid message format for {aspect}.{action}: {message}\n"
            + f"Expected format: {expected_format}" if expected_format else ""
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

class UnrecognizedCommandError(ActionError):
    """Raised when a command is not recognized by an action"""
    def __init__(self, command: str, action: str | None = None, aspect: str | None = None):
        self.command = command
        self.action = action
        self.aspect = aspect
        context = f" in {aspect}.{action}" if aspect and action else ""
        super().__init__(
            f"Unrecognized command{context}: {command}"
        )

class InvalidLLMResponseFormatError(ActionError):
    """Raised when LLM response format is invalid"""
    def __init__(self, action: str, aspect: str, response: str, expected_format: str):
        self.action = action
        self.aspect = aspect
        self.response = response
        self.expected_format = expected_format
        super().__init__(
            f"Invalid LLM response format for {aspect}.{action}: {response}\n"
            f"Expected format: {expected_format}"
        )

class IntentNotFoundError(IntentError):
    """Raised when an intent is not found"""
    def __init__(self, intent: str):
        self.intent = intent
        super().__init__(f"Intent '{intent}' not found")


class IntentMissingNameError(IntentError):
    def __init__(self, msg: str | None = None):
        super().__init__(msg or "Intent missing name")


class InvalidIntentRegisterError(NovelRagError):
    def __init__(self, intent_cls: type):
        self.intent_cls = intent_cls
        super().__init__(f"Class '{intent_cls}' is not a valid UserIntent.")

class UnregisteredModelError(AspectError):
    """Raised when a model is not registered"""
    def __init__(self, model: str):
        self.model = model
        super().__init__(f"Model '{model}' not registered")

class ElementNotFoundError(OperationError):
    def __init__(self, element_uri: str):
        super().__init__(f'Element "{element_uri}" not found')

class ChildrenKeyNotFoundError(OperationError):
    def __init__(self, key: str, aspect: str):
        super().__init__(f"Children Key '{key}' not found in Aspect '{aspect}'")

class PropertyNotFoundError(OperationError):
    def __init__(self, element_uri: str, path: str):
        super().__init__(f"Path {path} on Element '{element_uri}' not found")

class InvalidOperationError(OperationError):
    def __init__(self, operation: str, obj):
        super().__init__(f"Operation {operation} is invalid on Type {type(obj)}.")
