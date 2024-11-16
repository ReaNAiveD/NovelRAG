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