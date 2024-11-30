from novelrag.core.exceptions import UnregiesteredModelError


class ModelRegistry:
    def __init__(self):
        self._models = {}

    def register(self, name: str):
        """Decorator to register actions for a specific aspect"""
        def decorator(model_cls):
            self._models[name] = model_cls
            return model_cls
        return decorator

    def __getitem__(self, name: str):
        if name not in self._models:
            raise UnregiesteredModelError(name)
        return self._models[name]


model_registry = ModelRegistry()


def register_model(name: str):
    def decorator(model_cls):
        model_registry.register(name)(model_cls)
        return model_cls
    return decorator


class AspectRegistry:
    """Registry to manage aspect-specific actions"""
    def __init__(self):
        self.actions = {}

    def register(self, name: str):
        """Decorator to register actions for a specific aspect"""
        def decorator(action_cls):
            self.actions[name] = action_cls
            action_cls._action_name = name
            return action_cls
        return decorator
