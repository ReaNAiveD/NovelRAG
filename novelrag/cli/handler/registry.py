import importlib

from novelrag.config.handler import HandlerConfig
from novelrag.cli.handler.handler import Handler


def build_handler(name: str, handler_config: HandlerConfig) -> tuple[str, Handler]:
    name = handler_config.name or name
    pkg, cls_name = handler_config.cls.rsplit('.', maxsplit=1)
    module = importlib.import_module(pkg)
    handler_cls = getattr(module, cls_name)
    handler = handler_cls(**handler_config.kwargs)
    return (name, handler)


class HandlerRegistry:
    def __init__(self, **kwargs: Handler):
        self.handlers = kwargs

    @classmethod
    def from_config(cls, config: dict[str, HandlerConfig]):
        handlers = dict(build_handler(name, conf) for name, conf in config.items())
        return cls(**handlers)

    async def get(self, name: str | None) -> Handler | None:
        if name is None:
            return self.handlers.get('_default')
        return self.handlers.get(name)
