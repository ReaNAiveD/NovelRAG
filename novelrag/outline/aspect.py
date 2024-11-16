from novelrag.aspect import AspectContext


class OutlineAspect(AspectContext):
    def __init__(self, name: str):
        super().__init__(name)

    @property
    def system_context(self):
        return super().system_context

    @property
    def tasks(self):
        return super().tasks
