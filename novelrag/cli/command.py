from dataclasses import dataclass


@dataclass
class Command:
    raw: str | None = None
    handler: str | None = None
    message: str | None = None
    is_redirect: bool = False
    redirect_source: str | None = None

    @property
    def text(self):
        parts = []
        if self.handler:
            parts.append(f"/{self.handler}")
        if self.message:
            parts.append(self.message)
        return " ".join(parts)
