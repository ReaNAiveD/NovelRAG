from dataclasses import dataclass


@dataclass
class Command:
    redirect_from: str | None = None
    raw: str | None = None
    aspect: str | None = None
    intent: str | None = None
    message: str | None = None

    @property
    def text(self):
        parts = []
        if self.aspect:
            parts.append(f"@{self.aspect}")
        if self.intent:
            parts.append(f"/{self.intent}")
        if self.message:
            parts.append(self.message)
        return " ".join(parts)
