from logging import Logger
from typing import Protocol

from novelrag.agent.types import AgentMessageLevel


class AgentChannel(Protocol):
    async def send_message(self, content: str, level: AgentMessageLevel = AgentMessageLevel.INFO) -> None:
        """Send a message to the user."""
        pass

    async def message(self, content: str):
        """Send a message to the user."""
        return await self.send_message(content, AgentMessageLevel.INFO)
    
    async def debug(self, content: str):
        """Send a debug message to the user."""
        return await self.send_message(content, AgentMessageLevel.DEBUG)
    
    async def warning(self, content: str):
        """Send a warning message to the user."""
        return await self.send_message(content, AgentMessageLevel.WARNING)
    
    async def error(self, content: str):
        """Send an error message to the user."""
        return await self.send_message(content, AgentMessageLevel.ERROR)
    
    async def output(self, content: str):
        """
        Send an output to the user.
        Note: This is usually different from a message, 
        """
        ...
    
    async def confirm(self, prompt: str) -> bool:
        """Ask the user to confirm an action."""
        ...

    async def request(self, prompt: str) -> str:
        """Request input from the user."""
        ...


class SessionChannel(AgentChannel):
    def __init__(self) -> None:
        super().__init__()
        self.output_buffer: list[str] = []
    
    async def output(self, content: str):
        self.output_buffer.append(content)

    def get_output(self):
        output = self.output_buffer
        self.output_buffer = []
        return output


class ShellSessionChannel(SessionChannel):
    def __init__(self, logger: Logger) -> None:
        super().__init__()
        self.logger = logger

    async def send_message(self, content: str, level: AgentMessageLevel = AgentMessageLevel.INFO) -> None:
        match level:
            case AgentMessageLevel.DEBUG:
                self.logger.debug(content)
            case AgentMessageLevel.INFO:
                self.logger.info(content)
            case AgentMessageLevel.WARNING:
                self.logger.warning(content)
            case AgentMessageLevel.ERROR:
                self.logger.error(content)
            case _:
                self.logger.info(content)

    async def confirm(self, prompt: str) -> bool:
        result = input(prompt + 'y/N')
        if result.lower() in ['y', 'yes']:
            return True
        return False
    
    async def request(self, prompt: str) -> str:
        return input(prompt)
