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
