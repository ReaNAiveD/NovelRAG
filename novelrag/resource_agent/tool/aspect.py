"""Tool for creating new aspects in the resource repository."""

import json
from typing import Any

from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.agenturn.tool.types import ToolOutput
from novelrag.llm import LLMMixin
from langchain_core.language_models import BaseChatModel
from novelrag.resource.repository import ResourceRepository
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue
from novelrag.template import TemplateEnvironment


class AspectCreateTool(LLMMixin, SchematicTool):
    """Tool for creating new aspects in the resource repository."""
    
    def __init__(self, repo: ResourceRepository, template_env: TemplateEnvironment, chat_llm: BaseChatModel, undo_queue: UndoQueue | None = None):
        self.repo = repo
        self.undo = undo_queue
        super().__init__(template_env=template_env, chat_llm=chat_llm)
    
    @property
    def name(self):
        return self.__class__.__name__
    
    @property
    def description(self):
        return "This tool is used to create new aspects in the resource repository. " \
                "It allows you to define the structure and metadata of a new aspect, including its name, path, and any additional fields required for your application." \
                "Before using this tool, you should have a clear understanding of all other aspects in the repository, " \
                "as the new aspect will be added to the existing structure."
    
    @property
    def output_description(self) -> str | None:
        return "Returns the newly created aspect's metadata, including its name, path, and any additional fields defined during creation."
    
    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the new aspect to create, e.g., 'character', 'location', 'item'."
                },
                "description": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of strings describing the aspect and what kind of resources it contains."
                }
            },
            "required": ["name", "description"],
        }

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        name = kwargs.get('name')
        if not name:
            await runtime.error("No aspect name provided. Please provide a name for the aspect.")
            return self.error("No aspect name provided. Please provide a name for the aspect.")

        await runtime.message(f"Initializing aspect metadata for '{name}'...")
        description = kwargs.get('description', [])
        aspect_metadata = await self.initialize_aspect_metadata(name, description)
        await runtime.debug(f"Aspect metadata initialized: {aspect_metadata}")

        aspect = self.repo.add_aspect(name, aspect_metadata)
        await runtime.message(f"Aspect '{name}' created successfully.")
        if self.undo:
            self.undo.add_undo_item(
                ReversibleAction(method="remove_aspect", params={"name": name}),
                clear_redo=True,
            )
        return self.result(json.dumps(aspect.context_dict, ensure_ascii=False))

    async def initialize_aspect_metadata(self, name: str, description: list[str]) -> dict[str, Any]:
        return json.loads(await self.call_template(
            'initialize_aspect_metadata.jinja2',
            json_format=True,
            aspect_name=name,
            aspect_description=description,
        ))
