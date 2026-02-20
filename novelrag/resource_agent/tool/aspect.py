"""Tool for creating new aspects in the resource repository."""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.agenturn.tool.types import ToolOutput
from novelrag.resource.repository import ResourceRepository
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm
from novelrag.utils.language import schema_directive


class AspectCreateTool(SchematicTool):
    """Tool for creating new aspects in the resource repository."""

    PACKAGE_NAME = "novelrag.resource_agent.tool"
    TEMPLATE_NAME = "initialize_aspect_metadata.jinja2"

    def __init__(self, repo: ResourceRepository, chat_llm: BaseChatModel, lang: str = "en", lang_directive: str = "", undo_queue: UndoQueue | None = None):
        self.repo = repo
        self.undo = undo_queue
        self.chat_llm = chat_llm
        # For aspect schema, use schema_directive (keys English, descriptions in content lang)
        self._lang_directive = lang_directive
        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._template = template_env.load_template(self.TEMPLATE_NAME)
    
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

        await runtime.info(f"Initializing aspect metadata for '{name}'...")
        description = kwargs.get('description', [])
        aspect_metadata = await self.initialize_aspect_metadata(name, description)
        await runtime.debug(f"Aspect metadata initialized: {aspect_metadata}")

        aspect = self.repo.add_aspect(name, aspect_metadata)
        await runtime.info(f"Aspect '{name}' created successfully.")
        if self.undo:
            self.undo.add_undo_item(
                ReversibleAction(method="remove_aspect", params={"name": name}),
                clear_redo=True,
            )
        return self.result(json.dumps(aspect.context_dict, ensure_ascii=False))

    @trace_llm("aspect_metadata")
    async def initialize_aspect_metadata(self, name: str, description: list[str]) -> dict[str, Any]:
        prompt = self._template.render(
            aspect_name=name,
            aspect_description=description,
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content=f"Generate metadata for the aspect '{name}' based on the description provided."),
        ], response_format={"type": "json_object"})
        assert isinstance(response.content, str)
        
        return json.loads(response.content)
