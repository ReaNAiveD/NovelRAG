"""Tool for writing relations between resources."""

import json
from typing import Annotated, Any

from pydantic import BaseModel, Field

from novelrag.agenturn.tool import SchematicTool, ToolRuntime
from novelrag.agenturn.tool.types import ToolOutput
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from novelrag.resource.aspect import ResourceAspect
from novelrag.resource.element import DirectiveElement
from novelrag.resource.repository import ResourceRepository
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm


class GetUpdatedRelationsResponse(BaseModel):
    """LLM response containing updated relation descriptions."""
    relations: Annotated[list[str], Field(
        description="Updated list of relation descriptions between the resources.",
    )]


class ResourceRelationWriteTool(SchematicTool):
    """Tool for writing relations between resources in the repository."""

    PACKAGE_NAME = "novelrag.resource_agent.tool"
    TEMPLATE_NAME = "get_updated_relations.jinja2"

    def __init__(self, repo: ResourceRepository, chat_llm: BaseChatModel, lang: str = "en", lang_directive: str = "", undo_queue: UndoQueue | None = None):
        self.repo = repo
        self.undo = undo_queue
        self._lang_directive = lang_directive
        self._relations_llm = chat_llm.with_structured_output(GetUpdatedRelationsResponse)
        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._template = template_env.load_template(self.TEMPLATE_NAME)

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str:
        return 'This tool is used to write relations between resources in the repository. ' \
                'It allows you to define relationships between resources, such as A do something with object B or A take part in event C. ' \
                'The tool will generate a proposal for the relationship and apply it to the repository.'

    @property
    def output_description(self) -> str | None:
        return 'Returns the updated resource with the new relationship applied.'

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_resource_uri": {
                    "type": "string",
                    "description": "The URI of the source resource to which the relation will be added."
                },
                "target_resource_uri": {
                    "type": "string",
                    "description": "The URI of the target resource to which the relation will be added."
                },
                "operation": {
                    "type": "string",
                    "description": "The operation to perform on the relation, e.g., 'add', 'remove', 'update'."
                },
                "relation_description": {
                    "type": "string",
                    "description": "A human-readable description of the relationship between the source and target resources."
                }
            },
            "required": ["source_resource_uri", "target_resource_uri", "operation", "relation_description"],
        }

    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        source_resource_uri = kwargs.get('source_resource_uri')
        target_resource_uri = kwargs.get('target_resource_uri')
        operation = kwargs.get('operation')
        relation_description = kwargs.get('relation_description')
        if not source_resource_uri or not target_resource_uri or not operation or not relation_description:
            await runtime.error("Missing required parameters: source_resource_uri, target_resource_uri, operation, relation_description.")
            return self.error("Missing required parameters: source_resource_uri, target_resource_uri, operation, relation_description.")
        source_resource = await self.repo.find_by_uri(source_resource_uri)
        target_resource = await self.repo.find_by_uri(target_resource_uri)
        if isinstance(target_resource, list):
            await runtime.error(f"Target resource URI '{target_resource_uri}' points to multiple resources. Please specify a single resource.")
            return self.error(f"Target resource URI '{target_resource_uri}' points to multiple resources. Please specify a single resource.")
        if not target_resource:
            await runtime.error(f"Target resource URI '{target_resource_uri}' not found in the repository.")
            return self.error(f"Target resource URI '{target_resource_uri}' not found in the repository.")
        if isinstance(source_resource, DirectiveElement):
            existing_relation = source_resource.relationships.get(target_resource_uri)
            updated_relations = await self.get_updated_relations(
                source_resource, target_resource, existing_relation or [], operation, relation_description
            )
            undo_relationships = await self.repo.update_relationships(source_resource_uri, target_resource_uri, updated_relations)
            if self.undo is not None:
                self.undo.add_undo_item(ReversibleAction(
                    method="update_relationships",
                    params={
                        "source_uri": source_resource_uri,
                        "target_uri": target_resource_uri,
                        "relations": undo_relationships
                    }
                ), clear_redo=True)
            await runtime.message(f"Updated relations for source resource '{source_resource_uri}' to target resource '{target_resource_uri}'.")
            if isinstance(target_resource, DirectiveElement):
                existing_relation = target_resource.relationships.get(source_resource_uri)
                updated_relations = await self.get_updated_relations(
                    target_resource, source_resource, existing_relation or [], operation, relation_description
                )
                undo_relationships = await self.repo.update_relationships(target_resource_uri, source_resource_uri, updated_relations)
                if self.undo is not None:
                    self.undo.add_undo_item(ReversibleAction(
                        method="update_relationships",
                        params={
                            "source_uri": target_resource_uri,
                            "target_uri": source_resource_uri,
                            "relations": undo_relationships
                        }
                    ), clear_redo=True)
                await runtime.message(f"Updated relations for target resource '{target_resource_uri}' to source resource '{source_resource_uri}'.")
            return self.result(json.dumps(source_resource.context_dict, ensure_ascii=False))
        else:
            await runtime.error(f"Source resource URI '{source_resource_uri}' does not point to a valid resource. Please check the URI.")
            return self.error(f"Source resource URI '{source_resource_uri}' does not point to a valid resource. Please check the URI.")

    @trace_llm("update_relations")
    async def get_updated_relations(self, source_resource: DirectiveElement, target_resource: DirectiveElement | ResourceAspect, existing_relation: list[str], operation: str, relation_description: str) -> list[str]:
        prompt = self._template.render(
            source_resource=source_resource.context_dict,
            target_resource=target_resource.context_dict,
            existing_relation=existing_relation,
            operation=operation,
            relation_description=relation_description,
        )
        response = await self._relations_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Generate the updated relations."),
        ])
        assert isinstance(response, GetUpdatedRelationsResponse)
        return response.relations
