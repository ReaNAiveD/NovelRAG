"""Tool for editing existing content in the resource repository."""

import json
import pydantic
from typing import Any

from novelrag.agenturn.tool import SchematicTool, ToolOutput
from novelrag.agenturn.procedure import ExecutionContext
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from novelrag.resource.repository import ResourceRepository
from novelrag.resource.operation import validate_op
from novelrag.resource_agent.backlog.types import Backlog, BacklogEntry
from novelrag.resource_agent.undo import ReversibleAction, UndoQueue
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm

from novelrag.resource_agent.tool.types import ContentGenerationTask
from novelrag.resource_agent.workspace import ResourceContext, ContextSnapshot
from novelrag.resource_agent.propose import ContentProposer, LLMContentProposer
from novelrag.resource_agent.pipeline import (
    ContentGenerationProcedure,
    CascadeUpdateProcedure,
    BacklogDiscoveryProcedure,
)


class ResourceWriteTool(SchematicTool):
    """Tool for editing existing content in the resource repository."""

    PACKAGE_NAME = "novelrag.resource_agent.tool"
    BUILD_OPERATION_TEMPLATE = "build_operation.jinja2"
    SUMMARIZE_TEMPLATE = "summarize_resource_write.jinja2"

    def __init__(self, repo: ResourceRepository, context: ResourceContext, chat_llm: BaseChatModel,
                 lang: str = "en", lang_directive: str = "", backlog: Backlog[BacklogEntry] | None = None, undo_queue: UndoQueue | None = None):
        self.context = context
        self.repo = repo
        self.undo = undo_queue
        self.chat_llm = chat_llm
        self._lang_directive = lang_directive

        content_proposers: list[ContentProposer] = [LLMContentProposer(chat_llm=chat_llm, lang=lang, lang_directive=lang_directive)]
        self._content_generation = ContentGenerationProcedure(
            content_proposers=content_proposers,
            chat_llm=chat_llm,
            context=context,
            lang=lang,
            lang_directive=lang_directive,
        )
        self._cascade_update = CascadeUpdateProcedure(
            repo=repo,
            chat_llm=chat_llm,
            context=context,
            lang=lang,
            lang_directive=lang_directive,
            undo_queue=undo_queue,
        )
        self._backlog_discovery = BacklogDiscoveryProcedure(
            chat_llm=chat_llm,
            context=context,
            lang=lang,
            lang_directive=lang_directive,
            backlog=backlog,
        )

        template_env = TemplateEnvironment(package_name=self.PACKAGE_NAME, default_lang=lang)
        self._build_operation_tmpl = template_env.load_template(self.BUILD_OPERATION_TEMPLATE)
        self._summarize_tmpl = template_env.load_template(self.SUMMARIZE_TEMPLATE)

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def prerequisites(self) -> str | None:
        return "Requires the target aspect to already exist in the repository. Use AspectCreateTool first if the aspect is missing."

    @property
    def description(self):
        return "Use this tool to modify existing resources within established aspects. " \
        "This tool plans and executes repository operations based on your specified intent. " \
        "Supports comprehensive operations including creating/updating/deleting resources, " \
        "modifying properties, splicing element lists (insert/remove/replace), " \
        "flattening hierarchies, and merging resources. " \
        "Automatically handles cascading updates to related resources and discovers follow-up work for your backlog."

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation_specification": {
                    "type": "string",
                    "description": "Detailed natural language description of the operation to be performed on existing aspects. Include target resource URIs (e.g., /character/john), specific fields to update, and any relationships to consider. Be specific about what exactly needs to be done, include all fields/properties that need updates, mention any relationships or dependencies, and use natural language but be precise."
                },
                "content_generation_tasks": {
                    "type": "array",
                    "description": "List of content generation tasks to perform as part of the operation. Break down into focused, specific tasks where each task should generate content for a clear purpose. All available context will be used for content generation.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": "What content to generate for this specific task. Should be focused and specific, generating content for a clear purpose."
                            },
                            'content_key': {
                                'type': 'string',
                                'description': 'Key identifier for organizing this content in the resource structure (optional). Used when building the final resource object from multiple generated content pieces.'
                            }
                        },
                        "required": ["description"]
                    }
                }
            },
            "required": ["operation_specification", "content_generation_tasks"]
        }

    async def call(self, ctx: ExecutionContext, **kwargs) -> ToolOutput:
        """Edit existing content using the new planning-based workflow."""
        operation_specification = kwargs.get('operation_specification')
        if not operation_specification:
            return self.error("No operation specification provided. Please provide a detailed description of the operation to perform.")
        content_generation_tasks = kwargs.get('content_generation_tasks', [])
        if not content_generation_tasks:
            return self.error("No content generation tasks provided. Please provide at least one content generation task.")
        content_generation_tasks = [ContentGenerationTask(**task) for task in content_generation_tasks]

        await ctx.output(f"Operation planned: {operation_specification}")
        await ctx.info(f"Content generation tasks: {len(content_generation_tasks)}")

        # Phase 1: Content generation
        content_results = await self._content_generation.execute(
            operation_specification, content_generation_tasks, ctx,
        )
        if not content_results:
            return self.error("No content was generated for any task.")
        await ctx.output(f"Generated content for {len(content_results)} tasks.")

        # Phase 2: Build and apply operations
        await ctx.info("Building operations from generated content...")
        try:
            operations = await self.build_operations(
                action=operation_specification,
                context=await self.context.snapshot(),
                content_results=content_results
            )
            operations = [validate_op(op) for op in operations]
        except pydantic.ValidationError as e:
            return self.error("Operation validation failed: " + str(e))

        await ctx.info("Operation validated successfully. Preparing to apply.")
        if not await ctx.confirm(f"Do you want to apply {len(operations)} operation(s)?\n{json.dumps([op.model_dump() for op in operations], indent=2, ensure_ascii=False)}"):
            await ctx.info("Operation application cancelled by user.")
            return self.result("Operation application cancelled by user.")
        undo_operations = [await self.repo.apply(op) for op in operations][::-1]
        await ctx.debug(f"Undo operations created: {undo_operations}")
        if self.undo is not None:
            for undo_op in undo_operations:
                await self.undo.add_undo_item(ReversibleAction(method="apply", params={"op": undo_op.model_dump()}), clear_redo=True)
        await ctx.output(f"Applied {len(operations)} operation(s) successfully.")

        applied_operations_data = [op.model_dump() for op in operations]
        undo_operations_data = [op.model_dump() for op in undo_operations]

        # Phase 3: Cascade updates
        perspective_updates_applied, relation_updates_applied = await self._cascade_update.execute(
            step_description=operation_specification,
            applied_operations=applied_operations_data,
            undo_operations=undo_operations_data,
            ctx=ctx,
        )

        # Phase 4: Backlog discovery
        backlog_count = await self._backlog_discovery.execute(
            step_description=operation_specification,
            applied_operations=applied_operations_data,
            undo_operations=undo_operations_data,
            ctx=ctx,
        )

        # Generate summary
        summary = await self._summarize_resource_write(
            operation_specification=operation_specification,
            applied_operations=applied_operations_data,
            undo_operations=undo_operations_data,
            perspective_updates_applied=perspective_updates_applied,
            relation_updates_applied=relation_updates_applied,
            backlog_count=backlog_count,
        )
        return self.result(summary)

    @trace_llm("build_operations")
    async def build_operations(self, action: str, context: ContextSnapshot, content_results: list[dict] | None = None) -> list[dict]:
        """Build operations from action description and optional content results."""
        prompt = self._build_operation_tmpl.render(
            action=action,
            context=context,
            content_results=content_results or [],
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Generate the operations in JSON format."),
        ], response_format={"type": "json_object"})
        assert isinstance(response.content, str), "Expected string content from LLM response"
        return json.loads(response.content)["operations"]

    @trace_llm("summarize_resource_write")
    async def _summarize_resource_write(
        self,
        operation_specification: str,
        applied_operations: list[dict],
        undo_operations: list[dict],
        perspective_updates_applied: list[dict],
        relation_updates_applied: list[dict],
        backlog_count: int,
    ) -> str:
        """Produce a concise natural-language summary of the resource write execution."""
        prompt = self._summarize_tmpl.render(
            operation_specification=operation_specification,
            applied_operations=applied_operations,
            undo_operations=undo_operations,
            perspective_updates_applied=perspective_updates_applied,
            relation_updates_applied=relation_updates_applied,
            backlog_count=backlog_count,
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Write the summary."),
        ])
        assert isinstance(response.content, str), "Expected string content from LLM response"
        return response.content
