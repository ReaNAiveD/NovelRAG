import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from novelrag.agenturn.goal import Goal
from novelrag.agenturn.pursuit import PursuitAssessment
from novelrag.agenturn.step import OperationOutcome
from novelrag.agenturn.tool import SchematicTool
from novelrag.resource_agent.action_determine.action_determine_loop import (
    ActionDecider, ActionDecision, ExecutionDetail, FinalizationDetail,
)
from novelrag.resource_agent.workspace import SegmentData
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm

logger = logging.getLogger(__name__)


def _build_tool_defs(expanded_tools: dict[str, SchematicTool]) -> list[dict]:
    """Convert SchematicTools into OpenAI-format function tool definitions."""
    tool_defs = []
    for name, tool in expanded_tools.items():
        description = tool.description or ""
        if tool.prerequisites:
            description += f"\n\nPrerequisites: {tool.prerequisites}"
        if tool.output_description:
            description += f"\n\nExpected Output: {tool.output_description}"

        schema = dict(tool.input_schema) if tool.input_schema else {"type": "object", "properties": {}}

        tool_defs.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema,
            },
        })
    return tool_defs


# Finalization tool definition with a fixed schema (no bare dict fields)
_FINALIZE_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "Finalize",
        "description": (
            "Finalize the goal with a response. Use this when the goal is achieved (success), "
            "impossible (failed), or cannot be completed with available context (incomplete). "
            "Do NOT use this when a tool execution can still advance the goal."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Outcome status: 'success', 'failed', or 'incomplete'.",
                    "enum": ["success", "failed", "incomplete"],
                },
                "response": {
                    "type": "string",
                    "description": "Complete user-facing response explaining the outcome. "
                                   "Respond in the same language as the user's input. "
                                   "If the user writes in a specific language, provide your entire response "
                                   "in that language.",
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "References to specific segments with supporting information.",
                },
                "gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific missing information (only if status is incomplete).",
                },
            },
            "required": ["status", "response"],
            "additionalProperties": False,
        },
    },
}


def _parse_tool_call_to_action_decision(
        message: AIMessage,
        expanded_tool_names: set[str],
) -> ActionDecision:
    """Convert an AIMessage with tool_calls into an ActionDecision."""
    if not message.tool_calls:
        # No tool call — treat text content as a finalization
        content = message.content if isinstance(message.content, str) else str(message.content)
        return ActionDecision(
            situation_analysis=content or "LLM returned no tool call and no content.",
            decision_type="finalize",
            finalization=FinalizationDetail(
                status="incomplete",
                response=content or "Unable to determine action.",
            ),
        )

    call = message.tool_calls[0]
    tool_name = call["name"]
    tool_args = call["args"]

    # Extract situation_analysis from message content if available
    content = message.content if isinstance(message.content, str) else ""

    if tool_name == "Finalize":
        return ActionDecision(
            situation_analysis=content or "Finalizing goal.",
            decision_type="finalize",
            finalization=FinalizationDetail(
                status=tool_args.get("status", "incomplete"),
                response=tool_args.get("response", ""),
                evidence=tool_args.get("evidence", []),
                gaps=tool_args.get("gaps", []),
            ),
        )
    elif tool_name in expanded_tool_names:
        return ActionDecision(
            situation_analysis=content or f"Executing tool: {tool_name}",
            decision_type="execute",
            execution=ExecutionDetail(
                tool=tool_name,
                params=tool_args,
                confidence="high",
                reasoning=content or "",
            ),
        )
    else:
        logger.warning(f"LLM called unknown tool '{tool_name}', treating as finalization.")
        return ActionDecision(
            situation_analysis=content or f"Unknown tool called: {tool_name}",
            decision_type="finalize",
            finalization=FinalizationDetail(
                status="failed",
                response=f"Attempted to use unknown tool: {tool_name}",
            ),
        )


class LLMActionDecider(ActionDecider):
    TEMPLATE_NAME = "action_decision.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en", lang_directive: str = ""):
        self.chat_llm = chat_llm
        self._lang_directive = lang_directive
        template_env = TemplateEnvironment(package_name="novelrag.resource_agent.action_determine", default_lang=lang)
        self.template = template_env.load_template(self.TEMPLATE_NAME, lang=lang)

    @trace_llm("action_decision")
    async def decide(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            completed_steps: list[OperationOutcome],
            workspace_segment: list[SegmentData],
            expanded_tools: dict[str, SchematicTool],
    ) -> ActionDecision:
        prompt = self.template.render(
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            completed_steps=completed_steps,
            workspace_segment=workspace_segment,
            expanded_tools=expanded_tools,
        )

        # Build native tool definitions from expanded tools + finalize pseudo-tool
        tool_defs = _build_tool_defs(expanded_tools)
        tool_defs.append(_FINALIZE_TOOL_DEF)

        bound_llm = self.chat_llm.bind_tools(tool_defs)
        response = await bound_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Based on the above information, make a decisive action choice: either execute a tool or finalize with a response."),
        ])

        assert isinstance(response, AIMessage), f"Expected AIMessage, got {type(response)}"

        expanded_tool_names = set(expanded_tools.keys())
        try:
            return _parse_tool_call_to_action_decision(response, expanded_tool_names)
        except Exception as e:
            raise ValueError(f"Failed to parse LLM tool call into ActionDecision: {e}")
