from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from novelrag.agenturn.goal import Goal
from novelrag.agenturn.pursuit import PursuitAssessment
from novelrag.agenturn.step import OperationOutcome
from novelrag.agenturn.tool.schematic import SchematicTool
from novelrag.resource_agent.action_determine.action_determine_loop import (
    RefinementAnalyzer,
    ActionDecision,
    RefinementDecision,
)
from novelrag.resource_agent.workspace import SegmentData
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm


class LLMRefinementAnalyzer(RefinementAnalyzer):
    TEMPLATE_NAME = "refinement_analysis.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en", lang_directive: str = ""):
        self.chat_llm = chat_llm.with_structured_output(RefinementDecision)
        self._lang_directive = lang_directive
        template_env = TemplateEnvironment(package_name="novelrag.resource_agent.action_determine", default_lang=lang)
        self.template = template_env.load_template(self.TEMPLATE_NAME, lang=lang)

    @trace_llm("refinement_analysis")
    async def analyze(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            action_decision: ActionDecision,
            completed_steps: list[OperationOutcome],
            workspace_segment: list[SegmentData],
            expanded_tools: dict[str, SchematicTool],
            collapsed_tools: dict[str, SchematicTool],
    ) -> RefinementDecision:
        prompt = self.template.render(
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            action_decision=action_decision,
            completed_steps=completed_steps,
            workspace_segment=workspace_segment,
            expanded_tools=expanded_tools,
            collapsed_tools=collapsed_tools,
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=f"{self._lang_directive}\n\n{prompt}" if self._lang_directive else prompt),
            HumanMessage(content="Based on the above information, analyze the action decision and determine whether to approve execution or refine the approach."),
        ])
        assert isinstance(response, RefinementDecision), "Expected RefinementDecision from LLM response"
        return response
