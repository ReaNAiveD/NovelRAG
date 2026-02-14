from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

from novelrag.agenturn.goal import Goal
from novelrag.agenturn.pursuit import PursuitAssessment
from novelrag.agenturn.tool.schematic import SchematicTool
from novelrag.resource_agent.action_determine.action_determine_loop import ContextAnalyser, RefinementPlan
from novelrag.resource_agent.workspace import SegmentData
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm

class LLMContextAnalyzer(ContextAnalyser):
    TEMPLATE_NAME = "context_relevance.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en"):
        self.chat_llm = chat_llm.with_structured_output(RefinementPlan)
        template_env = TemplateEnvironment(package_name="novelrag.resource_agent.action_determine", default_lang=lang)
        self.template = template_env.load_template(self.TEMPLATE_NAME, lang=lang)

    @trace_llm("context_analysis")
    async def analyse(
        self,
        goal: Goal,
        pursuit_assessment: PursuitAssessment,
        workspace_segment: list[SegmentData],
        expanded_tools: dict[str, SchematicTool],
        collapsed_tools: dict[str, SchematicTool],
        discovery_analysis: str
    ) -> RefinementPlan:
        prompt = self.template.render(
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            workspace_segment=workspace_segment,
            expanded_tools=expanded_tools,
            collapsed_tools=collapsed_tools,
            discovery_analysis=discovery_analysis
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Based on the above information, analyze the relevance and utility of the discovered contexts for supporting progress toward the goal, and determine how to refine the resource context for action planning.")
        ])
        assert isinstance(response, RefinementPlan), "Expected RefinementPlan from LLM response"
        return response
