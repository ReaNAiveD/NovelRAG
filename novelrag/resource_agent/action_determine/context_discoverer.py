from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from novelrag.agenturn.goal import Goal
from novelrag.agenturn.pursuit import PursuitAssessment
from novelrag.agenturn.tool.schematic import SchematicTool
from novelrag.resource_agent.action_determine.action_determine_loop import ContextDiscoverer, DiscoveryPlan
from novelrag.resource_agent.workspace import SegmentData, SearchHistoryItem
from novelrag.template import TemplateEnvironment
from novelrag.tracer import trace_llm


class LLMContextDiscoverer(ContextDiscoverer):
    TEMPLATE_NAME = "context_discovery.jinja2"

    def __init__(self, chat_llm: BaseChatModel, lang: str = "en"):
        self.chat_llm = chat_llm.with_structured_output(DiscoveryPlan)
        template_env = TemplateEnvironment(package_name="novelrag.resource_agent.action_determine", default_lang=lang)
        self.template = template_env.load_template(self.TEMPLATE_NAME, lang=lang)

    @trace_llm("context_discovery")
    async def discover(
            self,
            goal: Goal,
            pursuit_assessment: PursuitAssessment,
            workspace_segment: list[SegmentData],
            non_existed_uris: list[str],
            search_history: list[SearchHistoryItem],
            expanded_tools: dict[str, SchematicTool],
            collapsed_tools: dict[str, SchematicTool],
    ) -> DiscoveryPlan:
        prompt = self.template.render(
            goal=goal,
            pursuit_assessment=pursuit_assessment,
            workspace_segment=workspace_segment,
            non_existed_uris=non_existed_uris,
            search_history=search_history,
            expanded_tools=expanded_tools,
            collapsed_tools=collapsed_tools,
        )
        response = await self.chat_llm.ainvoke([
            SystemMessage(content=prompt),
            HumanMessage(content="Based on the above information, determine the following resource contexts to discover that would best support progress toward the goal.")
        ])
        assert isinstance(response, DiscoveryPlan), "Expected DiscoveryPlan from LLM response"
        return response
