from .agent import Agent
from .channel import SessionChannel
from .planning import PursuitPlanner
from .pursuit import PursuitSummarizer
from .resource_tools import ResourceFetchTool, ResourceSearchTool, ResourceWriteTool, AspectCreateTool, ResourceRelationWriteTool
from novelrag.intent import LLMIntent, IntentContext, Action
from novelrag.template import TemplateEnvironment


# TODO: Tools should be defined in config
class AgentIntent(LLMIntent):
    def __init__(self, *, name: str | None, chat_llm: dict | None = None, lang: str | None = None, channel: SessionChannel, **kwargs):
        super().__init__(name=name, chat_llm=chat_llm, lang=lang, **kwargs)
        self.template_env = TemplateEnvironment("novelrag.agent", "en")
        self.channel = channel

    @property
    def default_name(self) -> str | None:
        return 'agent'
    
    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        if message is None:
            raise ValueError("Message cannot be None for AgentIntent")

        tools = {}
        if context.resource_repository:
            fetch_tool = ResourceFetchTool(context.resource_repository)
            search_tool = ResourceSearchTool(context.resource_repository)
            writer_tool = ResourceWriteTool(context.resource_repository, template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
            aspect_create_tool = AspectCreateTool(context.resource_repository, template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
            relation_tool = ResourceRelationWriteTool(context.resource_repository, template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
            tools['resource_fetch'] = fetch_tool
            tools['resource_search'] = search_tool
            tools['resource_write'] = writer_tool
            tools['aspect_create'] = aspect_create_tool
            tools['resource_relation_write'] = relation_tool
        planner = PursuitPlanner(template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
        summarizer = PursuitSummarizer(template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
        agent = Agent(
            tools=tools,
            channel=self.channel,
            planner=planner,
            summarizer=summarizer,
            template_env=self.template_env,
            chat_llm=self.chat_llm(context.chat_llm_factory)
        )

        await agent.pursue_goal(message)
        output = self.channel.get_output()
        return Action(
            message=output,
        )
