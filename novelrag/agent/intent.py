from novelrag.resource.repository import ResourceRepository
from .agent import create_agent
from .channel import SessionChannel
from .resource_tools import ResourceFetchTool, ResourceSearchTool, ResourceWriteTool, AspectCreateTool, ResourceRelationWriteTool
from novelrag.intent import LLMIntent, IntentContext, Action
from novelrag.template import TemplateEnvironment


# TODO: Tools should be defined in config
class AgentIntent(LLMIntent):
    def __init__(self, *, resource_repo: ResourceRepository, name: str | None, chat_llm: dict | None = None, lang: str | None = None, channel: SessionChannel, **kwargs):
        super().__init__(name=name, chat_llm=chat_llm, lang=lang, **kwargs)
        self.resource_repo = resource_repo
        self.template_env = TemplateEnvironment("novelrag.agent", "en")
        self.channel = channel

    @property
    def default_name(self) -> str | None:
        return 'agent'
    
    async def handle(self, message: str | None = None, *, context: IntentContext) -> Action:
        if message is None:
            raise ValueError("Message cannot be None for AgentIntent")

        # Set up tools for the agent (using SchematicTool interface)
        tools = {}
        if context.resource_repository:
            fetch_tool = ResourceFetchTool(context.resource_repository)
            search_tool = ResourceSearchTool(context.resource_repository)
            writer_tool = ResourceWriteTool(context.resource_repository, template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
            aspect_create_tool = AspectCreateTool(context.resource_repository, template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
            relation_tool = ResourceRelationWriteTool(context.resource_repository, template_env=self.template_env, chat_llm=self.chat_llm(context.chat_llm_factory))
            
            # Only include SchematicTool instances
            tools['resource_write'] = writer_tool
            tools['aspect_create'] = aspect_create_tool
            tools['resource_relation_write'] = relation_tool

        # Create agent using the new OrchestrationLoop approach
        agent = create_agent(
            tools=tools,
            resource_repo=self.resource_repo,
            template_env=self.template_env,
            chat_llm=self.chat_llm(context.chat_llm_factory),
            channel=self.channel
        )

        # Execute the goal pursuit
        result = await agent.handle_request(message)
        
        # Get any accumulated output from the channel
        channel_output = self.channel.get_output()
        
        # Combine agent result with channel output, ensuring list format
        messages = []
        if channel_output:
            if isinstance(channel_output, list):
                messages.extend(channel_output)
            else:
                messages.append(channel_output)
        
        if result:
            messages.append(result)
        
        return Action(
            message=messages if messages else None,
        )
