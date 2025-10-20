"""Agent implementation using OrchestrationLoop for strategic execution."""

import logging
from datetime import datetime
from typing import Any

from novelrag.agent.channel import AgentChannel
from novelrag.agent.orchestrate import OrchestrationLoop, OrchestrationExecutionPlan, OrchestrationFinalization
from novelrag.agent.steps import StepDefinition, StepOutcome, StepStatus
from novelrag.agent.tool import SchematicTool, ToolRuntime
from novelrag.agent.workspace import ResourceContext
from novelrag.agent.llm_logger import initialize_logger, get_logger
from novelrag.llm.types import ChatLLM
from novelrag.resource.repository import ResourceRepository
from novelrag.template import TemplateEnvironment

logger = logging.getLogger(__name__)


class AgentToolRuntime(ToolRuntime):
    """Agent's implementation of ToolRuntime that routes calls to AgentChannel."""
    
    def __init__(self, channel: AgentChannel):
        self.channel = channel
        self._backlog: list[str] = []
        self._progress: dict[str, list[str]] = {}
        self._triggered_actions: list[dict[str, str]] = []

    async def debug(self, content: str):
        await self.channel.debug(content)

    async def message(self, content: str):
        await self.channel.info(content)

    async def warning(self, content: str):
        await self.channel.warning(content)

    async def error(self, content: str):
        await self.channel.error(content)

    async def confirmation(self, prompt: str) -> bool:
        return await self.channel.confirm(prompt)

    async def user_input(self, prompt: str) -> str:
        return await self.channel.request(prompt)

    async def progress(self, key: str, value: str, description: str | None = None):
        if key not in self._progress:
            self._progress[key] = []
        self._progress[key].append(value)

    async def trigger_action(self, action: dict[str, str]):
        self._triggered_actions.append(action)

    async def backlog(self, content: dict, priority: str | None = None):
        self._backlog.append(str(content))


class Agent:
    """Agent using OrchestrationLoop for strategic goal pursuit."""

    def __init__(
        self,
        tools: dict[str, SchematicTool],
        orchestrator: OrchestrationLoop,
        resource_repo: ResourceRepository,
        template_env: TemplateEnvironment,
        chat_llm: ChatLLM,
        channel: AgentChannel
    ):
        self.tools = tools
        self.orchestrator = orchestrator
        self.resource_repo = resource_repo
        self.template_env = template_env
        self.chat_llm = chat_llm
        self.channel = channel

    async def pursue_goal(self, goal: str) -> str:
        """Pursue a goal using the OrchestrationLoop approach.
        
        Returns:
            Final response message for the user
        """
        await self.channel.info(f"Starting pursuit of goal: {goal}")
        
        # Start LLM logging for this pursuit
        llm_logger = get_logger()
        if llm_logger:
            llm_logger.start_pursuit(goal)
        
        # Create a fresh ResourceContext for this goal
        context = ResourceContext(self.resource_repo, self.template_env, self.chat_llm)
        
        # Track execution state
        completed_steps: list[StepOutcome] = []
        pending_steps: list[str] = []
        
        try:
            while True:
                # Let orchestrator decide next action (execution or finalization)
                decision = await self.orchestrator.execution_advance(
                    goal=goal,
                    completed_steps=completed_steps,
                    pending_steps=pending_steps,
                    available_tools=self.tools,
                    context=context
                )
                
                if isinstance(decision, OrchestrationFinalization):
                    # Goal pursuit is complete
                    await self.channel.info(f"Goal pursuit finalized: {decision.reason}")
                    return decision.response
                
                elif isinstance(decision, OrchestrationExecutionPlan):
                    # Execute the recommended tool
                    await self.channel.info(f"Executing: {decision.reason}")
                    
                outcome = await self._execute_tool(
                    tool_name=decision.tool,
                    params=decision.params,
                    reason=decision.reason,
                    context=context
                )
                completed_steps.append(outcome)
                pending_steps = decision.future_steps
                
                # Log execution result
                if outcome.status == StepStatus.SUCCESS:
                    await self.channel.info(f"✓ Completed: {outcome.action.reason}")
                    for result in outcome.results:
                        await self.channel.output(result)
                else:
                    await self.channel.error(f"✗ Failed: {outcome.action.reason} - {outcome.error_message}")
            
            else:
                raise ValueError(f"Unknown orchestration decision type: {type(decision)}")
                    
        except Exception as e:
            error_msg = f"Goal pursuit failed with error: {str(e)}"
            await self.channel.error(error_msg)
            logger.exception("Error during goal pursuit")
            return error_msg
        finally:
            # Complete the pursuit logging and dump to file
            if llm_logger:
                llm_logger.complete_pursuit()
                try:
                    log_file = llm_logger.dump_to_file()
                    await self.channel.debug(f"LLM logs saved to: {log_file}")
                except Exception as log_error:
                    logger.warning(f"Failed to save LLM logs: {log_error}")

    async def _execute_tool(self, tool_name: str, params: dict[str, Any], reason: str, context: ResourceContext) -> StepOutcome:
        """Execute a single tool and return the outcome."""
        start_time = datetime.now()
        
        # Create step definition for tracking
        step = StepDefinition(reason=reason, tool=tool_name, parameters=params)
        
        # Check if tool exists
        if tool_name not in self.tools:
            return StepOutcome(
                action=step,
                status=StepStatus.FAILED,
                error_message=f"Tool {tool_name} not found",
                started_at=start_time,
                completed_at=datetime.now()
            )
        
        tool = self.tools[tool_name]
        runtime = AgentToolRuntime(self.channel)
        
        try:
            # Add context to params if the tool requires it
            if tool.require_context:
                # Build context from the current workspace state
                tool_context = await context._generate_final_context()
                params = {**params, 'context': tool_context}
            
            await self.channel.debug(f"Calling tool {tool_name} with params: {params}")
            result = await tool.call(runtime, **params)
            
            from novelrag.agent.types import ToolResult, ToolError
            
            if isinstance(result, ToolResult):
                return StepOutcome(
                    action=step,
                    status=StepStatus.SUCCESS,
                    results=[result.result],
                    started_at=start_time,
                    completed_at=datetime.now(),
                    triggered_actions=runtime._triggered_actions,
                    backlog_items=runtime._backlog,
                    progress=runtime._progress,
                )
            elif isinstance(result, ToolError):
                return StepOutcome(
                    action=step,
                    status=StepStatus.FAILED,
                    error_message=result.error_message,
                    started_at=start_time,
                    completed_at=datetime.now(),
                    triggered_actions=runtime._triggered_actions,
                    backlog_items=runtime._backlog,
                    progress=runtime._progress,
                )
            else:
                raise ValueError(f"Unexpected tool result type: {type(result)}")
                
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}")
            return StepOutcome(
                action=step,
                status=StepStatus.FAILED,
                error_message=str(e),
                started_at=start_time,
                completed_at=datetime.now(),
            )


def create_agent(
    tools: dict[str, SchematicTool],
    resource_repo: ResourceRepository,
    template_env: TemplateEnvironment,
    chat_llm: ChatLLM,
    channel: AgentChannel,
    max_iterations: int = 5,
    min_iterations: int | None = None,
    log_directory: str = "logs"
) -> Agent:
    """Create a new Agent using the OrchestrationLoop approach.
    
    Args:
        tools: Dictionary of SchematicTool instances
        resource_repo: Resource repository for workspace management
        template_env: Template environment for LLM calls
        chat_llm: Chat LLM interface
        channel: Communication channel for user interaction
        max_iterations: Maximum orchestration iterations
        min_iterations: Minimum orchestration iterations
        log_directory: Directory to store LLM logs
        
    Returns:
        Configured Agent instance ready for goal pursuit
    """
    # Initialize the LLM logger
    initialize_logger(log_directory)
    
    # Create the orchestration loop
    orchestrator = OrchestrationLoop(
        template_env=template_env,
        chat_llm=chat_llm,
        max_iter=max_iterations,
        min_iter=min_iterations
    )
    
    # Create and return the agent
    return Agent(
        tools=tools,
        orchestrator=orchestrator,
        resource_repo=resource_repo,
        template_env=template_env,
        chat_llm=chat_llm,
        channel=channel
    )
