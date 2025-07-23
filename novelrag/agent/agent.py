"""Main Agent class for orchestrating execution."""

from typing import Any, AsyncGenerator

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .tool import BaseTool, LLMToolMixin
from .schedule import Scheduler, Step
from .proposals import TargetProposer
from .types import (
    AgentOutput, AgentMessage, AgentConfirmation, AgentUserInput, AgentResult,
    AgentMessageLevel, ToolMessage, ToolConfirmation, ToolResult, ToolUserInput,
    ToolStepProgress, ToolStepDecomposition, ToolBacklogOutput
)


class AgentCommunicator:
    """Handles all agent communication with users and external systems.
    
    This represents the agent's ability to express thoughts, ask questions,
    and interact with the user in natural language.
    """
    
    @staticmethod
    def message(content: str, level: AgentMessageLevel = AgentMessageLevel.INFO) -> AgentMessage:
        """Express a thought or information to the user."""
        return AgentMessage(content=content, level=level)
    
    @staticmethod
    def debug(content: str) -> AgentMessage:
        """Share internal reasoning/debugging information."""
        return AgentMessage(content=content, level=AgentMessageLevel.DEBUG)
    
    @staticmethod
    def warning(content: str) -> AgentMessage:
        """Express concern or caution about something."""
        return AgentMessage(content=content, level=AgentMessageLevel.WARNING)
    
    @staticmethod
    def error(content: str) -> AgentMessage:
        """Report an error or problem encountered."""
        return AgentMessage(content=content, level=AgentMessageLevel.ERROR)
    
    @staticmethod
    def ask_confirmation(prompt: str) -> AgentConfirmation:
        """Ask the user to confirm a potentially risky action."""
        return AgentConfirmation(prompt=prompt)
    
    @staticmethod
    def request_input(prompt: str) -> AgentUserInput:
        """Request specific information from the user."""
        return AgentUserInput(prompt=prompt)
    
    @staticmethod
    def report_result(output: Any) -> AgentResult:
        """Report the completion and result of a task."""
        return AgentResult(result=output)


class AgentMind:
    """The cognitive core of the agent - beliefs, goals, and decision-making.
    
    This represents the agent's mental model of the world, its objectives,
    and its ability to reason and make decisions.
    """
    
    def __init__(self):
        self.beliefs: list[str] = []  # What the agent believes to be true
        self.current_goal: str | None = None  # What the agent is trying to achieve
        self.backlog: list[str] = []  # Things the agent wants to remember for later
        
    def set_goal(self, goal: str):
        """Set a new primary goal for the agent to pursue."""
        self.current_goal = goal
    
    def add_belief(self, belief: str):
        """Add a new belief to the agent's world model."""
        if belief not in self.beliefs:
            self.beliefs.append(belief)
    
    def remove_belief(self, belief: str):
        """Remove a belief that is no longer valid."""
        if belief in self.beliefs:
            self.beliefs.remove(belief)
    
    def add_to_backlog(self, item: str):
        """Remember something for later consideration."""
        if item not in self.backlog:
            self.backlog.append(item)
    
    def get_world_view(self) -> dict[str, Any]:
        """Get the agent's current understanding of the world."""
        return {
            "beliefs": self.beliefs.copy(),
            "current_goal": self.current_goal,
            "backlog_items": len(self.backlog)
        }


class AgentPlanner:
    """The agent's planning and scheduling capabilities.
    
    This represents the agent's ability to break down goals into actionable steps,
    sequence activities, and adapt plans based on results.
    """
    
    def __init__(self, mind: AgentMind, communicator: AgentCommunicator):
        self.mind = mind
        self.communicator = communicator
        self.scheduler: Scheduler | None = None
    
    def create_plan(self, initial_steps: list[Step]):
        """Create a new execution plan for achieving the current goal."""
        self.scheduler = Scheduler(initial_steps)
    
    def adapt_plan(self, new_steps: list[Step], insert_at_front: bool = False):
        """Adapt the current plan by adding new steps."""
        if not self.scheduler:
            raise ValueError("No plan exists. Create a plan first.")
        
        if insert_at_front:
            self.scheduler.pending_steps = new_steps + self.scheduler.pending_steps
        else:
            self.scheduler.pending_steps.extend(new_steps)
    
    def get_plan_status(self) -> dict:
        """Get the current status of the execution plan."""
        if not self.scheduler:
            return {"status": "no_plan", "goal": self.mind.current_goal}
        
        return {
            "status": "active",
            "goal": self.mind.current_goal,
            "pending_steps": len(self.scheduler.pending_steps),
            "completed_steps": len(self.scheduler.completed_steps),
            "failed_steps": len(self.scheduler.failed_steps),
            "current_step": self.scheduler.executing_step.description if self.scheduler.executing_step else None
        }
    
    def get_context_for_current_step(self) -> list[str]:
        """Get relevant context for the step being executed."""
        if not self.scheduler or not self.scheduler.executing_step:
            return []
        return self.scheduler.build_context_for_step(self.scheduler.executing_step)
    
    async def process_step_decomposition(self, decomposition: ToolStepDecomposition, 
                                       current_step: Step) -> list[AgentOutput]:
        """React to a step being broken down into smaller steps."""
        if not self.scheduler:
            return []
            
        self.scheduler.add_decomposed_steps(current_step, decomposition.steps)
        
        return [
            self.communicator.message(f"I've broken this step into {len(decomposition.steps)} smaller tasks"),
            self.communicator.debug(f"Reasoning: {decomposition.rationale}")
        ]
    
    async def process_plan_updates(self, updates: list[str], triggering_step: Step) -> list[AgentOutput]:
        """React to discovering new things that need to be done."""
        if not updates or not self.scheduler:
            return []
            
        self.scheduler.add_chain_update_steps(triggering_step, updates)
        return [self.communicator.message(f"I realized I need to do {len(updates)} additional things")]
    
    async def process_backlog_additions(self, items: list[str]) -> list[AgentOutput]:
        """React to discovering things to remember for later."""
        if not items:
            return []
            
        for item in items:
            self.mind.add_to_backlog(item)
            
        if self.scheduler:
            self.scheduler.add_backlog_items(items)
            
        return [self.communicator.message(f"I've noted {len(items)} things to remember for later")]


class AgentExecutor:
    """The agent's action execution capabilities.
    
    This represents the agent's ability to actually perform actions in the world
    through tools, handle results, and adapt to unexpected situations.
    """
    
    def __init__(self, mind: AgentMind, planner: AgentPlanner, communicator: AgentCommunicator):
        self.mind = mind
        self.planner = planner  
        self.communicator = communicator
    
    async def execute_plan(self, tools: dict[str, Any], template_env: Any, 
                          chat_llm: Any) -> AsyncGenerator[AgentOutput, None]:
        """Execute the current plan step by step."""
        if not self.planner.scheduler:
            yield self.communicator.error("I don't have a plan to execute. Please give me a goal first.")
            return
            
        yield self.communicator.message(f"Starting to work on: {self.mind.current_goal}")
        
        while True:
            step = self.planner.scheduler.start_next_step()
            if not step:
                yield self.communicator.message("I've completed all planned steps!")
                break
                
            yield self.communicator.debug(f"Working on: {step.description}")
            
            try:
                async for output in self.execute_action(step, tools, template_env, chat_llm):
                    yield output
                    if isinstance(output, AgentResult):
                        self.planner.scheduler.complete_current_step()
                        break
            except Exception as e:
                self.planner.scheduler.fail_current_step(str(e))
                yield self.communicator.error(f"Something went wrong: {str(e)}")

    async def execute_action(self, step: Step, tools: dict[str, Any], 
                           template_env: Any, chat_llm: Any) -> AsyncGenerator[AgentOutput, None]:
        """Execute a single action using the appropriate tool."""
        from .tool import BaseTool, SchematicTool, ContextualTool
        
        if not self.planner.scheduler:
            yield self.communicator.error("I've lost track of my plan!")
            return
            
        tool_name = step.tool
        if tool_name not in tools:
            yield self.communicator.error(f"I don't know how to use the '{tool_name}' tool.")
            return
            
        tool = tools[tool_name]
        if not isinstance(tool, BaseTool):
            yield self.communicator.error(f"The '{tool_name}' tool doesn't seem to be working properly.")
            return
            
        if isinstance(tool, SchematicTool):
            tool = tool.wrapped(template_env, chat_llm)
            
        if not isinstance(tool, ContextualTool):
            yield self.communicator.error(f"The '{tool_name}' tool can't understand context.")
            return

        # Execute the action
        context = self.planner.scheduler.build_context_for_step(step)
        tool_call = tool.call(
            believes=self.mind.beliefs,
            step_description=step.description,
            context=context,
            tools={name: t.description or '' for name, t in tools.items() if isinstance(t, BaseTool)}
        )
        
        user_input = None
        while True:
            try:
                output = await tool_call.asend(user_input)
                user_input = None
                
                if isinstance(output, ToolMessage):
                    agent_level = AgentMessageLevel(output.level.value)
                    yield self.communicator.message(output.content, level=agent_level)
                    
                elif isinstance(output, ToolConfirmation):
                    user_input = yield self.communicator.ask_confirmation(output.prompt)
                    
                elif isinstance(output, ToolResult):
                    step.result.append(output.result)
                    yield self.communicator.report_result(output.result)
                    return  # Action completed
                    
                elif isinstance(output, ToolUserInput):
                    user_input = yield self.communicator.request_input(output.prompt)
                    
                elif isinstance(output, ToolStepProgress):
                    if output.field == 'proposal':
                        step.proposal = output.value
                    else:
                        step.additional_context[output.field] = output.value
                    yield self.communicator.message(f"Progress: {output.description or output.field}")
                    
                elif isinstance(output, ToolStepDecomposition):
                    # The agent realizes this step needs to be broken down
                    decomp_outputs = await self.planner.process_step_decomposition(output, step)
                    for decomp_output in decomp_outputs:
                        yield decomp_output
                    yield self.communicator.report_result("I've broken this down into smaller steps")
                    return
                    
                elif isinstance(output, ToolBacklogOutput):
                    # The agent discovers something to remember for later
                    backlog_outputs = await self.planner.process_backlog_additions([output.content])
                    for backlog_output in backlog_outputs:
                        yield backlog_output
                    
                else:
                    yield self.communicator.error(f"The tool gave me something I don't understand: {type(output)}")
                    return
                    
            except StopAsyncIteration:
                yield self.communicator.warning(f"The '{tool_name}' tool finished without giving me a result.")
                return


class Agent(LLMToolMixin):
    """Main agent class for orchestrating tool execution and managing state.
    
    This is a composition of the core agent components:
    - AgentMind: Handles beliefs, goals, and mental state
    - AgentCommunicator: Manages all communication and output
    - AgentPlanner: Handles planning and scheduling
    - AgentExecutor: Executes actions and tools
    """
    
    def __init__(self, tools: dict[str, BaseTool], template_env: TemplateEnvironment, chat_llm: ChatLLM):
        # Initialize core components
        self.mind = AgentMind()
        self.communicator = AgentCommunicator()
        self.planner = AgentPlanner(self.mind, self.communicator)
        self.executor = AgentExecutor(self.mind, self.planner, self.communicator)
        
        # Legacy compatibility - delegate to mind
        self.tools: dict[str, BaseTool] = tools
        self.target_proposers: list[TargetProposer] = []
        super().__init__(template_env=template_env, chat_llm=chat_llm)

    # Properties for backward compatibility
    @property
    def believes(self) -> list[str]:
        """Access beliefs through the mind component."""
        return self.mind.beliefs
    
    @believes.setter
    def believes(self, value: list[str]):
        """Set beliefs through the mind component."""
        self.mind.beliefs = value
    
    @property
    def target(self) -> str | None:
        """Access current goal through the mind component."""
        return self.mind.current_goal
    
    @target.setter
    def target(self, value: str | None):
        """Set current goal through the mind component."""
        if value:
            self.mind.set_goal(value)
        else:
            self.mind.current_goal = None
    
    @property
    def backlog(self) -> list[str]:
        """Access backlog through the mind component."""
        return self.mind.backlog
    
    @backlog.setter
    def backlog(self, value: list[str]):
        """Set backlog through the mind component."""
        self.mind.backlog = value
    
    @property
    def scheduler(self) -> Scheduler | None:
        """Access scheduler through the planner component."""
        return self.planner.scheduler
    
    @scheduler.setter
    def scheduler(self, value: Scheduler | None):
        """Set scheduler through the planner component."""
        self.planner.scheduler = value

    def decide(self):
        """Auto Decision - delegates to mind component."""
        # TODO: Implement decision logic in mind component
        pass

    def set_target(self, target: str):
        """Set target goal - delegates to mind component."""
        self.mind.set_goal(target)

    def set_schedule(self, initial_steps: list[Step]):
        """Initialize the scheduler with initial steps - delegates to planner."""
        self.planner.create_plan(initial_steps)
    
    def add_steps(self, steps: list[Step], insert_at_front: bool = False):
        """Add steps to the current schedule - delegates to planner."""
        self.planner.adapt_plan(steps, insert_at_front)
    
    def get_schedule_status(self) -> dict:
        """Get current status of the schedule - delegates to planner."""
        return self.planner.get_plan_status()
    
    def get_context_for_current_step(self) -> list[str]:
        """Get context for the currently executing step - delegates to planner."""
        return self.planner.get_context_for_current_step()

    async def execute_schedule(self) -> AsyncGenerator[AgentOutput, None]:
        """Execute the complete schedule - delegates to executor."""
        async for output in self.executor.execute_plan(self.tools, self.template_env, self.chat_llm):
            yield output

    async def execute_step(self, step: Step) -> AsyncGenerator[AgentOutput, None]:
        """Execute a single step - delegates to executor component."""
        async for output in self.executor.execute_action(step, self.tools, self.template_env, self.chat_llm):
            yield output

    async def _process_step_decomposition(self, output: ToolStepDecomposition, current_step: Step) -> list[AgentOutput]:
        """Process step decomposition - delegates to planner component."""
        return await self.planner.process_step_decomposition(output, current_step)
    
    async def _process_chain_updates(self, updates: list[str], triggering_step: Step) -> list[AgentOutput]:
        """Process chain updates - delegates to planner component."""
        return await self.planner.process_plan_updates(updates, triggering_step)
    
    async def _process_backlog_items(self, items: list[str]) -> list[AgentOutput]:
        """Process backlog items - delegates to planner component."""
        return await self.planner.process_backlog_additions(items)
