"""Main Agent class for orchestrating execution."""

from typing import Any, AsyncGenerator

from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment

from .tool import BaseTool, LLMToolMixin, SchematicTool, ContextualTool
from .schedule import Scheduler, Step
from .proposals import TargetProposer
from .types import (
    AgentOutput, AgentMessage, AgentConfirmation, AgentUserInput, AgentResult,
    AgentMessageLevel, ToolMessage, ToolConfirmation, ToolResult, ToolUserInput,
    ToolStepProgress, ToolStepDecomposition, ToolBacklogOutput
)


class ToolOutputProcessor:
    """Processes tool outputs and updates schedule accordingly"""
    
    def __init__(self, scheduler: Scheduler, agent: 'Agent'):
        self.scheduler = scheduler
        self.agent = agent
    
    async def process_step_decomposition(self, output: ToolStepDecomposition, current_step: Step) -> list[AgentOutput]:
        """Process step decomposition and update schedule"""
        self.scheduler.add_decomposed_steps(current_step, output.steps)
        
        return [
            self.agent.message(f"Step decomposed into {len(output.steps)} sub-steps"),
            self.agent.debug(f"Decomposition rationale: {output.rationale}")
        ]
    
    async def process_chain_updates(self, updates: list[str], triggering_step: Step) -> list[AgentOutput]:
        """Process chain updates and create follow-up steps"""
        if not updates:
            return []
            
        self.scheduler.add_chain_update_steps(triggering_step, updates)
        return [self.agent.message(f"Created {len(updates)} chain update steps")]
    
    async def process_backlog_items(self, items: list[str]) -> list[AgentOutput]:
        """Process backlog items"""
        if not items:
            return []
            
        self.scheduler.add_backlog_items(items)
        return [self.agent.message(f"Added {len(items)} items to backlog")]


class Agent(LLMToolMixin):
    """Main agent class for orchestrating tool execution and managing state."""
    
    def __init__(self, tools: dict[str, BaseTool], template_env: TemplateEnvironment, chat_llm: ChatLLM):
        self.believes: list[str] = []
        self.target: str | None = None
        self.backlog: list[str] = []
        self.tools: dict[str, BaseTool] = tools
        self.target_proposers: list[TargetProposer] = []
        self.scheduler: Scheduler | None = None
        self.output_processor: ToolOutputProcessor | None = None
        super().__init__(template_env=template_env, chat_llm=chat_llm)

    def decide(self):
        # Auto Decision
        pass

    def set_target(self, target: str):
        self.target = target

    def set_schedule(self, initial_steps: list[Step]):
        """Initialize the scheduler with initial steps"""
        self.scheduler = Scheduler(initial_steps)
        self.output_processor = ToolOutputProcessor(self.scheduler, self)
    
    def add_steps(self, steps: list[Step], insert_at_front: bool = False):
        """Add steps to the current schedule"""
        if not self.scheduler:
            raise ValueError("No scheduler configured. Use set_schedule() first.")
        
        if insert_at_front:
            self.scheduler.pending_steps = steps + self.scheduler.pending_steps
        else:
            self.scheduler.pending_steps.extend(steps)
    
    def get_schedule_status(self) -> dict:
        """Get current status of the schedule"""
        if not self.scheduler:
            return {"status": "no_scheduler"}
        
        return {
            "status": "active",
            "pending_steps": len(self.scheduler.pending_steps),
            "completed_steps": len(self.scheduler.completed_steps),
            "failed_steps": len(self.scheduler.failed_steps),
            "current_step": self.scheduler.executing_step.description if self.scheduler.executing_step else None
        }
    
    def get_context_for_current_step(self) -> list[str]:
        """Get context for the currently executing step"""
        if not self.scheduler or not self.scheduler.executing_step:
            return []
        return self.scheduler.build_context_for_step(self.scheduler.executing_step)

    async def execute_schedule(self) -> AsyncGenerator[AgentOutput, None]:
        """Execute the complete schedule"""
        if not self.scheduler:
            yield self.error("No schedule set. Use set_schedule() first.")
            return
            
        while True:
            step = self.scheduler.start_next_step()
            if not step:
                yield self.message("Schedule execution completed.")
                break
                
            yield self.debug(f"Starting step: {step.description}")
            
            try:
                async for output in self.execute_step(step):
                    yield output
                    if isinstance(output, AgentResult):
                        self.scheduler.complete_current_step()
                        break
            except Exception as e:
                self.scheduler.fail_current_step(str(e))
                yield self.error(f"Step failed: {str(e)}")

    async def execute_step(self, step: Step) -> AsyncGenerator[AgentOutput, None]:
        """Execute a single step and handle its outputs"""
        if not self.scheduler:
            yield self.error("No scheduler configured.")
            return
            
        tool_name = step.tool
        if tool_name not in self.tools:
            yield self.error(f"Tool '{tool_name}' not found in available tools.")
            return
            
        tool = self.tools[tool_name]
        if not isinstance(tool, BaseTool):
            yield self.error(f"Tool '{tool_name}' is not a valid BaseTool instance.")
            return
            
        if isinstance(tool, SchematicTool):
            # Use the wrapped version for schematic tools
            tool = tool.wrapped(self.template_env, self.chat_llm)
            
        if not isinstance(tool, ContextualTool):
            yield self.error(f"Tool '{tool_name}' does not support contextual execution.")
            return

        # Execute the tool with current context
        context = self.scheduler.build_context_for_step(step)
        tool_call = tool.call(
            believes=self.believes,
            step_description=step.description,
            context=context,
            tools={name: t.description or '' for name, t in self.tools.items() if isinstance(t, BaseTool)}
        )
        
        user_input = None
        while True:
            try:
                output = await tool_call.asend(user_input)
                user_input = None
                
                if isinstance(output, ToolMessage):
                    agent_level = AgentMessageLevel(output.level.value)
                    yield self.message(output.content, level=agent_level)
                    
                elif isinstance(output, ToolConfirmation):
                    user_input = yield self.confirmation(output.prompt)
                    
                elif isinstance(output, ToolResult):
                    step.result.append(output.result)
                    yield self.result(output.result)
                    return  # Step completed
                    
                elif isinstance(output, ToolUserInput):
                    user_input = yield self.user_input(output.prompt)
                    
                elif isinstance(output, ToolStepProgress):
                    if output.field == 'proposal':
                        step.proposal = output.value
                    else:
                        step.additional_context[output.field] = output.value
                    yield self.message(f"Step progress: {output.description or output.field}")
                    
                elif isinstance(output, ToolStepDecomposition):
                    # Handle step decomposition by modifying the schedule
                    if self.output_processor:
                        decomp_outputs = await self.output_processor.process_step_decomposition(output, step)
                        for decomp_output in decomp_outputs:
                            yield decomp_output
                    yield self.result("Step decomposed")  # Mark step as completed
                    return  # Current step is replaced by decomposed steps
                    
                elif isinstance(output, ToolBacklogOutput):
                    self.backlog.append(f"{output.content} (priority: {output.priority})")
                    if self.output_processor:
                        backlog_outputs = await self.output_processor.process_backlog_items([f"{output.content}"])
                        for backlog_output in backlog_outputs:
                            yield backlog_output
                    
                else:
                    yield self.error(f"Unknown output type from tool {tool_name}: {type(output)}")
                    return
                    
            except StopAsyncIteration:
                yield self.warning(f"Tool {tool_name} completed without producing a result.")
                return


    async def need_write(self, step: Step) -> bool:
        return (await self.call_template(
            'need_write.jinja2',
            step_description=step.description,
        )).lower() in ['yes', 'true', '1']

    async def build_query(self, step: Step) -> str:
        return await self.call_template(
            'build_query.jinja2',
            step_description=step.description,
            json_format=True
        )
    
    async def query_result_filter(self, qry_step: Step, results: list[str]) -> list[str]:
        return (await self.call_template(
            'query_result_filter.jinja2',
            step_description=qry_step.description,
            results=results,
        )).splitlines()
    
    def message(self, content: str, level: AgentMessageLevel = AgentMessageLevel.INFO) -> AgentMessage:
        """Create a message output."""
        return AgentMessage(content=content, level=level)
    
    def debug(self, content: str) -> AgentMessage:
        """Create a debug message output."""
        return AgentMessage(content=content, level=AgentMessageLevel.DEBUG)
    
    def warning(self, content: str) -> AgentMessage:
        """Create a warning message output."""
        return AgentMessage(content=content, level=AgentMessageLevel.WARNING)
    
    def error(self, content: str) -> AgentMessage:
        """Create an error message output."""
        return AgentMessage(content=content, level=AgentMessageLevel.ERROR)
    
    def confirmation(self, prompt: str) -> AgentConfirmation:
        """Create a confirmation output."""
        return AgentConfirmation(prompt=prompt)
    
    def user_input(self, prompt: str) -> AgentUserInput:
        """Create a user input request."""
        return AgentUserInput(prompt=prompt)
    
    def result(self, output: Any) -> AgentResult:
        """Create a result output."""
        return AgentResult(result=output)
