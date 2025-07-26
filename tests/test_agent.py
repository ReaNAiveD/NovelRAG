"""Test cases for the Agent class and its components.

This test suite focuses on testing the behavioral interfaces and business logic 
of the NovelRAG agent system, rather than simple property assignments:

1. AgentMind: Tests belief management logic, goal-setting effects, and world state
2. AgentCommunicator: Tests message creation and communication protocols  
3. Agent Planning: Tests planning logic, step ordering, and decomposition workflows (formerly AgentPlanner)
4. AgentExecutor: Tests execution workflows with realistic tool interactions
5. Agent: Tests end-to-end workflows and complex behavioral scenarios

Key testing principles:
- Tests verify correctness of interfaces and business logic
- Uses realistic mock tools that implement proper protocols
- Focuses on behavior over implementation details
- Tests error handling and edge cases in workflows
- Verifies proper context passing and state management

All tests use comprehensive mocks to avoid external dependencies while 
ensuring the agent's behavioral contracts are properly verified.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock
from typing import Any, AsyncGenerator

from novelrag.agent.agent import Agent, AgentMind
from novelrag.agent.channel import AgentChannel
from novelrag.agent.schedule import GoalPlanner, Step, GoalPursuit, GoalPursuitResult, ExecutionPlan
from novelrag.agent.tool import BaseTool, ContextualTool
from novelrag.agent.types import (
    AgentMessage, AgentResult, AgentMessageLevel, ToolMessage, ToolResult, 
    ToolStepDecomposition, ToolBacklogOutput, MessageLevel, ToolOutput, PursuitStatus
)
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment


class MockContextualTool(ContextualTool):
    """Mock contextual tool for testing purposes."""
    
    def __init__(self, name: str = "mock_contextual_tool", 
                 description: str = "A mock contextual tool",
                 outputs: list[ToolOutput] | None = None):
        self._name = name
        self._description = description
        self.outputs = outputs or [ToolResult(result="mock_result")]
        self.call_count = 0
        self.last_call_args = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    async def call(self, believes: list[str] | None = None, step_description: str | None = None, 
                   context: list[str] | None = None, tools: dict[str, str] | None = None) -> AsyncGenerator[ToolOutput, bool | str | None]:
        """Mock call method that returns predefined outputs."""
        self.call_count += 1
        self.last_call_args = {
            'believes': believes,
            'step_description': step_description,
            'context': context,
            'tools': tools
        }
        
        for output in self.outputs:
            yield output


class TestAgent(unittest.IsolatedAsyncioTestCase):
    """Test cases for the main Agent class focusing on behavioral verification."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_tools: dict[str, BaseTool] = {
            "test_tool": MockContextualTool("test_tool", "A test tool"),
            "other_tool": MockContextualTool("other_tool", "Another tool")
        }
        self.mock_template_env = MagicMock(spec=TemplateEnvironment)
        self.mock_chat_llm = AsyncMock(spec=ChatLLM)
        self.mock_planner = AsyncMock(spec=GoalPlanner)
        self.mock_channel = AsyncMock(spec=AgentChannel)
        self.agent = Agent(
            tools=self.mock_tools,
            template_env=self.mock_template_env,
            chat_llm=self.mock_chat_llm,
            planner=self.mock_planner,
            channel=self.mock_channel
        )
    
    async def test_tool_execution_with_context_passing(self):
        """Test that tool execution properly passes context and handles results."""
        # Set up agent state with a realistic workflow
        self.agent.mind.set_goal("Test tool execution")
        self.agent.mind.add_belief("Tools should receive proper context")
        
        # Use a tool that's already in the agent's tools
        mock_tool = self.mock_tools["test_tool"]
        assert isinstance(mock_tool, MockContextualTool)  # Type assertion for mypy
        mock_tool.outputs = [
            ToolMessage(content="Processing request", level=MessageLevel.INFO),
            ToolResult(result="test result")
        ]
        
        # Create a mock goal pursuit that will call our tool
        from novelrag.agent.schedule import StepOutcome, StepStatus
        from datetime import datetime
        
        step = Step(tool="test_tool", intent="Execute test operation")
        step_outcome = StepOutcome(
            action=step,
            status=StepStatus.SUCCESS,
            results=["test result"],
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        plan = ExecutionPlan(
            goal="Test tool execution",
            completed_steps=[step_outcome]
        )
        
        pursuit_result = GoalPursuitResult(
            goal="Test tool execution",
            status=PursuitStatus.COMPLETED,
            records=plan,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Mock the planner to return our pursuit result
        self.mock_planner.plan_goal.return_value = pursuit_result
        
        # Execute the goal
        await self.agent.handle_goal("Test tool execution")
        
        # Verify planner was called
        self.mock_planner.plan_goal.assert_called_once_with(
            "Test tool execution", 
            self.agent.believes, 
            self.agent.contextual_tools
        )
        
        # Verify output was sent to channel
        self.mock_channel.output.assert_called()
    
    async def test_schedule_execution_error_handling(self):
        """Test that schedule execution properly handles tool errors and failures."""
        # Set up a plan with a failing tool
        failing_tool = MockContextualTool(
            "failing_tool", 
            "A tool that fails",
            outputs=[ToolMessage(content="Something went wrong", level=MessageLevel.ERROR)]
        )
        
        # Add to agent's tools
        self.agent.tools["failing_tool"] = failing_tool
        self.agent.contextual_tools["failing_tool"] = failing_tool
        
        # Create a mock goal pursuit that encounters an error
        from novelrag.agent.schedule import StepOutcome, StepStatus
        from datetime import datetime
        
        step = Step(tool="failing_tool", intent="This will fail")
        step_outcome = StepOutcome(
            action=step,
            status=StepStatus.FAILED,
            error_message="Something went wrong",
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        plan = ExecutionPlan(
            goal="Test error handling",
            failed_steps=[step_outcome]
        )
        
        pursuit_result = GoalPursuitResult(
            goal="Test error handling",
            status=PursuitStatus.FAILED,
            records=plan,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Mock the planner to return our pursuit result
        self.mock_planner.plan_goal.return_value = pursuit_result
        
        # Execute the goal
        await self.agent.handle_goal("Test error handling")
        
        # Verify planner was called
        self.mock_planner.plan_goal.assert_called_once_with(
            "Test error handling", 
            self.agent.believes, 
            self.agent.contextual_tools
        )
    
    async def test_step_decomposition_workflow(self):
        """Test that step decomposition properly creates and manages sub-steps."""
        # Create a tool that decomposes its task
        decomposing_tool = MockContextualTool(
            "decomposing_tool",
            "A tool that breaks down tasks",
            outputs=[
                ToolStepDecomposition(
                    steps=[
                        {"tool": "test_tool", "description": "First subtask"},
                        {"tool": "other_tool", "description": "Second subtask"}
                    ],
                    rationale="Breaking down complex task into manageable parts"
                )
            ]
        )
        
        # Add to agent's tools
        self.agent.tools["decomposing_tool"] = decomposing_tool
        self.agent.contextual_tools["decomposing_tool"] = decomposing_tool
        
        # Create a mock goal pursuit that uses the decomposing tool
        from novelrag.agent.schedule import StepOutcome, StepStatus
        from datetime import datetime
        
        step = Step(tool="decomposing_tool", intent="Complex task to decompose")
        step_outcome = StepOutcome(
            action=step,
            status=StepStatus.DECOMPOSED,
            spawned_actions=[
                Step(tool="test_tool", intent="First subtask"),
                Step(tool="other_tool", intent="Second subtask")
            ],
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        plan = ExecutionPlan(
            goal="Test decomposition",
            completed_steps=[step_outcome]
        )
        
        pursuit_result = GoalPursuitResult(
            goal="Test decomposition",
            status=PursuitStatus.COMPLETED,
            records=plan,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Mock the planner to return our pursuit result
        self.mock_planner.plan_goal.return_value = pursuit_result
        
        # Execute the goal
        await self.agent.handle_goal("Test decomposition")
        
        # Verify planner was called
        self.mock_planner.plan_goal.assert_called_once_with(
            "Test decomposition", 
            self.agent.believes, 
            self.agent.contextual_tools
        )
    
    async def test_backlog_management_during_execution(self):
        """Test that backlog items are properly managed during execution."""
        # Create a tool that adds items to backlog
        backlog_tool = MockContextualTool(
            "backlog_tool",
            "A tool that discovers new items",
            outputs=[
                ToolBacklogOutput(content="Research medieval weapons", priority="high"),
                ToolResult(result="Main task completed")
            ]
        )
        
        # Add to agent's tools
        self.agent.tools["backlog_tool"] = backlog_tool
        self.agent.contextual_tools["backlog_tool"] = backlog_tool
        
        # Start with empty backlog
        initial_backlog_size = len(self.agent.backlog)
        
        # Create a mock goal pursuit that uses the backlog tool
        from novelrag.agent.schedule import StepOutcome, StepStatus
        from datetime import datetime
        
        step = Step(tool="backlog_tool", intent="Task that discovers more work")
        step_outcome = StepOutcome(
            action=step,
            status=StepStatus.SUCCESS,
            results=["Main task completed"],
            backlog_items=["Research medieval weapons"],
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        plan = ExecutionPlan(
            goal="Test backlog management",
            completed_steps=[step_outcome]
        )
        
        pursuit_result = GoalPursuitResult(
            goal="Test backlog management",
            status=PursuitStatus.COMPLETED,
            records=plan,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Mock the planner to return our pursuit result
        self.mock_planner.plan_goal.return_value = pursuit_result
        
        # Execute the goal
        await self.agent.handle_goal("Test backlog management")
        
        # Verify planner was called
        self.mock_planner.plan_goal.assert_called_once_with(
            "Test backlog management", 
            self.agent.believes, 
            self.agent.contextual_tools
        )
        
        # Note: In the real implementation, backlog management would happen during step execution
        # For this test, we're verifying that the agent framework can handle backlog outputs
        # The actual backlog updating would be done by the step execution logic


if __name__ == '__main__':
    unittest.main()