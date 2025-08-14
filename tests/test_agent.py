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
from datetime import datetime

from novelrag.agent.agent import Agent, AgentMind
from novelrag.agent.channel import AgentChannel
from novelrag.agent.execution import ExecutableStep, StepDefinition, ExecutionPlan, StepOutcome, StepStatus
from novelrag.agent.planning import PursuitPlanner
from novelrag.agent.pursuit import GoalPursuitResult, PursuitStatus, PursuitSummarizer
from novelrag.agent.tool import BaseTool, ContextualTool
from novelrag.agent.types import ToolResult
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment


class MockContextualTool(ContextualTool):
    """Mock contextual tool for testing purposes."""
    
    def __init__(self, name: str = "mock_contextual_tool", 
                 description: str = "A mock contextual tool",
                 output: ToolResult | None = None):
        self._name = name
        self._description = description
        self.output = output or ToolResult(result="mock_result")
        self.call_count = 0
        self.last_call_args = None
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def description(self) -> str:
        return self._description
    
    async def call(self, runtime, believes: list[str] | None = None, step_description: str | None = None,
                   context: list[str] | None = None, tools: dict[str, str] | None = None):
        """Mock call method that returns a predefined ToolResult."""
        self.call_count += 1
        self.last_call_args = {
            'believes': believes,
            'step_description': step_description,
            'context': context,
            'tools': tools
        }
        return self.output


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
        self.mock_planner = AsyncMock(spec=PursuitPlanner)
        self.mock_channel = AsyncMock(spec=AgentChannel)
        self.mock_summarizer = AsyncMock(spec=PursuitSummarizer)

        self.agent = Agent(
            tools=self.mock_tools,
            template_env=self.mock_template_env,
            chat_llm=self.mock_chat_llm,
            planner=self.mock_planner,
            channel=self.mock_channel,
            summarizer=self.mock_summarizer,
        )
    
    def test_agent_initialization(self):
        """Test that agent initializes correctly with all components."""
        self.assertIsInstance(self.agent.mind, AgentMind)
        self.assertEqual(len(self.agent.contextual_tools), 2)
        self.assertIn("test_tool", self.agent.contextual_tools)
        self.assertIn("other_tool", self.agent.contextual_tools)

    def test_agent_properties(self):
        """Test that agent properties delegate to mind component."""
        # Test belief property
        self.agent.mind.add_belief("Test belief")
        self.assertIn("Test belief", self.agent.believes)

        # Test target property
        self.agent.mind.set_goal("Test goal")
        self.assertEqual(self.agent.target, "Test goal")

        # Test backlog property
        self.agent.mind.add_to_backlog("Test item")
        self.assertIn("Test item", self.agent.backlog)

    async def test_pursue_goal_successful_completion(self):
        """Test successful goal pursuit with tool execution."""
        goal = "Test goal pursuit"

        # Set up mock planner to return initial steps
        mock_steps = [
            ExecutableStep(definition=StepDefinition(tool="test_tool", intent="Execute test operation"))
        ]
        self.mock_planner.create_initial_plan.return_value = mock_steps

        # Set up mock pursuit result
        execution_plan = ExecutionPlan(
            goal=goal,
            pending_steps=[],
            completed_steps=[
                StepOutcome(
                    action=mock_steps[0],
                    status=StepStatus.SUCCESS,
                    results=["test result"],
                    started_at=datetime.now(),
                    completed_at=datetime.now()
                )
            ]
        )
        
        pursuit_result = GoalPursuitResult(
            goal=goal,
            status=PursuitStatus.COMPLETED,
            records=execution_plan,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Mock the summarizer
        self.mock_summarizer.summarize_pursuit.return_value = "Goal completed successfully"

        # Mock GoalPursuit.initialize_pursuit and run_to_completion
        with unittest.mock.patch('novelrag.agent.agent.GoalPursuit') as mock_pursuit_class:
            mock_pursuit_instance = MagicMock()
            mock_pursuit_instance.plan.pending_steps = mock_steps
            mock_pursuit_instance.run_to_completion = AsyncMock(return_value=pursuit_result)
            mock_pursuit_class.initialize_pursuit = AsyncMock(return_value=mock_pursuit_instance)

            # Execute the goal
            await self.agent.pursue_goal(goal)

            # Verify initialization was called correctly
            mock_pursuit_class.initialize_pursuit.assert_called_once_with(
                goal=goal,
                believes=self.agent.believes,
                tools=self.agent.contextual_tools,
                planner=self.mock_planner
            )

            # Verify run_to_completion was called
            mock_pursuit_instance.run_to_completion.assert_called_once_with(
                self.agent.contextual_tools,
                self.mock_channel,
                self.mock_planner
            )

            # Verify summarizer was called
            self.mock_summarizer.summarize_pursuit.assert_called_once_with(pursuit_result)

            # Verify output was sent to channel
            self.mock_channel.output.assert_called_with("Goal completed successfully")

    async def test_pursue_goal_with_no_completed_steps(self):
        """Test goal pursuit when no steps are completed."""
        goal = "Test goal with no completion"

        # Set up mock pursuit result with no completed steps
        execution_plan = ExecutionPlan(
            goal=goal,
            pending_steps=[],
            completed_steps=[]
        )
        
        pursuit_result = GoalPursuitResult(
            goal=goal,
            status=PursuitStatus.FAILED,
            records=execution_plan,
            started_at=datetime.now(),
            completed_at=datetime.now()
        )
        
        # Mock GoalPursuit.initialize_pursuit and run_to_completion
        with unittest.mock.patch('novelrag.agent.agent.GoalPursuit') as mock_pursuit_class:
            mock_pursuit_instance = MagicMock()
            mock_pursuit_instance.plan.pending_steps = []
            mock_pursuit_instance.run_to_completion = AsyncMock(return_value=pursuit_result)
            mock_pursuit_class.initialize_pursuit = AsyncMock(return_value=mock_pursuit_instance)

            # Execute the goal
            await self.agent.pursue_goal(goal)

            # Verify error was sent to channel
            self.mock_channel.error.assert_called_with(f'No steps completed for goal "{goal}".')

            # Verify summarizer was not called
            self.mock_summarizer.summarize_pursuit.assert_not_called()

    async def test_contextual_tools_setup(self):
        """Test that contextual tools are properly set up from regular tools."""
        # Test that ContextualTool instances are added directly
        test_tool = self.agent.contextual_tools["test_tool"]
        self.assertIsInstance(test_tool, MockContextualTool)
        self.assertEqual(test_tool.name, "test_tool")
        self.assertEqual(test_tool.description, "A test tool")


if __name__ == '__main__':
    unittest.main()