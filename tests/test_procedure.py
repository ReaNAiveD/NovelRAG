"""Tests for the novelrag.agenturn.procedure module and procedure classes."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from novelrag.agenturn.procedure import (
    ExecutionContext,
    LoggingExecutionContext,
    ProcedureError,
)


# ---------------------------------------------------------------------------
# ProcedureError
# ---------------------------------------------------------------------------


class TestProcedureError:
    def test_basic_error(self):
        err = ProcedureError("something went wrong")
        assert str(err) == "something went wrong"
        assert err.message == "something went wrong"
        assert err.effects == []

    def test_error_with_effects(self):
        effects = ["created resource A", "updated resource B"]
        err = ProcedureError("failed at step 3", effects=effects)
        assert err.effects == effects
        assert err.message == "failed at step 3"

    def test_error_is_exception(self):
        err = ProcedureError("boom")
        assert isinstance(err, Exception)

    def test_error_can_be_raised_and_caught(self):
        with pytest.raises(ProcedureError) as exc_info:
            raise ProcedureError("oops", effects=["effect1"])
        assert exc_info.value.effects == ["effect1"]

    def test_effects_default_is_independent(self):
        """Each ProcedureError should get its own effects list."""
        a = ProcedureError("a")
        b = ProcedureError("b")
        a.effects.append("x")
        assert b.effects == []


# ---------------------------------------------------------------------------
# ExecutionContext (abstract)
# ---------------------------------------------------------------------------


class TestExecutionContext:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            ExecutionContext()

    def test_has_all_three_facets(self):
        """ExecutionContext defines messaging, output, and bidirectional methods."""
        import inspect
        members = {name for name, _ in inspect.getmembers(ExecutionContext, predicate=inspect.isfunction)}
        # Messaging
        assert "debug" in members
        assert "info" in members
        assert "warning" in members
        assert "error" in members
        # Output
        assert "output" in members
        # Bidirectional
        assert "confirm" in members
        assert "request" in members

    def test_tool_runtime_is_alias(self):
        """ToolRuntime should be a backward-compatible alias for ExecutionContext."""
        from novelrag.agenturn.tool.runtime import ToolRuntime
        assert ToolRuntime is ExecutionContext

    def test_agent_channel_extends_execution_context(self):
        """AgentChannel should be a subclass of ExecutionContext."""
        from novelrag.agenturn.channel import AgentChannel
        assert issubclass(AgentChannel, ExecutionContext)


# ---------------------------------------------------------------------------
# LoggingExecutionContext
# ---------------------------------------------------------------------------


class TestLoggingExecutionContext:
    @pytest.mark.asyncio
    async def test_debug_delegates_to_logger(self):
        mock_logger = MagicMock(spec=logging.Logger)
        ctx = LoggingExecutionContext(mock_logger)
        await ctx.debug("debug message")
        mock_logger.debug.assert_called_once_with("debug message")

    @pytest.mark.asyncio
    async def test_info_delegates_to_logger(self):
        mock_logger = MagicMock(spec=logging.Logger)
        ctx = LoggingExecutionContext(mock_logger)
        await ctx.info("info message")
        mock_logger.info.assert_called_once_with("info message")

    @pytest.mark.asyncio
    async def test_warning_delegates_to_logger(self):
        mock_logger = MagicMock(spec=logging.Logger)
        ctx = LoggingExecutionContext(mock_logger)
        await ctx.warning("warning message")
        mock_logger.warning.assert_called_once_with("warning message")

    @pytest.mark.asyncio
    async def test_error_delegates_to_logger(self):
        mock_logger = MagicMock(spec=logging.Logger)
        ctx = LoggingExecutionContext(mock_logger)
        await ctx.error("error message")
        mock_logger.error.assert_called_once_with("error message")

    @pytest.mark.asyncio
    async def test_output_delegates_to_logger_info(self):
        mock_logger = MagicMock(spec=logging.Logger)
        ctx = LoggingExecutionContext(mock_logger)
        await ctx.output("user output")
        mock_logger.info.assert_called_once_with("user output")

    @pytest.mark.asyncio
    async def test_confirm_auto_confirms(self):
        ctx = LoggingExecutionContext()
        result = await ctx.confirm("Proceed?")
        assert result is True

    @pytest.mark.asyncio
    async def test_request_returns_empty_string(self):
        ctx = LoggingExecutionContext()
        result = await ctx.request("Enter value:")
        assert result == ""

    @pytest.mark.asyncio
    async def test_default_logger(self):
        """When no logger is provided, a default should be used without error."""
        ctx = LoggingExecutionContext()
        # Should not raise
        await ctx.debug("test")
        await ctx.info("test")
        await ctx.warning("test")
        await ctx.error("test")
        await ctx.output("test")
        await ctx.confirm("test?")
        await ctx.request("test?")


# ---------------------------------------------------------------------------
# Procedure classes (ContextDiscoveryLoop, ActionLoop, ActionDetermineLoop)
# ---------------------------------------------------------------------------


class TestConvertToOrchestrationAction:
    """Test the module-level _convert_to_orchestration_action helper."""

    def test_execute_decision(self):
        from novelrag.resource_agent.action_determine.action_determine_loop import (
            _convert_to_orchestration_action,
            ActionDecision,
            ExecutionDetail,
        )
        from novelrag.agenturn.step import OperationPlan

        decision = ActionDecision(
            situation_analysis="ready to act",
            decision_type="execute",
            execution=ExecutionDetail(
                tool="resource_write",
                params={"key": "value"},
                confidence="high",
                reasoning="test",
            ),
        )
        result = _convert_to_orchestration_action(decision)
        assert isinstance(result, OperationPlan)
        assert result.tool == "resource_write"
        assert result.parameters == {"key": "value"}

    def test_finalize_decision(self):
        from novelrag.resource_agent.action_determine.action_determine_loop import (
            _convert_to_orchestration_action,
            ActionDecision,
            FinalizationDetail,
        )
        from novelrag.agenturn.step import Resolution

        decision = ActionDecision(
            situation_analysis="done",
            decision_type="finalize",
            finalization=FinalizationDetail(
                status="success",
                response="All done.",
            ),
        )
        result = _convert_to_orchestration_action(decision)
        assert isinstance(result, Resolution)
        assert result.status == "success"
        assert result.response == "All done."

    def test_invalid_decision(self):
        from novelrag.resource_agent.action_determine.action_determine_loop import (
            _convert_to_orchestration_action,
            ActionDecision,
        )
        from novelrag.agenturn.step import Resolution

        decision = ActionDecision(
            situation_analysis="confused",
            decision_type="unknown",
        )
        result = _convert_to_orchestration_action(decision)
        assert isinstance(result, Resolution)
        assert result.status == "failed"


class TestGetToolMap:
    def test_expanded_tools(self):
        from novelrag.resource_agent.action_determine.action_determine_loop import _get_tool_map

        tool_a = MagicMock()
        tool_b = MagicMock()
        available = {"a": tool_a, "b": tool_b}
        expanded = {"a"}

        result = _get_tool_map(available, expanded, expanded=True)
        assert result == {"a": tool_a}

    def test_collapsed_tools(self):
        from novelrag.resource_agent.action_determine.action_determine_loop import _get_tool_map

        tool_a = MagicMock()
        tool_b = MagicMock()
        available = {"a": tool_a, "b": tool_b}
        expanded = {"a"}

        result = _get_tool_map(available, expanded, expanded=False)
        assert result == {"b": tool_b}


class TestContextDiscoveryLoop:
    """Test ContextDiscoveryLoop as a procedure with flat execute params."""

    @pytest.mark.asyncio
    async def test_single_iteration_no_refinement(self):
        """When discoverer returns no refinement needed, loop exits after one iteration."""
        from novelrag.resource_agent.action_determine.action_determine_loop import (
            ContextDiscoveryLoop,
            DiscoveryPlan,
        )

        mock_context = AsyncMock()
        mock_context.snapshot.return_value = MagicMock(
            segments=[], nonexistent_uris=[]
        )
        mock_context.search_history = []

        mock_discoverer = AsyncMock()
        mock_discoverer.discover.return_value = DiscoveryPlan(
            discovery_analysis="Nothing needed",
            search_queries=[],
            query_resources=[],
            expand_tools=[],
        )

        expanded_tools: set[str] = set()
        loop = ContextDiscoveryLoop(
            context=mock_context,
            discoverer=mock_discoverer,
            analyser=AsyncMock(),
            expanded_tools=expanded_tools,
            max_iter=5,
        )

        mock_goal = MagicMock()
        mock_assessment = MagicMock()
        ctx = LoggingExecutionContext()

        result = await loop.execute(mock_goal, mock_assessment, {}, 0, ctx)
        assert result == 1
        mock_discoverer.discover.assert_called_once()

    @pytest.mark.asyncio
    async def test_discovery_expands_tools(self):
        """Tools listed in expand_tools should be added to the shared set."""
        from novelrag.resource_agent.action_determine.action_determine_loop import (
            ContextDiscoveryLoop,
            DiscoveryPlan,
        )

        mock_context = AsyncMock()
        mock_context.snapshot.return_value = MagicMock(
            segments=[], nonexistent_uris=[]
        )
        mock_context.search_history = []

        mock_discoverer = AsyncMock()
        mock_discoverer.discover.return_value = DiscoveryPlan(
            discovery_analysis="Expanding tools",
            search_queries=[],
            query_resources=[],
            expand_tools=["resource_write"],
        )

        expanded_tools: set[str] = set()
        loop = ContextDiscoveryLoop(
            context=mock_context,
            discoverer=mock_discoverer,
            analyser=AsyncMock(),
            expanded_tools=expanded_tools,
            max_iter=5,
        )

        ctx = LoggingExecutionContext()
        await loop.execute(MagicMock(), MagicMock(), {}, 0, ctx)
        assert "resource_write" in expanded_tools


class TestActionDetermineLoop:
    """Test ActionDetermineLoop satisfies both Procedure and ActionDeterminer patterns."""

    @pytest.mark.asyncio
    async def test_determine_action_delegates_to_execute(self):
        """determine_action should delegate to execute with a LoggingExecutionContext."""
        from novelrag.resource_agent.action_determine.action_determine_loop import (
            ActionDetermineLoop,
            ActionDecision,
            FinalizationDetail,
            DiscoveryPlan,
            RefinementDecision,
            RefinementApproval,
        )
        from novelrag.agenturn.pursuit import PursuitAssessment, PursuitProgress
        from novelrag.agenturn.goal import Goal, UserRequestSource

        goal = Goal(
            description="test goal",
            source=UserRequestSource(request="test"),
        )
        pursuit = PursuitProgress(goal=goal)
        assessment = PursuitAssessment(
            finished_tasks=[],
            remaining_work_summary="test",
            required_context="test",
            expected_actions="test",
            boundary_conditions=[],
            exception_conditions=[],
            success_criteria=[],
        )

        # Mock collaborators
        mock_context = AsyncMock()
        mock_context.snapshot.return_value = MagicMock(
            segments=[], nonexistent_uris=[]
        )
        mock_context.search_history = []

        mock_assessor = AsyncMock()
        mock_assessor.assess_progress.return_value = assessment

        mock_discoverer = AsyncMock()
        mock_discoverer.discover.return_value = DiscoveryPlan(
            discovery_analysis="done",
            search_queries=[],
            query_resources=[],
            expand_tools=[],
        )

        mock_decider = AsyncMock()
        mock_decider.decide.return_value = ActionDecision(
            situation_analysis="ready",
            decision_type="finalize",
            finalization=FinalizationDetail(
                status="success",
                response="Done.",
            ),
        )

        mock_refiner = AsyncMock()
        mock_refiner.analyze.return_value = RefinementDecision(
            analysis="looks good",
            verdict="approve",
            approval=RefinementApproval(ready=True, confidence="high"),
        )

        loop = ActionDetermineLoop(
            context=mock_context,
            pursuit_assessor=mock_assessor,
            discoverer=mock_discoverer,
            analyser=AsyncMock(),
            decider=mock_decider,
            refiner=mock_refiner,
        )

        # Call determine_action (backward compat interface)
        result = await loop.determine_action(
            beliefs=["test belief"],
            pursuit_progress=pursuit,
            available_tools={},
        )

        from novelrag.agenturn.step import Resolution
        assert isinstance(result, Resolution)
        assert result.status == "success"
        mock_assessor.assess_progress.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_with_custom_context(self):
        """execute accepts a custom ExecutionContext."""
        from novelrag.resource_agent.action_determine.action_determine_loop import (
            ActionDetermineLoop,
            ActionDecision,
            FinalizationDetail,
            DiscoveryPlan,
            RefinementDecision,
            RefinementApproval,
        )
        from novelrag.agenturn.pursuit import PursuitAssessment, PursuitProgress
        from novelrag.agenturn.goal import Goal, UserRequestSource

        goal = Goal(
            description="test",
            source=UserRequestSource(request="test"),
        )
        pursuit = PursuitProgress(goal=goal)
        assessment = PursuitAssessment(
            finished_tasks=[],
            remaining_work_summary="",
            required_context="",
            expected_actions="",
            boundary_conditions=[],
            exception_conditions=[],
            success_criteria=[],
        )

        mock_context = AsyncMock()
        mock_context.snapshot.return_value = MagicMock(
            segments=[], nonexistent_uris=[]
        )
        mock_context.search_history = []

        mock_assessor = AsyncMock()
        mock_assessor.assess_progress.return_value = assessment

        mock_discoverer = AsyncMock()
        mock_discoverer.discover.return_value = DiscoveryPlan(
            discovery_analysis="done",
            search_queries=[],
            query_resources=[],
            expand_tools=[],
        )

        mock_decider = AsyncMock()
        mock_decider.decide.return_value = ActionDecision(
            situation_analysis="ready",
            decision_type="finalize",
            finalization=FinalizationDetail(
                status="success",
                response="Done.",
            ),
        )

        mock_refiner = AsyncMock()
        mock_refiner.analyze.return_value = RefinementDecision(
            analysis="ok",
            verdict="approve",
            approval=RefinementApproval(ready=True, confidence="high"),
        )

        loop = ActionDetermineLoop(
            context=mock_context,
            pursuit_assessor=mock_assessor,
            discoverer=mock_discoverer,
            analyser=AsyncMock(),
            decider=mock_decider,
            refiner=mock_refiner,
        )

        # Use a custom execution context
        custom_ctx = AsyncMock(spec=ExecutionContext)
        result = await loop.execute(
            beliefs=[],
            pursuit_progress=pursuit,
            available_tools={},
            ctx=custom_ctx,
        )

        from novelrag.agenturn.step import Resolution
        assert isinstance(result, Resolution)
        # Verify the custom context received messages
        assert custom_ctx.info.call_count > 0
