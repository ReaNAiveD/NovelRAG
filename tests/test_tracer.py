"""Tests for the novelrag.tracer framework."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import yaml

from novelrag.tracer.span import Span, SpanKind, get_current_span, set_current_span, _current_span
from novelrag.tracer.tracer import (
    get_active_tracer,
    set_active_tracer,
    _active_tracer,
)
from novelrag.tracer.tracer import Tracer
from novelrag.tracer.exporter import YAMLExporter
from novelrag.tracer.callback import TracerCallbackHandler, _serialize_messages
from novelrag.tracer.decorators import (
    trace_session,
    trace_intent,
    trace_pursuit,
    trace_tool,
    trace_llm,
)


# ---------------------------------------------------------------------------
# Span
# ---------------------------------------------------------------------------


class TestSpan:
    def test_create_span(self):
        span = Span(kind=SpanKind.SESSION, name="test")
        assert span.kind == SpanKind.SESSION
        assert span.name == "test"
        assert span.status == "ok"
        assert span.children == []
        assert span.parent is None
        assert len(span.span_id) == 12

    def test_set_attribute(self):
        span = Span(kind=SpanKind.LLM_CALL, name="llm")
        span.set_attribute("model", "gpt-4")
        assert span.attributes["model"] == "gpt-4"

    def test_add_child(self):
        parent = Span(kind=SpanKind.SESSION, name="session")
        child = Span(kind=SpanKind.INTENT, name="intent")
        parent.add_child(child)
        assert child.parent is parent
        assert child in parent.children

    def test_finish_ok(self):
        span = Span(kind=SpanKind.LLM_CALL, name="test")
        span.finish()
        assert span.end_time is not None
        assert span.duration_ms is not None
        assert span.duration_ms >= 0
        assert span.status == "ok"

    def test_finish_error(self):
        span = Span(kind=SpanKind.LLM_CALL, name="test")
        span.finish(error=ValueError("boom"))
        assert span.status == "error"
        assert span.error == "boom"

    def test_to_dict(self):
        parent = Span(kind=SpanKind.SESSION, name="s")
        child = Span(kind=SpanKind.LLM_CALL, name="llm")
        child.set_attribute("model", "gpt-4")
        parent.add_child(child)
        child.finish()
        parent.finish()

        d = parent.to_dict()
        assert d["kind"] == "session"
        assert d["name"] == "s"
        assert d["status"] == "ok"
        assert len(d["children"]) == 1
        assert d["children"][0]["kind"] == "llm_call"
        assert d["children"][0]["attributes"]["model"] == "gpt-4"

    def test_to_dict_no_children(self):
        span = Span(kind=SpanKind.LLM_CALL, name="llm")
        span.finish()
        d = span.to_dict()
        assert "children" not in d


# ---------------------------------------------------------------------------
# Context vars
# ---------------------------------------------------------------------------


class TestContext:
    def test_default_none(self):
        assert get_current_span() is None
        assert get_active_tracer() is None

    def test_set_and_reset_span(self):
        span = Span(kind=SpanKind.LLM_CALL, name="test")
        token = set_current_span(span)
        assert get_current_span() is span
        _current_span.reset(token)
        assert get_current_span() is None

    def test_set_and_reset_tracer(self):
        tracer = Tracer()
        token = set_active_tracer(tracer)
        assert get_active_tracer() is tracer
        _active_tracer.reset(token)
        assert get_active_tracer() is None


# ---------------------------------------------------------------------------
# Tracer
# ---------------------------------------------------------------------------


class TestTracer:
    def test_activate_deactivate(self):
        tracer = Tracer()
        token = tracer.activate()
        assert get_active_tracer() is tracer
        tracer.deactivate(token)
        assert get_active_tracer() is None

    def test_start_end_span(self):
        tracer = Tracer()
        span, token = tracer.start_span(SpanKind.SESSION, "root")
        assert get_current_span() is span
        assert tracer.session_span is span

        child_span, child_token = tracer.start_span(SpanKind.INTENT, "intent")
        assert get_current_span() is child_span
        assert child_span.parent is span
        assert child_span in span.children

        tracer.end_span(child_span, child_token)
        assert get_current_span() is span
        assert child_span.duration_ms is not None

        tracer.end_span(span, token)
        assert get_current_span() is None

    def test_start_span_error(self):
        tracer = Tracer()
        span, token = tracer.start_span(SpanKind.LLM_CALL, "test")
        tracer.end_span(span, token, error=RuntimeError("fail"))
        assert span.status == "error"
        assert span.error == "fail"

    @pytest.mark.asyncio
    async def test_llm_span_context_manager(self):
        tracer = Tracer()
        parent, parent_token = tracer.start_span(SpanKind.PURSUIT, "pursuit")
        async with tracer.llm_span("test_llm") as span:
            assert get_current_span() is span
            assert span.kind == SpanKind.LLM_CALL
            assert span.parent is parent
        # Should be restored to parent
        assert get_current_span() is parent
        assert span.status == "ok"
        assert span.duration_ms is not None
        tracer.end_span(parent, parent_token)

    @pytest.mark.asyncio
    async def test_llm_span_error(self):
        tracer = Tracer()
        parent, parent_token = tracer.start_span(SpanKind.PURSUIT, "pursuit")
        with pytest.raises(ValueError, match="boom"):
            async with tracer.llm_span("test_llm") as span:
                raise ValueError("boom")
        assert span.status == "error"
        assert span.error == "boom"
        assert get_current_span() is parent
        tracer.end_span(parent, parent_token)

    def test_callback_handler_property(self):
        tracer = Tracer()
        handler = tracer.callback_handler
        assert isinstance(handler, TracerCallbackHandler)
        # Same instance on repeated access
        assert tracer.callback_handler is handler


# ---------------------------------------------------------------------------
# YAMLExporter
# ---------------------------------------------------------------------------


class TestYAMLExporter:
    def test_export_creates_file(self, tmp_path: Path):
        exporter = YAMLExporter(output_dir=tmp_path)
        root = Span(kind=SpanKind.SESSION, name="session")
        child = Span(kind=SpanKind.LLM_CALL, name="llm")
        child.set_attribute("model", "gpt-4")
        child.finish()
        root.add_child(child)
        root.finish()

        path = exporter.export(root, filename="test_trace.yaml")
        assert path.exists()

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        assert data["kind"] == "session"
        assert data["name"] == "session"
        assert len(data["children"]) == 1
        assert data["children"][0]["name"] == "llm"

    def test_export_auto_filename(self, tmp_path: Path):
        exporter = YAMLExporter(output_dir=tmp_path)
        root = Span(kind=SpanKind.SESSION, name="s")
        root.finish()
        path = exporter.export(root)
        assert path.name.startswith("trace_")
        assert path.suffix == ".yaml"


# ---------------------------------------------------------------------------
# TracerCallbackHandler
# ---------------------------------------------------------------------------


class TestTracerCallbackHandler:
    @pytest.mark.asyncio
    async def test_on_chat_model_start_populates_span(self):
        handler = TracerCallbackHandler()
        span = Span(kind=SpanKind.LLM_CALL, name="test")
        token = set_current_span(span)

        # Simulate LangChain message objects
        msg = MagicMock()
        msg.type = "human"
        msg.content = "Hello"

        run_id = uuid4()
        await handler.on_chat_model_start(
            serialized={"id": ["langchain", "chat_models", "openai", "ChatOpenAI"],
                        "kwargs": {"model_name": "gpt-4"}},
            messages=[[msg]],
            run_id=run_id,
        )

        assert span.attributes["model"] == "gpt-4"
        assert span.attributes["request"]["messages"][0]["role"] == "human"
        assert span.attributes["request"]["messages"][0]["content"] == "Hello"

        _current_span.reset(token)

    @pytest.mark.asyncio
    async def test_on_llm_end_populates_span(self):
        handler = TracerCallbackHandler()
        span = Span(kind=SpanKind.LLM_CALL, name="test")
        token = set_current_span(span)

        run_id = uuid4()
        # Register the span
        msg = MagicMock()
        msg.type = "system"
        msg.content = "prompt"
        await handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"], "kwargs": {}},
            messages=[[msg]],
            run_id=run_id,
        )

        # Simulate LLM response
        gen = MagicMock()
        gen.text = "response text"
        gen.message = MagicMock()
        gen.message.content = "response from message"
        result = MagicMock()
        result.generations = [[gen]]
        result.llm_output = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            }
        }

        await handler.on_llm_end(response=result, run_id=run_id)

        assert span.attributes["response"]["content"] == "response from message"
        assert span.attributes["token_usage"]["prompt_tokens"] == 100
        assert span.attributes["token_usage"]["completion_tokens"] == 50

        _current_span.reset(token)

    @pytest.mark.asyncio
    async def test_on_llm_error_marks_span(self):
        handler = TracerCallbackHandler()
        span = Span(kind=SpanKind.LLM_CALL, name="test")
        token = set_current_span(span)

        run_id = uuid4()
        msg = MagicMock()
        msg.type = "system"
        msg.content = "prompt"
        await handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"], "kwargs": {}},
            messages=[[msg]],
            run_id=run_id,
        )

        await handler.on_llm_error(
            error=RuntimeError("API error"),
            run_id=run_id,
        )

        assert span.status == "error"
        assert "API error" in span.error

        _current_span.reset(token)

    @pytest.mark.asyncio
    async def test_ignores_non_llm_call_span(self):
        handler = TracerCallbackHandler()
        # Set a non-LLM_CALL span
        span = Span(kind=SpanKind.PURSUIT, name="pursuit")
        token = set_current_span(span)

        run_id = uuid4()
        msg = MagicMock()
        msg.type = "system"
        msg.content = "prompt"
        await handler.on_chat_model_start(
            serialized={"id": ["ChatOpenAI"], "kwargs": {}},
            messages=[[msg]],
            run_id=run_id,
        )

        # Should NOT have recorded anything
        assert "request" not in span.attributes

        _current_span.reset(token)


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


class TestDecorators:
    @pytest.mark.asyncio
    async def test_trace_llm_creates_span(self):
        tracer = Tracer()
        token = tracer.activate()

        # Push a parent span (simulating a pursuit)
        parent, parent_token = tracer.start_span(SpanKind.PURSUIT, "parent")

        @trace_llm("my_llm_call")
        async def my_func():
            current = get_current_span()
            assert current is not None
            assert current.kind == SpanKind.LLM_CALL
            assert current.name == "my_llm_call"
            return "result"

        result = await my_func()
        assert result == "result"

        # After return, current span should be parent again
        assert get_current_span() is parent

        # Parent should have a child
        assert len(parent.children) == 1
        assert parent.children[0].name == "my_llm_call"
        assert parent.children[0].status == "ok"
        assert parent.children[0].duration_ms is not None

        tracer.end_span(parent, parent_token)
        tracer.deactivate(token)

    @pytest.mark.asyncio
    async def test_trace_llm_error(self):
        tracer = Tracer()
        token = tracer.activate()
        parent, parent_token = tracer.start_span(SpanKind.PURSUIT, "parent")

        @trace_llm("failing_call")
        async def failing():
            raise ValueError("fail!")

        with pytest.raises(ValueError, match="fail!"):
            await failing()

        assert parent.children[0].status == "error"
        assert parent.children[0].error == "fail!"
        assert get_current_span() is parent

        tracer.end_span(parent, parent_token)
        tracer.deactivate(token)

    @pytest.mark.asyncio
    async def test_no_tracer_passthrough(self):
        """When no tracer is active, decorated functions run normally."""
        assert get_active_tracer() is None

        @trace_llm("test")
        async def my_func():
            return 42

        result = await my_func()
        assert result == 42

    @pytest.mark.asyncio
    async def test_trace_intent(self):
        tracer = Tracer()
        token = tracer.activate()
        session, session_token = tracer.start_span(SpanKind.SESSION, "session")

        @trace_intent("my_intent")
        async def intent_func():
            current = get_current_span()
            assert current.kind == SpanKind.INTENT
            assert current.name == "my_intent"

        await intent_func()
        assert len(session.children) == 1
        assert session.children[0].kind == SpanKind.INTENT

        tracer.end_span(session, session_token)
        tracer.deactivate(token)

    @pytest.mark.asyncio
    async def test_trace_pursuit(self):
        tracer = Tracer()
        token = tracer.activate()
        parent, parent_token = tracer.start_span(SpanKind.INTENT, "intent")

        @trace_pursuit("my_pursuit")
        async def pursuit_func():
            current = get_current_span()
            assert current.kind == SpanKind.PURSUIT

        await pursuit_func()
        assert parent.children[0].kind == SpanKind.PURSUIT

        tracer.end_span(parent, parent_token)
        tracer.deactivate(token)

    @pytest.mark.asyncio
    async def test_trace_tool_name_from_kwarg(self):
        tracer = Tracer()
        token = tracer.activate()
        parent, parent_token = tracer.start_span(SpanKind.PURSUIT, "pursuit")

        @trace_tool()
        async def execute_tool(self, tool_name: str, params: dict):
            current = get_current_span()
            assert current.kind == SpanKind.TOOL_CALL
            assert current.name == "resource_write"

        await execute_tool(None, tool_name="resource_write", params={})

        assert parent.children[0].name == "resource_write"

        tracer.end_span(parent, parent_token)
        tracer.deactivate(token)

    @pytest.mark.asyncio
    async def test_trace_session_auto_export(self, tmp_path: Path):
        exporter = YAMLExporter(output_dir=tmp_path)
        tracer = Tracer(exporter=exporter)
        token = tracer.activate()

        @trace_session("test_session")
        async def session_func():
            current = get_current_span()
            assert current.kind == SpanKind.SESSION
            assert current.name == "test_session"

        await session_func()

        # Should have auto-exported
        files = list(tmp_path.glob("trace_*.yaml"))
        assert len(files) == 1

        with open(files[0], "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        assert data["kind"] == "session"
        assert data["name"] == "test_session"

        tracer.deactivate(token)

    @pytest.mark.asyncio
    async def test_nested_hierarchy(self, tmp_path: Path):
        """Full hierarchy: session → intent → pursuit → tool → llm."""
        exporter = YAMLExporter(output_dir=tmp_path)
        tracer = Tracer(exporter=exporter)
        token = tracer.activate()

        @trace_llm("my_llm")
        async def llm_func():
            return "content"

        @trace_tool("my_tool")
        async def tool_func():
            return await llm_func()

        @trace_pursuit("my_pursuit")
        async def pursuit_func():
            return await tool_func()

        @trace_intent("my_intent")
        async def intent_func():
            return await pursuit_func()

        @trace_session("my_session")
        async def session_func():
            return await intent_func()

        result = await session_func()
        assert result == "content"

        # Verify exported YAML
        files = list(tmp_path.glob("trace_*.yaml"))
        assert len(files) == 1
        with open(files[0], "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        # Walk the tree
        assert data["kind"] == "session"
        intent = data["children"][0]
        assert intent["kind"] == "intent"
        pursuit = intent["children"][0]
        assert pursuit["kind"] == "pursuit"
        tool = pursuit["children"][0]
        assert tool["kind"] == "tool_call"
        llm = tool["children"][0]
        assert llm["kind"] == "llm_call"
        assert llm["name"] == "my_llm"

        tracer.deactivate(token)
