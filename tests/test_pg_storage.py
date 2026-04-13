"""Tests for the PostgreSQL storage backend.

These tests require a running PostgreSQL instance with pgvector installed.
Set the ``NOVELRAG_TEST_PG_URL`` environment variable to a connection string,
e.g.::

    NOVELRAG_TEST_PG_URL=postgresql://user:pass@localhost:5432/novelrag_test

Tests are skipped automatically when the variable is not set.

The test database must already have the schema applied (``alembic upgrade head``)
and a list partition for workspace 999999::

    CREATE TABLE resource_elements_ws_999999
      PARTITION OF resource_elements FOR VALUES IN (999999);
"""

import os

import pytest
from langchain_core.embeddings import Embeddings

# Skip the entire module when no PG URL is configured.
PG_URL = os.environ.get("NOVELRAG_TEST_PG_URL")
pytestmark = pytest.mark.skipif(PG_URL is None, reason="NOVELRAG_TEST_PG_URL not set")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockEmbedder(Embeddings):
    """Deterministic embedder for testing (3072-dim zero vector)."""

    def __init__(self, dimension: int = 3072):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i % 7) * 0.01] * self.dimension for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.5] * self.dimension

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

WORKSPACE_ID = 999_999  # unlikely to collide


@pytest.fixture(scope="session")
async def pool():
    from psycopg_pool import AsyncConnectionPool

    p = AsyncConnectionPool(conninfo=PG_URL, min_size=1, max_size=5, open=False)
    await p.open()
    yield p
    await p.close()


@pytest.fixture()
async def clean_workspace(pool):
    """Ensure the workspace tables are empty before each test."""
    async with pool.connection() as conn, conn.transaction(), conn.cursor() as cur:
        for table in (
            "redo_items",
            "undo_items",
            "undo_redo_table",
            "backlog_entries",
            "resource_elements",
            "resource_aspects",
        ):
            await cur.execute(
                f"DELETE FROM {table} WHERE workspace_id = %(ws)s",
                {"ws": WORKSPACE_ID},
            )


@pytest.fixture()
def embedder():
    return MockEmbedder()


@pytest.fixture()
def resource_repo(pool, embedder, clean_workspace):
    from novelrag.storage.pg.resource import PostgresResourceRepository

    return PostgresResourceRepository(WORKSPACE_ID, pool, embedder)


@pytest.fixture()
def undo_queue(pool, clean_workspace):
    from novelrag.storage.pg.undo import PostgresUndoQueue

    return PostgresUndoQueue(WORKSPACE_ID, pool)


@pytest.fixture()
def backlog(pool, clean_workspace):
    from novelrag.storage.pg.backlog import PostgresBacklog

    return PostgresBacklog(WORKSPACE_ID, pool)


# ===================================================================
# ResourceRepository tests
# ===================================================================


class TestPostgresResourceRepository:
    async def test_add_and_get_aspect(self, resource_repo):
        aspect = await resource_repo.add_aspect(
            "character",
            {"description": "Characters", "children_keys": [], "root_element_names": []},
        )
        assert aspect.name == "character"
        assert aspect.description == "Characters"

        fetched = await resource_repo.get_aspect("character")
        assert fetched is not None
        assert fetched.name == "character"

    async def test_all_aspects(self, resource_repo):
        await resource_repo.add_aspect("a1", {"children_keys": []})
        await resource_repo.add_aspect("a2", {"children_keys": []})
        names = [a.name for a in await resource_repo.all_aspects()]
        assert "a1" in names
        assert "a2" in names

    async def test_remove_aspect(self, resource_repo):
        await resource_repo.add_aspect("temp", {"children_keys": []})
        result = await resource_repo.remove_aspect("temp")
        assert result is not None
        assert result.aspect.name == "temp"
        assert await resource_repo.get_aspect("temp") is None

    async def test_remove_nonexistent_aspect(self, resource_repo):
        result = await resource_repo.remove_aspect("nonexistent")
        assert result is None

    async def test_find_by_uri_root(self, resource_repo):
        await resource_repo.add_aspect("char", {"children_keys": []})
        result = await resource_repo.find_by_uri("/")
        assert isinstance(result, list)
        assert "char" in result

    async def test_find_by_uri_aspect(self, resource_repo):
        await resource_repo.add_aspect("char", {"description": "chars", "children_keys": []})
        result = await resource_repo.find_by_uri("/char")
        assert result is not None
        assert result.name == "char"

    async def test_apply_resource_op_insert(self, resource_repo):
        from novelrag.resource.operation import ResourceLocation, ResourceOperation

        await resource_repo.add_aspect("char", {"children_keys": [], "root_element_names": []})
        op = ResourceOperation.new(
            ResourceLocation.aspect("char"),
            data=[{"id": "alice", "name": "Alice", "age": 30}],
        )
        undo = await resource_repo.apply(op)
        assert undo is not None

        elem = await resource_repo.find_by_uri("/char/alice")
        assert elem is not None
        assert elem.id == "alice"

    async def test_apply_property_op(self, resource_repo):
        from novelrag.resource.operation import PropertyOperation, ResourceLocation, ResourceOperation

        await resource_repo.add_aspect("char", {"children_keys": [], "root_element_names": []})
        await resource_repo.apply(
            ResourceOperation.new(
                ResourceLocation.aspect("char"),
                data=[{"id": "bob", "name": "Bob", "age": 25}],
            )
        )
        undo_op = await resource_repo.apply(PropertyOperation.new("/char/bob", {"age": 26}))
        elem = await resource_repo.find_by_uri("/char/bob")
        assert elem["age"] == 26
        # undo should have old value
        assert undo_op.data.get("age") == 25

    async def test_apply_resource_op_delete(self, resource_repo):
        from novelrag.resource.operation import ResourceLocation, ResourceOperation

        await resource_repo.add_aspect("char", {"children_keys": [], "root_element_names": []})
        await resource_repo.apply(
            ResourceOperation.new(
                ResourceLocation.aspect("char"),
                data=[{"id": "carol", "name": "Carol"}],
            )
        )
        undo = await resource_repo.apply(
            ResourceOperation.new(ResourceLocation.aspect("char"), start=0, end=1),
        )
        assert undo.data is not None and len(undo.data) == 1
        assert await resource_repo.find_by_uri("/char/carol") is None

    async def test_vector_search(self, resource_repo):
        from novelrag.resource.operation import ResourceLocation, ResourceOperation

        await resource_repo.add_aspect("char", {"children_keys": [], "root_element_names": []})
        await resource_repo.apply(
            ResourceOperation.new(
                ResourceLocation.aspect("char"),
                data=[{"id": "dave", "name": "Dave", "description": "A warrior"}],
            )
        )
        results = await resource_repo.vector_search("warrior", aspect="char", limit=5)
        assert len(results) > 0
        assert results[0].element.id == "dave"

    async def test_update_relationships(self, resource_repo):
        from novelrag.resource.operation import ResourceLocation, ResourceOperation

        await resource_repo.add_aspect("char", {"children_keys": [], "root_element_names": []})
        await resource_repo.apply(
            ResourceOperation.new(
                ResourceLocation.aspect("char"),
                data=[
                    {"id": "eve", "name": "Eve"},
                    {"id": "frank", "name": "Frank"},
                ],
            )
        )
        old = await resource_repo.update_relationships("/char/eve", "/char/frank", ["friend"])
        assert old == []
        elem = await resource_repo.find_by_uri("/char/eve")
        assert "/char/frank" in elem.relationships

    async def test_iter_elements(self, resource_repo):
        from novelrag.resource.operation import ResourceLocation, ResourceOperation

        await resource_repo.add_aspect("char", {"children_keys": [], "root_element_names": []})
        await resource_repo.apply(
            ResourceOperation.new(
                ResourceLocation.aspect("char"),
                data=[{"id": "g1"}, {"id": "g2"}],
            )
        )
        elems = await resource_repo.iter_elements("char")
        assert len(elems) == 2

    async def test_add_aspect_with_elements_restore(self, resource_repo):
        """Test add_aspect with pre-built element dicts (undo restore path)."""
        from novelrag.resource.operation import ResourceLocation, ResourceOperation

        await resource_repo.add_aspect("char", {"children_keys": [], "root_element_names": ["h1"]})
        await resource_repo.apply(
            ResourceOperation.new(
                ResourceLocation.aspect("char"),
                data=[{"id": "h1", "name": "H1"}],
            )
        )
        removed = await resource_repo.remove_aspect("char")
        assert removed is not None
        # Re-add with saved elements
        restored = await resource_repo.add_aspect(
            "char",
            {
                "description": removed.aspect.description,
                "children_keys": removed.aspect.children_keys,
                "root_element_names": removed.aspect.root_element_names,
            },
            elements=removed.elements,
        )
        assert restored.name == "char"


# ===================================================================
# UndoQueue tests
# ===================================================================


class TestPostgresUndoQueue:
    async def test_add_and_pop_undo(self, undo_queue):
        from novelrag.resource_agent.undo import ReversibleAction

        action = ReversibleAction(method="apply", params={"key": "val"})
        await undo_queue.add_undo_item(action)
        popped = await undo_queue.pop_undo_item()
        assert popped is not None
        assert popped.method == "apply"
        assert popped.params == {"key": "val"}

    async def test_pop_empty_undo(self, undo_queue):
        result = await undo_queue.pop_undo_item()
        assert result is None

    async def test_add_and_pop_redo(self, undo_queue):
        from novelrag.resource_agent.undo import ReversibleAction

        action = ReversibleAction(method="apply", params={"x": 1})
        await undo_queue.add_redo_item(action)
        popped = await undo_queue.pop_redo_item()
        assert popped is not None
        assert popped.method == "apply"

    async def test_clear_redo_on_undo_add(self, undo_queue):
        from novelrag.resource_agent.undo import ReversibleAction

        await undo_queue.add_redo_item(ReversibleAction(method="r1", params={}))
        deleted = await undo_queue.add_undo_item(
            ReversibleAction(method="u1", params={}),
            clear_redo=True,
        )
        assert deleted is not None and len(deleted) == 1
        assert await undo_queue.pop_redo_item() is None

    async def test_undo_group(self, undo_queue):
        from novelrag.resource_agent.undo import ReversibleAction

        await undo_queue.add_undo_item(ReversibleAction(method="a", params={}, group="g1"))
        await undo_queue.add_undo_item(ReversibleAction(method="b", params={}, group="g1"))
        group = await undo_queue.pop_undo_group()
        assert group is not None
        assert len(group) == 2
        assert await undo_queue.pop_undo_item() is None

    async def test_redo_group(self, undo_queue):
        from novelrag.resource_agent.undo import ReversibleAction

        await undo_queue.add_redo_item(ReversibleAction(method="a", params={}, group="g2"))
        await undo_queue.add_redo_item(ReversibleAction(method="b", params={}, group="g2"))
        group = await undo_queue.pop_redo_group()
        assert group is not None
        assert len(group) == 2

    async def test_peek_recent(self, undo_queue):
        from novelrag.resource_agent.undo import ReversibleAction

        await undo_queue.add_undo_item(ReversibleAction(method="x1", params={}))
        await undo_queue.add_undo_item(ReversibleAction(method="x2", params={}))
        recent = await undo_queue.peek_recent(5)
        assert len(recent) == 2
        assert recent[0].method == "x2"  # newest first

    async def test_clear(self, undo_queue):
        from novelrag.resource_agent.undo import ReversibleAction

        await undo_queue.add_undo_item(ReversibleAction(method="c1", params={}))
        await undo_queue.add_redo_item(ReversibleAction(method="c2", params={}))
        await undo_queue.clear()
        assert await undo_queue.pop_undo_item() is None
        assert await undo_queue.pop_redo_item() is None


# ===================================================================
# Backlog tests
# ===================================================================


class TestPostgresBacklog:
    async def test_add_and_get_entries(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        entry = BacklogEntry(type="task", priority=20, description="Do something")
        await backlog.add_entry(entry)
        entries = await backlog.get_entries()
        assert len(entries) == 1
        assert entries[0].type == "task"

    async def test_priority_ordering(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        await backlog.add_entry(BacklogEntry(type="low", priority=10, description="low"))
        await backlog.add_entry(BacklogEntry(type="high", priority=30, description="high"))
        await backlog.add_entry(BacklogEntry(type="mid", priority=20, description="mid"))
        entries = await backlog.get_entries()
        priorities = [e.priority for e in entries]
        assert priorities == sorted(priorities, reverse=True)

    async def test_get_top(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        for i in range(5):
            await backlog.add_entry(BacklogEntry(type="t", priority=i * 10, description=f"item {i}"))
        top2 = await backlog.get_top(2)
        assert len(top2) == 2
        assert top2[0].priority >= top2[1].priority

    async def test_pop_entry(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        await backlog.add_entry(BacklogEntry(type="a", priority=10, description="low"))
        await backlog.add_entry(BacklogEntry(type="b", priority=30, description="high"))
        popped = await backlog.pop_entry()
        assert popped is not None
        assert popped.priority == 30
        remaining = await backlog.get_entries()
        assert len(remaining) == 1

    async def test_remove_entries(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        for i in range(4):
            await backlog.add_entry(BacklogEntry(type="t", priority=(i + 1) * 10, description=f"item {i}"))
        removed = await backlog.remove_entries([0, 2])
        assert len(removed) == 2
        remaining = await backlog.get_entries()
        assert len(remaining) == 2

    async def test_clear(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        await backlog.add_entry(BacklogEntry(type="t", priority=10, description="x"))
        await backlog.clear()
        entries = await backlog.get_entries()
        assert len(entries) == 0

    async def test_pop_empty(self, backlog):
        popped = await backlog.pop_entry()
        assert popped is None

    async def test_remove_out_of_range(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        await backlog.add_entry(BacklogEntry(type="t", priority=10, description="only"))
        removed = await backlog.remove_entries([5, 10])
        assert removed == []

    async def test_metadata_roundtrip(self, backlog):
        from novelrag.resource_agent.backlog import BacklogEntry

        entry = BacklogEntry(
            type="review",
            priority=20,
            description="Review chapter",
            metadata={"chapter": 3, "reviewer": "Alice"},
        )
        await backlog.add_entry(entry)
        entries = await backlog.get_entries()
        assert entries[0].metadata == {"chapter": 3, "reviewer": "Alice"}
