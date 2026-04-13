from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, JSON, ForeignKey, Index, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()

    workspaces: Mapped[list["Workspace"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Workspace(Base):
    __tablename__ = "workspaces"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    language: Mapped[str] = mapped_column(default="zh")
    # A single belief will not be queried separately, so we can store them as a list in the workspace table for simplicity.
    beliefs: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    user: Mapped["User"] = relationship(back_populates="workspaces")
    aspects: Mapped[list["ResourceAspect"]] = relationship(back_populates="workspace", cascade="all, delete-orphan")
    backlog_entries: Mapped[list["BacklogEntry"]] = relationship(
        back_populates="workspace", cascade="all, delete-orphan"
    )
    undo_redo_table: Mapped["UndoRedoTable"] = relationship(
        back_populates="workspace", uselist=False, cascade="all, delete-orphan"
    )
    undo_items: Mapped[list["UndoItem"]] = relationship(back_populates="workspace", cascade="all, delete-orphan")
    redo_items: Mapped[list["RedoItem"]] = relationship(back_populates="workspace", cascade="all, delete-orphan")


class ResourceAspect(Base):
    __tablename__ = "resource_aspects"
    __table_args__ = (UniqueConstraint("workspace_id", "uri", name="uq_resource_aspects_workspace_uri"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column()
    uri: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column(default="")
    children_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    # By default, SQLAlchemy does not detect in-place mutations to JSON.
    # We should submit the update manually.
    aspect_meta: Mapped[dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)
    root_element_names: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)

    workspace: Mapped["Workspace"] = relationship(back_populates="aspects")


class ResourceElement(Base):
    __tablename__ = "resource_elements"
    __table_args__ = (
        UniqueConstraint("workspace_id", "aspect_id", "uri", name="uq_resource_elements_aspect_uri"),
        Index(
            "ix_resource_elements_embedding",
            "embedding",
            # Considering the heavy insert/update operations and smaller scale of data, we choose HNSW for better overall performance.
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        {"postgresql_partition_by": "LIST (workspace_id)"},
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column()
    # The element aspect relationship is maintained with a foreign key without CASCADE delete
    # because we want to return the deleted elements for undo purposes before deleting them.
    aspect_id: Mapped[int] = mapped_column(ForeignKey("resource_aspects.id"))
    name: Mapped[str] = mapped_column()
    uri: Mapped[str] = mapped_column()
    relationships: Mapped[dict[str, list[str]]] = mapped_column(JSON, default=dict)
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    embedding: Mapped[list[float]] = mapped_column(Vector(3072))


class BacklogEntry(Base):
    __tablename__ = "backlog_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    type: Mapped[str] = mapped_column()
    priority: Mapped[int] = mapped_column()
    description: Mapped[str] = mapped_column()
    backlog_meta: Mapped[dict[str, Any]] = mapped_column(JSON, name="metadata", default=dict)

    workspace: Mapped["Workspace"] = relationship(back_populates="backlog_entries")


class UndoRedoTable(Base):
    __tablename__ = "undo_redo_table"

    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), primary_key=True)
    undo_stack: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    last_undo_group: Mapped[str | None] = mapped_column(name="last_undo_group", default=None)
    redo_stack: Mapped[list[int]] = mapped_column(ARRAY(Integer), default=list)
    curr_redo_group: Mapped[str | None] = mapped_column(name="curr_redo_group", default=None)


class UndoItem(Base):
    __tablename__ = "undo_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    method: Mapped[str] = mapped_column()
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    undo_group: Mapped[str | None] = mapped_column(name="group", default=None)

    workspace: Mapped["Workspace"] = relationship(back_populates="undo_items")


class RedoItem(Base):
    __tablename__ = "redo_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey("workspaces.id", ondelete="CASCADE"), index=True)
    method: Mapped[str] = mapped_column()
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    redo_group: Mapped[str | None] = mapped_column(name="group", default=None)

    workspace: Mapped["Workspace"] = relationship(back_populates="redo_items")
