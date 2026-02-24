from typing import Any

from sqlalchemy import ARRAY, JSON, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()

    workspaces: Mapped[list['Workspace']] = relationship(back_populates='user', cascade='all, delete-orphan')


class Workspace(Base):
    __tablename__ = 'workspaces'

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()
    user_id: Mapped[int] = mapped_column(ForeignKey('users.id', ondelete='CASCADE'), index=True)
    language: Mapped[str] = mapped_column(default='zh')
    # A single belief will not be queried separately, so we can store them as a list in the workspace table for simplicity.
    beliefs: Mapped[list[str]] = mapped_column(ARRAY(String), default_factory=list)

    user: Mapped['User'] = relationship(back_populates='workspaces')
    aspects: Mapped[list['ResourceAspect']] = relationship(back_populates='workspace', cascade='all, delete-orphan')
    backlog_entries: Mapped[list['BacklogEntry']] = relationship(back_populates='workspace', cascade='all, delete-orphan')
    undo_items: Mapped[list['UndoItem']] = relationship(back_populates='workspace', cascade='all, delete-orphan')
    redo_items: Mapped[list['RedoItem']] = relationship(back_populates='workspace', cascade='all, delete-orphan')


class ResourceAspect(Base):
    __tablename__ = 'resource_aspects'
    __table_args__ = (
        UniqueConstraint('workspace_id', 'uri', name='uq_resource_aspects_workspace_uri'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey('workspaces.id', ondelete='CASCADE'))
    name: Mapped[str] = mapped_column()
    uri: Mapped[str] = mapped_column()
    description: Mapped[str] = mapped_column(default='')
    children_keys: Mapped[list[str]] = mapped_column(ARRAY(String), default_factory=list)
    # By default, SQLAlchemy does not detect in-place mutations to JSON.
    # We should submit the update manually.
    aspect_meta: Mapped[dict[str, Any]] = mapped_column(JSON, name='metadata', default_factory=dict)

    workspace: Mapped['Workspace'] = relationship(back_populates='aspects')
    elements: Mapped[list['ResourceElement']] = relationship(back_populates='aspect', cascade='all, delete-orphan')


class ResourceElement(Base):
    __tablename__ = 'resource_elements'
    __table_args__ = (
        UniqueConstraint('aspect_id', 'uri', name='uq_resource_elements_aspect_uri'),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    aspect_id: Mapped[int] = mapped_column(ForeignKey('resource_aspects.id', ondelete='CASCADE'))
    name: Mapped[str] = mapped_column()
    uri: Mapped[str] = mapped_column()
    data: Mapped[dict[str, Any]] = mapped_column(JSON, default_factory=dict)

    aspect: Mapped['ResourceAspect'] = relationship(back_populates='elements')


class BacklogEntry(Base):
    __tablename__ = 'backlog_entries'

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey('workspaces.id', ondelete='CASCADE'), index=True)
    type: Mapped[str] = mapped_column()
    priority: Mapped[int] = mapped_column()
    description: Mapped[str] = mapped_column()
    backlog_meta: Mapped[dict[str, Any]] = mapped_column(JSON, name='metadata', default_factory=dict)

    workspace: Mapped['Workspace'] = relationship(back_populates='backlog_entries')


class UndoItem(Base):
    __tablename__ = 'undo_items'

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey('workspaces.id', ondelete='CASCADE'), index=True)
    method: Mapped[str] = mapped_column()
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default_factory=dict)
    undo_group: Mapped[str | None] = mapped_column(name='group', default=None)

    workspace: Mapped['Workspace'] = relationship(back_populates='undo_items')


class RedoItem(Base):
    __tablename__ = 'redo_items'

    id: Mapped[int] = mapped_column(primary_key=True)
    workspace_id: Mapped[int] = mapped_column(ForeignKey('workspaces.id', ondelete='CASCADE'), index=True)
    method: Mapped[str] = mapped_column()
    params: Mapped[dict[str, Any]] = mapped_column(JSON, default_factory=dict)
    redo_group: Mapped[str | None] = mapped_column(name='group', default=None)

    workspace: Mapped['Workspace'] = relationship(back_populates='redo_items')
