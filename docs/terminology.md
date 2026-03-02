# NovelRAG System Terminology

This document defines the core terminology used throughout the NovelRAG system.

---

## Table of Contents

1. [Resource System](#resource-system)
2. [Agent System](#agent-system)

---

## Resource System

### Aspect

A named category of resources (e.g. `character`, `faction`, `narrative_event`). Each aspect owns a collection of root-level resources, is backed by its own YAML file, and carries schema metadata such as `children_keys` and `description` that define the shape of its resource tree.

### Resource

An individual narrative entity within an aspect — a character, a location, an event, etc. Each resource is identified by a **URI** (a hierarchical path like `/<aspect>/<resource_id>[/<nested_id>...]`), carries arbitrary properties, and may contain nested child resources.

In memory, a resource is represented as an **element** — the atomic data unit. `Element` is the core Pydantic data model that carries tree-navigation pointers (parent) and supports in-place splicing of child lists, enabling traversal and mutation of the resource tree.

### Repository

The central manager for the entire resource system. Holds all aspects, an **element look-up table** (URI-keyed dictionary for O(1) element retrieval), and a **vector store** (LanceDB-backed embedding store for semantic search). Provides resource lookup by URI, semantic vector search, and atomic operation application. The concrete implementation is backed by YAML files and a LanceDB vector store.

### Operation

An atomic, undoable mutation to the resource system. Two variants exist:

- **Property Operation** — updates properties on an existing element by URI.
- **Resource Operation** — splices (inserts, removes, or replaces) elements in a resource list at a given location.

Both produce an inverse operation on application, enabling undo.

---

## Agent System

### Agent

The goal-pursuing entity. Three variants exist:

- **Goal Executor** — the core agent loop. Given a goal, it repeatedly **determines the next action** — either an operation plan (execute a tool) or a resolution (terminate the pursuit) — based on current beliefs, pursuit progress, and available tools. It executes the chosen tool, records the outcome, and loops until a resolution is reached.
- **Request Handler** — handles user request strings end-to-end. First **translates** the natural-language request into a goal (via an LLM-based translator), then delegates to a goal executor to pursue it.
- **Autonomous Agent** — autonomously **decides** the next goal to pursue (or determines none is available), then delegates to a goal executor. Used for hands-free exploration and backlog processing.

### Goal

A clear objective for the agent, carrying a description and a source tracing its origin — either a user request or an autonomous decision.

The **goal decision** subsystem is used by the autonomous agent. A composite decider selects among multiple sources via weighted random, dynamically adjusting weights based on data availability:

- **Backlog Goal Decider** — formulates goals from the highest-priority backlog entries via LLM, consuming used entries.
- **Exploration Goal Decider** — generates goals by exploring the repository in three tiers: *bootstrap* (no aspects — propose first aspects from beliefs), *populate* (empty aspects — populate one), *explore* (random-walk, context expansion, concept-gap analysis).
- **Recency Weighter** — derives per-aspect and per-element weight penalties from recent undo history to promote exploration diversity.

### Pursuit

The lifecycle of executing a goal.

- **Pursuit Progress** — tracks the in-flight state: the goal itself and the list of operation outcomes executed so far.
- **Pursuit Outcome** — an immutable record of a completed pursuit: status (completed, failed, or abandoned), all executed steps, the final resolution, and a user-facing response.
- **Pursuit Assessment** — a structured LLM assessment of current progress: finished tasks, remaining work, required context, expected actions, boundary conditions, exception conditions, and success criteria. Produced during action determination to guide the next decision.

### Step

A single unit of work within a pursuit.

- **Operation Plan** — an immutable directive to execute a specific tool, containing a reason, tool name, and parameters.
- **Resolution** — an immutable directive to terminate the pursuit with a final status, a reason, and a user-facing response.
- **Operation Outcome** — the result of executing an operation plan: success or failure status, result text, error message, and timing information.

### Tool

An executable unit that performs a specific action. Each tool has a name, description, a JSON Schema for its input parameters, and an async execution method. The agent can only invoke tools registered in its tool set. Results are either a successful `ToolResult` or a `ToolError`.

The concrete **resource tools** available to the agent:

- **Resource Fetch Tool** — read-only; fetches a resource, aspect, or repository root by URI.
- **Resource Search Tool** — read-only; performs semantic vector search across all resources.
- **Aspect Create Tool** — creates new aspects with LLM-generated metadata. Supports undo.
- **Resource Write Tool** — the main editing tool. Orchestrates a four-phase workflow: content generation → operation building & application → cascade updates → backlog discovery. Supports undo.
- **Resource Relation Write Tool** — manages bidirectional relationships between resources. Supports undo.

### Action Determination

The multi-phase orchestration system that **determines the next action** during goal pursuit. Composes two sub-loops:

- **Context Discovery Loop** — iteratively discovers and refines the relevant context. Each iteration has a *discovery* phase (proposes search queries, resource URIs to load, and tools to expand) and a *refinement* phase (filters out irrelevant resources/properties, collapses verbose tool schemas, reorders segments by priority). Repeats until context is sufficient.
- **Action Loop** — makes the action decision and validates it. Each iteration has a *decision* phase (analyzes the situation, decides to execute a tool or finalize) and a *refinement analysis* phase (approves the decision or requests refinement with an updated pursuit assessment).

The **resource context** maintains the evolving workspace state during action determination. Wraps the repository to provide resource querying, semantic search, inclusion/exclusion of resources and properties, and immutable context snapshots for LLM prompt rendering. Internally tracks each loaded resource as a **segment** — a partially loaded resource identified by URI with a set of excluded properties.

### Content Pipeline

Reusable multi-step procedures for content generation and consistency maintenance:

- **Content Generation** — generates content through a competitive proposal process. Multiple content proposers independently produce proposals using Sequential Diverse Prompting (generate diverse creative perspectives, then produce content from sampled perspectives). Proposals are ranked via LLM and selected by weighted random.
- **Cascade Update** — after a primary write operation, discovers and applies follow-on updates — both perspective/content changes and relationship changes on related resources — to maintain cross-resource consistency.
- **Backlog Discovery** — analyzes completed operations and discovers follow-up work items to add to the backlog for future autonomous processing.

### Backlog

A priority-sorted work queue for tracking follow-up work items. Each entry has a type, numeric priority (`high=30`, `normal=20`, `low=10`), description, and extensible metadata. Provides in-memory and file-persisted implementations.

### Undo System

Tracks reversible actions for undo/redo. Each reversible action records a method name, parameters, and optional group tag for batched undo. The undo queue provides in-memory and file-persisted implementations with bounded stack size.

### Execution Context

The unified runtime interface shared by procedures, tools, and the agent loop. Provides three facets: messaging (info, debug, warning, error), user-facing output, and bidirectional prompts (confirm, request). Implementations route these to CLI, logger, or test harnesses.

A **procedure error** is an exception that preserves partial side effects already applied when a procedure fails midway, enabling informed rollback decisions.
