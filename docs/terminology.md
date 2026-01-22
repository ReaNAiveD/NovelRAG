# NovelRAG System Terminology

This document defines the core terminology used throughout the NovelRAG system.

---

## Table of Contents

1. [Resource System Terminology](#resource-system-terminology)
2. [Agent System Terminology](#agent-system-terminology)

---

## Resource System Terminology

### Resource System
The hierarchical data management system for organizing and managing narrative content. It provides a unified interface for storing, querying, and manipulating story-related data.

### Aspect
A **category or type** of resources in the system. Aspects define the classification of narrative entities.

**Examples:**
- `character` - Character entities
- `location` - Location entities  
- `scene` - Scene entities

**Technical Details:**
- Defined in configuration as `AspectConfig`
- Stored in `ResourceAspect` class at runtime
- Each aspect has its own root-level resource list

### Resource
An **individual instance** within an aspect. Resources are the conceptual entities that represent actual narrative elements.

**Examples:**
- `/character/sarah_chen_detective`
- `/location/europe/london`

**Characteristics:**
- Identified by a **URI** (Unique Resource Identifier)
- Has properties (attributes/fields)
- May have children (nested resources)
- Represented by an `Element` data structure

### Element
The **atomic data unit** - the actual data structure that represents a resource in memory.

**Key Classes:**
- `Element` (Pydantic model) - Core data structure
- `DirectiveElement` - Wrapper providing tree structure and manipulation
- `ElementLookUpTable` - Index for fast element retrieval by URI

### URI (Uniform Resource Identifier)
A **hierarchical path** that uniquely identifies a resource in the system.

**Format:** `/<aspect>[/<resource_id>[/<nested_id>...]]`

**Examples:**
- `/` - Root (all aspects)
- `/character` - Character aspect
- `/character/john_doe` - Specific character

### Repository
The `ResourceRepository` class manages the entire resource system.

**Key Components:**
- `resource_aspects: dict[str, ResourceAspect]` - All aspects
- `lut: ElementLookUpTable` - Fast lookup by URI
- `vector_store: LanceDBStore` - Vector embeddings for semantic search

**Key Operations:**
- `find_by_uri(resource_uri)` - Find resource by URI
- `vector_search(query)` - Semantic search across resources
- `apply(operation)` - Modify resources via operations

### Operation
An atomic change to the resource system with undo capability.

**Operation Types:**
- `PropertyOperation` - Updates properties of an existing resource
- `ResourceOperation` - Adds, removes, or replaces resources in lists

---

## Agent System Terminology

### GoalExecutor
The **main controller** for goal pursuit. It coordinates the execution of goals using tools and an action determiner.

**Location**: `novelrag/agenturn/agent.py`

**Key Methods:**
- `handle_goal(goal)` - Execute a goal and return outcome
- `_execute_tool(tool_name, params, reason)` - Execute a single tool
- `create_request_handler(goal_translator)` - Create a RequestHandler
- `create_autonomous_agent(goal_decider)` - Create an AutonomousAgent

### Goal
A **refined statement of intent** describing what the agent aims to achieve.

**Structure:**
```python
@dataclass
class Goal:
    description: str
    source: GoalSource  # UserRequestSource or AutonomousSource
```

### GoalTranslator
Protocol for translating user requests into goals.

**Implementation**: `LLMGoalTranslator` uses LLM to translate requests.

### ActionDeterminer
Protocol for determining the next action during goal pursuit.

**Implementation**: `ActionDetermineLoop` in `novelrag/resource_agent/action_determine_loop.py`

**Method:**
```python
async def determine_action(
    beliefs: list[str],
    pursuit_progress: PursuitProgress,
    available_tools: dict[str, SchematicTool]
) -> OperationPlan | Resolution
```

### PursuitProgress
Tracks the **progress** of a goal pursuit.

**Structure:**
```python
@dataclass
class PursuitProgress:
    goal: Goal
    pending_steps: list[str]
    executed_steps: list[OperationOutcome]
```

### PursuitOutcome
The **final outcome** of a goal pursuit.

**Structure:**
```python
@dataclass
class PursuitOutcome:
    goal: Goal
    reason: str
    response: str
    status: PursuitStatus  # COMPLETED, FAILED, ABANDONED
    executed_steps: list[OperationOutcome]
    resolution: Resolution
    resolve_at: datetime
```

### PursuitAssessment
Assessment of current pursuit progress toward a goal.

**Structure:**
```python
@dataclass
class PursuitAssessment:
    finished_tasks: list[str]
    remaining_work_summary: str
    required_context: str
    expected_actions: str
    boundary_conditions: list[str]
    exception_conditions: list[str]
    success_criteria: list[str]
```

### OperationPlan
A directive to **execute a specific tool** and continue the pursuit.

**Structure:**
```python
@dataclass(frozen=True)
class OperationPlan:
    reason: str
    tool: str
    parameters: dict
```

### Resolution
A directive to **terminate the pursuit** with a final status.

**Structure:**
```python
@dataclass(frozen=True)
class Resolution:
    reason: str
    response: str
    status: str  # success, failed, abandoned
```

### OperationOutcome
The **result of executing** an OperationPlan.

**Structure:**
```python
@dataclass
class OperationOutcome:
    operation: OperationPlan
    status: StepStatus  # SUCCESS, FAILED, CANCELLED
    results: list[str]
    error_message: str | None
    started_at: datetime | None
    completed_at: datetime | None
```

### Tool
An **executable unit** that performs a specific action.

**Types:**
- `BaseTool` - Abstract base interface
- `SchematicTool` - Tools with JSON schema for parameters

**Interface:**
```python
class SchematicTool:
    name: str
    description: str
    input_schema: dict
    
    async def call(runtime: ToolRuntime, **params) -> ToolOutput
```

### ToolRuntime
The **interface provided to tools** during execution.

**Methods:**
- `debug(content)`, `message(content)` - Output messages
- `warning(content)`, `error(content)` - Error messages
- `confirmation(prompt)` - Ask yes/no question
- `user_input(prompt)` - Request input from user
- `progress(key, value, description)` - Track progress
- `backlog(content, priority)` - Add to backlog

### ToolOutput
The **result returned by a tool** after execution.

**Types:**
- `ToolResult` - Successful execution with result string
- `ToolError` - Failed execution with error message

### AgentChannel
The **communication interface** between agent and user.

**Methods:**
- `send_message(content, level)` - Send message at specified level
- `confirm(prompt)` - Boolean confirmation
- `request(prompt)` - String input

**Implementations:**
- `SessionChannel` - Session-based communication in CLI

### RequestHandler
Handles incoming requests by translating them to goals and executing.

**Usage:**
```python
handler = executor.create_request_handler(goal_translator)
response = await handler.handle_request("Find the protagonist")
```

### AutonomousAgent
Agent that autonomously generates and pursues goals.

**Usage:**
```python
agent = executor.create_autonomous_agent(goal_decider)
outcome = await agent.pursue_next_goal()
```

---

## ActionDetermineLoop Terminology

### ActionDetermineLoop
The **resource-aware action determiner** using multi-phase orchestration.

**Location**: `novelrag/resource_agent/action_determine_loop.py`

**Phases:**
1. Context Discovery - Find relevant context
2. Context Refinement - Filter and prioritize context
3. Action Decision - Decide to execute tool or finalize
4. Refinement Analysis - Validate decision or refine assessment

### DiscoveryPlan
Result from context discovery phase.

**Fields:**
- `discovery_analysis` - Analysis text
- `search_queries` - Terms to search
- `query_resources` - URIs to load
- `expand_tools` - Tools to expand

### RefinementPlan
Result from context refinement phase.

**Fields:**
- `exclude_resources` - URIs to exclude
- `exclude_properties` - Properties to exclude
- `collapse_tools` - Tools to collapse
- `sorted_segments` - Priority ordering

### ActionDecision
Result from action decision phase.

**Fields:**
- `situation_analysis` - Current state assessment
- `decision_type` - "execute" or "finalize"
- `execution` - Tool and params if executing
- `finalization` - Response if finalizing
- `context_verification` - Verification details

### RefinementDecision
Result from refinement analysis phase.

**Fields:**
- `analysis` - Quality assessment
- `verdict` - "approve" or "refine"
- `approval` - Details if approving
- `refinement` - Updated assessment if refining

### ResourceContext
The **context builder** that manages workspace state during orchestration.

**Location**: `novelrag/resource_agent/workspace.py`

**Key Operations:**
- `search_resources(query)` - Semantic search
- `query_resource(uri)` - Load specific resource
- `exclude_resource(uri)` - Remove from context

---

## Decision Matrix

| Scenario | Use Term | Example |
|----------|----------|---------|
| Main execution controller | **GoalExecutor** | `executor.handle_goal(goal)` |
| User request to goal | **GoalTranslator** | `translator.translate(request)` |
| Action determination | **ActionDeterminer** | `determiner.determine_action()` |
| Tool execution directive | **OperationPlan** | Execute a specific tool |
| Pursuit termination | **Resolution** | End with success/failure |
| Execution result | **OperationOutcome** | Tracks status and results |
| Tool interface | **SchematicTool** | Has name, schema, call() |
| Tool result | **ToolOutput** | ToolResult or ToolError |
| User communication | **AgentChannel** | send_message, confirm, request |
| Request handling | **RequestHandler** | handle_request() |
| Autonomous operation | **AutonomousAgent** | pursue_next_goal() |

---

## Package Structure

```
novelrag/
+-- agenturn/                    # Generic agent framework
|   +-- agent.py                 # GoalExecutor, RequestHandler, AutonomousAgent
|   +-- channel.py               # AgentChannel protocol
|   +-- goal.py                  # Goal, GoalTranslator
|   +-- pursuit.py               # PursuitProgress, PursuitOutcome, ActionDeterminer
|   +-- step.py                  # OperationPlan, OperationOutcome, Resolution
|   +-- tool/                    # Tool abstractions
|
+-- resource_agent/              # Resource-specific implementation
|   +-- action_determine_loop.py # ActionDetermineLoop
|   +-- workspace.py             # ResourceContext
|   +-- tool/                    # Resource tools
|
+-- resource/                    # Resource system
    +-- repository.py            # ResourceRepository
    +-- aspect.py                # ResourceAspect
    +-- element.py               # Element, DirectiveElement
```
