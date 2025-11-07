# NovelRAG System Terminology

This document defines the core terminology used throughout the NovelRAG system to ensure consistent understanding and usage across both the Resource System and Agent System.

---

## Table of Contents

1. [Resource System Terminology](#resource-system-terminology)
2. [Agent System Terminology](#agent-system-terminology)

---

## Resource System Terminology

Core concepts related to the hierarchical data management system for organizing and managing narrative content.

### Resource System
The entire hierarchical data management system for organizing and managing narrative content. It provides a unified interface for storing, querying, and manipulating story-related data.

### Aspect
A **category or type** of resources in the system. Aspects define the classification of narrative entities.

**Examples:**
- `character` - Character entities
- `location` - Location entities  
- `scene` - Scene entities
- `outline` - Outline/chapter entities
- `culture` - Cultural concept entities
- `law` - Legal/rule system entities
- `theme` - Thematic elements

**Technical Details:**
- Defined in configuration as `AspectConfig`
- Contains metadata: description, file path, children keys
- Stored in `ResourceAspect` class at runtime
- Each aspect has its own root-level resource list

### Resource
An **individual instance** within an aspect. Resources are the conceptual entities that represent actual narrative elements, whether concrete (characters, locations) or abstract (themes, cultures, laws).

**Examples:**
- Concrete: `/character/sarah_chen_detective`, `/location/london`
- Abstract: `/culture/victorian_era`, `/law/maritime_code`, `/theme/redemption`
- Nested: `/location/europe/london/baker_street`, `/culture/victorian_era/social_norms`

**Characteristics:**
- Identified by a **URI** (Unique Resource Identifier)
- Has properties (attributes/fields)
- May have children (nested resources)
- May have relations to other resources
- Represented by an `Element` data structure

### Element
The **atomic data unit** - the actual data structure that represents a resource in memory. An element is the technical/implementation concept, while a resource is the conceptual/semantic entity.

**Key Classes:**
- `Element` (Pydantic model) - Core data structure with id, uri, relations, properties
- `DirectiveElement` - Wrapper providing tree structure and manipulation capabilities
- `ElementLookUpTable` - Index for fast element retrieval by URI

**Element Structure:**
```python
class Element(BaseModel):
    id: str              # Identifier within parent
    uri: str             # Full resource URI
    relations: dict      # Relations to other resources
    aspect: str          # Aspect this element belongs to
    children_keys: list  # Keys for nested children
    # ... additional properties in model_extra
```

**Relationship:**
```
Resource URI ──(identifies)──> Resource ──(represented by)──> Element
   ↓                              ↓                             ↓
"/character/john_doe"      John Doe character          Element data structure
```

---

## URI (Uniform Resource Identifier)

A **hierarchical path** that uniquely identifies a resource in the system.

**Format:** `/<aspect>[/<resource_id>[/<nested_id>...]]`

**Examples:**
- `/` - Root (all aspects)
- `/character` - Character aspect
- `/character/john_doe` - Specific character
- `/location/europe/london` - Nested location
- `/outline/chapter_1/scene_opening` - Deeply nested resource

**Semantics:**
- URIs identify **resources** (conceptual entities)
- URIs do NOT identify "elements" (data structures)
- Used as keys in lookups, operations, and references
- Each resource has exactly one URI (unique identifier)

---

## Data Organization

### Resource Hierarchy

```
Root (/)
├── Aspect: character
│   ├── Resource: /character/john_doe
│   │   ├── Properties: {name, age, description, ...}
│   │   └── Relations: {friend_of: [...], knows: [...]}
│   └── Resource: /character/sarah_chen
├── Aspect: location
│   └── Resource: /location/london
│       └── Children: {districts: [/location/london/westminster, ...]}
└── Aspect: outline
    └── Resource: /outline/chapter_1
        └── Children: {scenes: [...]}
```

### Repository
The `ResourceRepository` class manages the entire resource system.

**Key Components:**
- `resource_aspects: dict[str, ResourceAspect]` - All aspects
- `lut: ElementLookUpTable` - Fast lookup by URI
- `vector_store: LanceDBStore` - Vector embeddings for semantic search
- `embedding_llm: EmbeddingLLM` - Generates embeddings

**Key Operations:**
- `find_by_uri(resource_uri)` - Find resource by URI
- `vector_search(query)` - Semantic search across resources
- `apply(operation)` - Modify resources via operations
- `update_relations(resource_uri, target_uri, relations)` - Update relations

---

## Operations System

### Operation
An atomic change to the resource system. Operations provide undo capability and form the basis of the modification system.

**Operation Types:**

#### PropertyOperation
Updates properties of an existing resource.

```python
PropertyOperation(
    target="property",
    resource_uri="/character/john_doe",  # Which resource to update
    data={                                # Property updates
        "age": 35,
        "occupation": "detective"
    }
)
```

#### ResourceOperation
Adds, removes, or replaces resources in lists (splice operation).

```python
ResourceOperation(
    target="resource",
    location=ResourceLocation(          # Where to operate
        type="resource",
        resource_uri="/outline/chapter_1",
        children_key="scenes"
    ),
    start=2,                            # Start index (inclusive)
    end=3,                              # End index (exclusive)
    data=[...]                          # New resources to insert
)
```

### Operation Location

Specifies where a resource operation occurs.

**ResourceLocation:**
- Points to a children list within a specific resource
- Fields: `resource_uri`, `children_key`
- Example: `/outline/chapter_1` with `children_key="scenes"`

**AspectLocation:**
- Points to the root level of an aspect
- Field: `aspect`
- Example: aspect `"character"` (operates on root character list)

---

## Vector Storage

### Vector Store
Manages embeddings for semantic search over resources.

**Implementation:** `LanceDBStore` using LanceDB

**Schema:**
```python
class EmbeddingSearch(LanceModel):
    vector: Vector(3072)     # Embedding vector
    hash: str                # Content hash (for change detection)
    resource_uri: str        # URI of the resource
    aspect: str              # Aspect for filtering
```

**Key Operations:**
- `vector_search(vector, aspect, limit)` - Find similar resources
- `add(element)` - Add resource embedding
- `update(element)` - Update resource embedding
- `delete(resource_uri)` - Remove resource embedding
- `get(resource_uri)` - Retrieve embedding by URI
- `cleanup_invalid_resources(valid_resource_uris)` - Remove stale embeddings

**Usage Pattern:**
```python
# Search returns URIs
results = await vector_store.vector_search(query_vector)

# Use URIs to lookup elements
for result in results:
    element = lut.find_by_uri(result.resource_uri)
```

---

## Terminology Decision Matrix

Use this table to choose the correct term:

| Scenario | Use Term | Example |
|----------|----------|---------|
| Referring to a conceptual entity | **Resource** | "Add a new character resource" |
| Referring to a category/type | **Aspect** | "The character aspect" |
| Identifying an entity | **Resource URI** | `"/character/john_doe"` |
| Referring to the data structure | **Element** | `element.props()`, `DirectiveElement` |
| Looking up by identifier | **resource_uri** parameter | `find_by_uri(resource_uri)` |
| Returning data structure | **Element** return type | Returns `DirectiveElement` |
| Operation parameter (what to modify) | **resource_uri** | `PropertyOperation(resource_uri=...)` |
| User-facing documentation | **Resource** | "Update resource properties" |
| Technical/implementation docs | **Element** | "Element data structure" |
| Index/lookup table context | **ElementLookUpTable** | Lookups element structures by URI |
| Database storage context | **resource_uri** field | Store URI in database |

---

## Naming Conventions

### Class Names
- `Element` - Data structure representing a resource
- `DirectiveElement` - Tree-structured element wrapper
- `ResourceAspect` - Aspect container
- `ResourceRepository` - Main repository
- `ResourceOperation` - Operation on resource lists
- `ResourceLocation` - Location pointing to a resource
- `ElementLookUpTable` - Lookup table for elements (correct: looks up element structures)

### Variable Names
```python
# Good - Clear semantic distinction
resource_uri: str = "/character/john_doe"
element: DirectiveElement = lut.find_by_uri(resource_uri)
aspect: ResourceAspect = repository.resource_aspects["character"]

# Good - Clear context
async def find_by_uri(self, resource_uri: str) -> DirectiveElement | None:
    """Find element representing the resource at the given URI."""
    return self.lut.find_by_uri(resource_uri)

# Good - Operation parameters
PropertyOperation(
    resource_uri="/character/john_doe",  # What resource
    data={...}                            # What changes
)
```

### Method Names
- `find_by_uri(resource_uri)` - Find using resource URI
- `apply(operation)` - Apply operation
- `update_relations(resource_uri, target_uri, relations)` - Update resource relations
- `vector_search(query)` - Search resources
- `cleanup_invalid_resources(valid_resource_uris)` - Clean up resources

---

## Conceptual Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Resource System                          │
│                                                             │
│  ┌──────────────┐      ┌──────────────┐                   │
│  │   Aspect     │      │   Aspect     │                   │
│  │  "character" │      │  "location"  │                   │
│  └──────┬───────┘      └──────┬───────┘                   │
│         │                      │                            │
│         │ contains             │ contains                   │
│         ↓                      ↓                            │
│  ┌─────────────┐       ┌─────────────┐                    │
│  │  Resource   │       │  Resource   │                    │
│  │ (Conceptual)│       │ (Conceptual)│                    │
│  └──────┬──────┘       └──────┬──────┘                    │
│         │                      │                            │
│         │ identified by        │ identified by              │
│         ↓                      ↓                            │
│     ┌────────┐            ┌────────┐                       │
│     │  URI   │            │  URI   │                       │
│     └───┬────┘            └───┬────┘                       │
│         │                     │                             │
│         │ represented by      │ represented by              │
│         ↓                     ↓                             │
│  ┌─────────────┐       ┌─────────────┐                    │
│  │   Element   │       │   Element   │                    │
│  │ (Data Struct)│      │ (Data Struct)│                   │
│  └─────────────┘       └─────────────┘                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Lookup Flow:
  resource_uri → ElementLookUpTable → Element (data structure)
  
Operation Flow:
  Operation(resource_uri) → Repository → Element → Modify → Save
  
Search Flow:
  Query → Vector → resource_uri → ElementLookUpTable → Element
```

---

## Common Patterns

### Finding a Resource
```python
# By URI (returns element representing the resource)
element = await repository.find_by_uri(resource_uri="/character/john_doe")

# By semantic search (returns elements of matching resources)
results = await repository.vector_search(query="detective character")
for result in results:
    element = result.element  # Element structure
    uri = element.uri         # Resource URI
```

### Modifying a Resource
```python
# Update properties
await repository.apply(PropertyOperation(
    resource_uri="/character/john_doe",
    data={"age": 35}
))

# Add child resources
await repository.apply(ResourceOperation(
    location=ResourceLocation(
        resource_uri="/outline/chapter_1",
        children_key="scenes"
    ),
    start=0, end=0,
    data=[{"id": "opening_scene", ...}]
))
```

### Working with Elements
```python
# Get element properties
element = await repository.find_by_uri(resource_uri)
properties = element.props()          # Dict of properties
children = element.children_of("scenes")  # List of child elements

# Navigate hierarchy
for child in element.children_of("relationships"):
    print(f"Child URI: {child.uri}")  # Each child has its own URI
```

---

## Migration Notes

For developers working with legacy code that uses `element_uri`:

### What Changed
- `element_uri` → `resource_uri` (parameter names, fields)
- `ElementLocation` → `ResourceLocation` (class name)
- `ElementOperation` → `ResourceOperation` (class name)
- `OperationTarget.ELEMENT` → `OperationTarget.RESOURCE` (enum value)
- `cleanup_invalid_elements()` → `cleanup_invalid_resources()` (method name)

### What Did NOT Change
- `Element` class (data structure)
- `ElementLookUpTable` class (lookup table for element structures)
- `DirectiveElement` class (element wrapper)
- Methods returning `Element` or `DirectiveElement`

### Rationale
URIs identify resources (conceptual entities), not elements (data structures). The refactoring clarifies that:
- **Parameters with URIs** should be named `resource_uri` (they identify resources)
- **Return types** can be `Element` or `DirectiveElement` (they are data structures)
- **Lookup operations** use `resource_uri` to find the `Element` representing that resource

---

## Summary

**Key Takeaway:** Resources are conceptual entities identified by URIs. Elements are data structures that represent resources. URIs are resource identifiers, not element identifiers.

```
Resource (Concept) ←─[identified by]─← URI ─[used to lookup]→ Element (Data Structure)
```

This distinction enables clear communication and consistent implementation throughout the NovelRAG resource system.

---

## Agent System Terminology

Core concepts related to the multi-phase orchestration system for goal pursuit and tool execution.

### Agent
The main controller for **goal pursuit** and **user interaction**. The Agent receives user requests, coordinates the orchestration process, executes tools, and returns final responses.

**Key Responsibilities:**
- Convert user requests into structured goals
- Create and manage OrchestrationLoop instances
- Execute tools based on orchestration decisions
- Track execution history (completed steps, pending steps)
- Handle errors and communicate with users

**Main Method:** `handle_request(request: str) -> str`

### OrchestrationLoop
The **multi-phase strategic decision engine** that determines what action to take next. Uses a sophisticated four-phase architecture to discover context, refine it, make decisions, and validate those decisions.

**Four Phases:**
1. **Context Discovery** - Identify and load relevant resources and tools
2. **Context Refinement** - Filter and prioritize discovered context
3. **Action Decision** - Choose to execute a tool or finalize
4. **Refinement Analysis** - Validate decision or refine goal

**Returns:**
- `OrchestrationExecutionPlan` - Execute a specific tool
- `OrchestrationFinalization` - Complete goal pursuit with response

**Main Method:** `execution_advance(...) -> OrchestrationExecutionPlan | OrchestrationFinalization`

### Goal
A **refined statement of intent** that evolves across iterations. Goals accumulate discovered prerequisites and context requirements through the refinement process.

**Evolution Example:**
```
Iteration 1: "Create protagonist named 余归"
Iteration 2: "Create protagonist named 余归 (Prerequisites: Check if character aspect exists)"
Iteration 3: "Create protagonist named 余归 (Prerequisites: 1. Verify character aspect, 2. Check for existing character...)"
```

**Built by:** `GoalBuilder` from user request

### Tool
An **executable unit** that performs a specific action. Tools are atomic, composable functions with well-defined schemas.

**Types:**
- `BaseTool` - Abstract base interface
- `SchematicTool` - Tools with JSON schema for parameters
- Tool has `name`, `description`, `input_schema`, `prerequisites`, `output_description`

**Execution:** `await tool.call(runtime, **params) -> ToolOutput`

### Tool State
Tools can be in two states within orchestration:

**Collapsed State:**
- Only name and description visible to LLM
- Minimal context consumption
- Default state for all tools

**Expanded State:**
- Full schema with parameters, types, descriptions visible
- Higher context consumption but necessary for execution planning
- Dynamically expanded/collapsed by orchestration phases

**Managed by:** `OrchestrationLoop.expanded_tools` set

### Phase
A distinct stage in the orchestration process, each with specific responsibility and dedicated LLM template.

**Phase 1: Context Discovery**
- **Template:** `context_discovery.jinja2`
- **Returns:** `DiscoveryPlan`
- **Purpose:** Aggressively explore and identify relevant context
- **Outputs:** search queries, resource URIs, tools to expand

**Phase 2: Context Refinement**
- **Template:** `refine_context_for_execution.jinja2` (via `context_relevance.jinja2`)
- **Returns:** `RefinementPlan`
- **Purpose:** Filter and prioritize discovered context
- **Outputs:** exclusions, collapses, sorted segments

**Phase 3: Action Decision**
- **Template:** `action_decision.jinja2`
- **Returns:** `ActionDecision`
- **Purpose:** Make decisive action choice (execute or finalize)
- **Outputs:** situation analysis, execution plan OR finalization

**Phase 4: Refinement Analysis**
- **Template:** `refinement_analysis.jinja2`
- **Returns:** `RefinementDecision`
- **Purpose:** Strategic oversight and goal evolution
- **Outputs:** approval OR refined goal with exploration hints

### Iteration
A **complete cycle** through the orchestration loop. Each iteration may contain:
- Multiple context discovery/refinement cycles (inner loop)
- One action decision
- One refinement analysis
- Goal refinement if needed

**Controlled by:**
- `min_iter` - Minimum iterations before allowing execution
- `max_iter` - Maximum iterations to prevent infinite loops

### Context Loop
The **inner loop** within an iteration that repeatedly discovers and refines context until adequate. Allows multiple discovery/refinement cycles before making an action decision.

**Pattern:**
```
while True:
    discovery = discover_context()
    apply(discovery)
    if not discovery.refinement_needed:
        break
    refinement = refine_context()
    apply(refinement)
```

### StepOutcome
The **result of tool execution**, tracking success/failure and metadata.

**Contains:**
- `action`: StepDefinition (tool name, parameters, reason)
- `status`: StepStatus (SUCCESS, FAILED, SKIPPED)
- `results`: List of result strings
- `error_message`: Error details if failed
- `started_at`, `completed_at`: Timestamps
- `triggered_actions`: Actions triggered during execution
- `backlog_items`: Items added to backlog
- `progress`: Progress tracking information

### Exploration Hints
**Guidance provided by Refinement Analysis** when refining goals. Helps the next iteration focus on relevant areas.

**Components:**
- `search_terms` - Keywords to search for
- `resource_paths` - Specific resource URIs to load
- `tools_to_expand` - Additional tools that might be needed
- `focus_areas` - Conceptual areas to explore

### Last Planned Action
A **fallback mechanism** ensuring graceful degradation. The system tracks the most recent planned action throughout execution.

**Purpose:**
- Provides meaningful response if max iterations reached
- Can be either `OrchestrationExecutionPlan` or `OrchestrationFinalization`
- Updated with every action decision
- Never leaves user without response

### ResourceContext
The **context builder** that manages workspace state during orchestration. Handles resource loading, filtering, and search.

**Key Operations:**
- `search_resources(query)` - Semantic search
- `query_resource(uri)` - Load specific resource
- `exclude_resource(uri)` - Remove from context
- `exclude_property(uri, property)` - Filter property
- `sort_resources(uris)` - Reorder by priority
- `build_segment_data(segment)` - Generate context data

### ToolRuntime
The **interface provided to tools during execution**. Enables tools to interact with users and track state.

**Methods:**
- `debug(content)`, `message(content)` - Output messages
- `warning(content)`, `error(content)` - Error messages
- `confirmation(prompt)` - Ask yes/no question
- `user_input(prompt)` - Request input from user
- `progress(key, value, description)` - Track progress
- `trigger_action(action)` - Trigger future actions
- `backlog(content, priority)` - Add to backlog

**Implementation:** `AgentToolRuntime` routes to `AgentChannel`

### AgentChannel
The **communication interface** between Agent and user. Abstracts different interaction modes (session, shell, etc.).

**Methods:**
- `info(message)` - Informational message
- `error(message)` - Error message
- `debug(message)` - Debug output
- `confirm(prompt)` - Boolean confirmation
- `request(prompt)` - String input

**Implementations:**
- `SessionChannel` - Session-based communication
- `ShellSessionChannel` - Shell environment interaction

---

## Agent System Decision Matrix

Use this table to choose the correct term:

| Scenario | Use Term | Example |
|----------|----------|---------|
| Main controller | **Agent** | "Agent receives user request" |
| Decision-making engine | **OrchestrationLoop** | "OrchestrationLoop determines next action" |
| User's intent | **Goal** | "Goal evolves with discovered prerequisites" |
| Executable action | **Tool** | "Tool performs resource creation" |
| Tool visibility state | **Expanded/Collapsed** | "Tool is expanded to show full schema" |
| Orchestration stage | **Phase** | "Phase 3 makes action decision" |
| Complete orchestration cycle | **Iteration** | "After 3 iterations, goal is achieved" |
| Context discovery cycle | **Context Loop** | "Context loop runs until refinement not needed" |
| Tool execution result | **StepOutcome** | "StepOutcome tracks execution success" |
| Refinement guidance | **Exploration Hints** | "Exploration hints suggest resources to load" |
| Graceful degradation | **Last Planned Action** | "Return last planned action if max_iter reached" |
| Workspace state | **ResourceContext** | "ResourceContext manages loaded resources" |
| Tool interaction interface | **ToolRuntime** | "Tool receives ToolRuntime for user communication" |
| User communication | **AgentChannel** | "AgentChannel abstracts communication mode" |

---

## Agent System Naming Conventions

### Class Names
- `Agent` - Main goal pursuit controller
- `OrchestrationLoop` - Multi-phase decision engine
- `GoalBuilder` - Converts requests to goals
- `SchematicTool` - Tool with JSON schema
- `ToolRuntime` - Tool execution interface
- `AgentToolRuntime` - Agent's ToolRuntime implementation
- `AgentChannel` - Communication interface
- `ResourceContext` - Context management
- `StepDefinition` - Action specification
- `StepOutcome` - Execution result
- `OrchestrationExecutionPlan` - Tool execution decision
- `OrchestrationFinalization` - Completion decision
- `DiscoveryPlan` - Phase 1 output
- `RefinementPlan` - Phase 2 output
- `ActionDecision` - Phase 3 output
- `RefinementDecision` - Phase 4 output

### Method Names
- `handle_request(request)` - Agent's main entry point
- `execution_advance(...)` - OrchestrationLoop main method
- `build_goal(request)` - GoalBuilder creates goal
- `call(runtime, **params)` - Tool execution
- `_discover_and_expand_context()` - Phase 1 method
- `_filter_and_refine_context()` - Phase 2 method
- `_make_action_decision()` - Phase 3 method
- `_analyze_and_refine()` - Phase 4 method

### Variable Names
```python
# Good - Clear semantic meaning
agent: Agent = create_agent(...)
orchestrator: OrchestrationLoop = OrchestrationLoop(...)
goal: str = "Create protagonist named 余归"
tool: SchematicTool = tools["create_aspect"]
runtime: ToolRuntime = AgentToolRuntime(channel)
outcome: StepOutcome = await execute_tool(...)

# Good - Phase outputs
discovery_plan: DiscoveryPlan = await discover_context()
refinement_plan: RefinementPlan = await refine_context()
action_decision: ActionDecision = await make_decision()
refinement_decision: RefinementDecision = await analyze()

# Good - Orchestration state
completed_steps: list[StepOutcome] = []
pending_steps: list[str] = []
expanded_tools: set[str] = set()
last_planned_action: OrchestrationExecutionPlan | OrchestrationFinalization
```

---

## Integrated Conceptual Model

```
┌─────────────────────────────────────────────────────────────┐
│                    NovelRAG System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Agent System                            │  │
│  │  ┌────────┐    ┌──────────────────┐   ┌──────────┐ │  │
│  │  │ Agent  │───→│OrchestrationLoop │──→│  Tools   │ │  │
│  │  │        │    │  (4 Phases)      │   │          │ │  │
│  │  └────┬───┘    └────────┬─────────┘   └────┬─────┘ │  │
│  │       │                 │                    │       │  │
│  │       │      ┌──────────▼──────────┐        │       │  │
│  │       │      │  ResourceContext    │        │       │  │
│  │       │      │  (Context Builder)  │        │       │  │
│  │       │      └──────────┬──────────┘        │       │  │
│  └───────┼─────────────────┼───────────────────┼───────┘  │
│          │                 │                    │          │
│  ┌───────▼─────────────────▼────────────────────▼───────┐ │
│  │              Resource System                         │ │
│  │  ┌─────────┐   ┌──────────────┐   ┌─────────────┐  │ │
│  │  │ Aspects │──→│  Resources   │──→│  Elements   │  │ │
│  │  │         │   │  (Entities)  │   │  (Data)     │  │ │
│  │  └─────────┘   └──────────────┘   └─────────────┘  │ │
│  └──────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Execution Flow:
  User Request → Agent → OrchestrationLoop → [4 Phases] → Decision
                   ↓                             ↓
             Execute Tool ←──────────────── Execute/Finalize
                   ↓
         Modify Resources in Resource System
                   ↓
             Return Response to User
```

---

This terminology guide ensures consistent understanding across both the Resource System (data management) and Agent System (orchestration and execution) components of NovelRAG.
