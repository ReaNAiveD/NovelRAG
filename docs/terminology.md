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

Core concepts related to the autonomous goal pursuit system, including belief-driven decision-making, step-based execution, and tool interaction.

### Agent
An **autonomous intelligent entity** capable of self-directed decision-making to continuously refine and improve the Resource System content. The Agent operates based on its beliefs, interacts with the environment through a set of tools, pursues goals, and may autonomously decide to define new goals.

**Key Characteristics:**
- **Belief-driven**: Actions are guided by the Agent's beliefs
- **Goal-oriented**: Pursues observable, judgeable objectives
- **Tool-equipped**: Interacts with the environment via tools
- **Autonomous capability**: Can self-generate new goals in autonomous mode

**Operating Modes:**
- **Request-driven mode**: User requests are converted into goals for execution
- **Autonomous mode**: Agent decides goals based on beliefs, recent operations, backlog, and user context

**Main Method:** `handle_request(request: str) -> str`

---

### Belief
A **constraint and reference framework** that guides the Agent's behavior. Beliefs define how goals should be accomplished and serve as reference points when making decisions about new goals.

**Role in Agent System:**
- Constrains the approach to goal completion
- Provides reference for goal decision-making in autonomous mode
- Shapes the Agent's understanding of appropriate actions

**Note:** The belief system provides the foundational principles that ensure consistent and purposeful Agent behavior across different goals and contexts.

---

### Agent Channel
The **communication adapter** that enables the Agent to interact with external environments. Abstracts different interaction modes to allow the Agent framework to operate across various platforms.

**Supported Environments:**
- Shell/CLI interaction
- HTTP-based communication
- Session-based communication

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

### Request
A **user-initiated input** that triggers Agent activity.

**Behavior by Mode:**
- **Request-driven mode**: The request is transformed into a Goal and executed immediately
- **Autonomous mode**: The request is stored in a dedicated context for consideration during Goal Decision

**Processing:** In request-driven mode, requests are converted to goals via `GoalBuilder`.

---

### Goal Decision
The **strategic decision-making process** in autonomous mode that generates new goals. This process balances priority, depth, and breadth of exploration.

**Input Factors:**
- Agent's beliefs (guiding principles)
- Recent operations (execution history)
- Backlog items (pending tasks)
- User requests (stored in context)

**Output:** A prioritized Goal with clear success criteria.

**Note:** Goal Decision is only active in autonomous mode. In request-driven mode, goals are directly derived from user requests.

---

### Goal
An **observable and judgeable result description** that the Agent aims to achieve in the current context. Goals are independent of specific implementation steps and directly constrain the Agent's action scope and completion criteria.

**Characteristics:**
- **Observable**: Progress and completion can be measured
- **Judgeable**: Success or failure can be definitively determined
- **Implementation-agnostic**: Describes WHAT, not HOW
- **Constraining**: Bounds the scope of Agent actions

**Example:**
```
Goal: "Create a protagonist character named 余归 with background as a scholar"
- Observable: Character resource can be queried
- Judgeable: Resource exists with correct properties
- Does not specify: Which tools to use or in what order
```

---

### Pursuit
The **dynamic process** of an Agent working to complete a Goal. A Pursuit consists of multiple Steps and grows as the Agent takes concrete actions toward the goal.

**Characteristics:**
- **Goal-bound**: Each Pursuit is associated with exactly one Goal
- **Multi-step**: Composed of a sequence of Steps
- **Dynamic**: Evolves as the Agent executes actions
- **Tracked**: Maintains state via Pursuit State

**Lifecycle:**
```
Goal Created → Pursuit Started → [Steps Executed...] → Resolution → Pursuit Ended
```

---

### Step
An **atomic behavioral unit** within a Pursuit. Each Step aims to move closer to the goal state. Steps encapsulate the smallest granularity of Agent choices, such as invoking specific tools or taking specific actions.

**Characteristics:**
- **Goal-approaching**: Each step moves toward the goal state
- **Atomic**: Represents a single decision point
- **Bounded**: Contains one Determination cycle
- **Executable**: Results in a concrete action or termination

**Step Structure:**
- Contains a Determination Loop that produces a Directive
- Directive is either an Operation (continue) or Resolution (terminate)

---

### Pursuit State
A **comprehensive description** of the current pursuit progress and its expected evolution. Pursuit State serves as the foundation for Step-level action decisions.

**Components:**
- **Goal**: The target being pursued
- **Completed items**: What has been accomplished
- **Remaining requirements**: What still needs to be done
- **Expected actions**: Anticipated next steps
- **Boundary conditions**: Constraints and limitations
- **Fallback strategies**: Error handling approaches
- **Success criteria**: How to judge completion

**Usage:** Each Step's Determination process consults the Pursuit State to make informed decisions.

---

### Determination
The **action decision loop** within each Step. Determination operates under incomplete context, repeatedly discovering and refining information until it can produce an executable Directive or decide to terminate.

**Process:**
```
while not decided:
    discover_context()
    refine_context()
    attempt_decision()
    if can_decide:
        produce Directive
        break
```

**Output:** A Directive (either Operation or Resolution)

**Note:** The current implementation is tightly coupled with the Resource System. Future refactoring will abstract this dependency.

---

### Context
The **internal implementation detail** of the Determination Loop. Context manages workspace state during the decision-making process.

**Design Approach:**
For tools that require detailed context data, the `ResourceContext` is passed as a constructor parameter to the tool. This design decouples the Agent framework from Context implementation.

**Pattern:**
```python
# Context-dependent tool receives ResourceContext at construction
tool = ContextAwareTool(resource_context=context)

# Agent framework remains context-agnostic
agent.register_tool(tool)
```

**Key Operations:**
- `search_resources(query)` - Semantic search
- `query_resource(uri)` - Load specific resource
- `exclude_resource(uri)` - Remove from context
- `sort_resources(uris)` - Reorder by priority

---

### Directive
The **decision product** of a Step's Determination Loop. A Directive represents the Agent's decision at a given stage.

**Types:**
- **Operation**: Execute a specific action and continue the Pursuit
- **Resolution**: Terminate the Pursuit with a final status

**Relationship:**
```
Determination Loop ──produces──> Directive
                                    │
                    ┌───────────────┴───────────────┐
                    ↓                               ↓
                Operation                       Resolution
            (continue pursuit)              (terminate pursuit)
```

---

### Operation
A **type of Directive** that instructs the Agent to execute a specific action (typically a tool invocation) and continue the Pursuit.

**Characteristics:**
- Specifies the tool to invoke
- Includes parameters for the tool
- Provides reasoning for the action
- Expects execution to continue after completion

**Structure:**
```python
Operation(
    tool="create_resource",
    parameters={"aspect": "character", "id": "protagonist"},
    reason="Creating the protagonist character as required by the goal"
)
```

---

### Resolution
A **type of Directive** that terminates the Pursuit. Resolution includes the final status (success or failure) and the response to return.

**Status Types:**
- **Success**: Goal was achieved
- **Failure**: Goal could not be achieved

**Structure:**
```python
Resolution(
    status=ResolutionStatus.SUCCESS,
    response="Successfully created protagonist character 余归"
)
```

---

### Tool
An **executable interface** for Agent-environment interaction. Tools are the primary mechanism through which the Agent affects the world and gathers information.

**Capabilities:**
- **Self-describing**: Provides schema and documentation for Agent decision-making
- **User communication**: Can interact with users through various means
- **Agent communication**: Can influence Agent internals (backlog, progress tracking)
- **Environment interaction**: Performs actual operations on resources

**Types:**
- `BaseTool` - Abstract base interface
- `SchematicTool` - Tools with JSON schema for parameters

**Tool Interface:**
```python
class SchematicTool:
    name: str                    # Tool identifier
    description: str             # What the tool does
    input_schema: dict           # JSON Schema for parameters
    prerequisites: str           # Conditions for use
    output_description: str      # What the tool returns
    
    async def call(runtime: ToolRuntime, **params) -> ToolOutput
```

---

### Tool Runtime
The **interface provided to tools during execution**. Enables tools to communicate with users and affect Agent state.

**Communication Methods:**
- `debug(content)`, `message(content)` - Output messages
- `warning(content)`, `error(content)` - Error messages
- `confirmation(prompt)` - Ask yes/no question
- `user_input(prompt)` - Request input from user

**State Influence Methods:**
- `progress(key, value, description)` - Track progress
- `backlog(content, priority)` - Add items to backlog

**Implementation:** `AgentToolRuntime` routes calls to `AgentChannel`

---

### Tool Output
The **result returned by a tool** after execution. Tool Output has two categories.

**Output Types:**
- **ToolResult**: Successful execution with result data
- **ToolError**: Failed execution with error information

**Structure:**
```python
ToolOutput = ToolResult | ToolError

class ToolResult:
    type: "output"
    content: str      # Success result

class ToolError:
    type: "error"
    message: str      # Error description
```

---

### Operation Outcome
A **wrapped representation** of Tool Output with additional contextual information. Operation Outcome captures not just the tool's response but also the broader context of the Step execution.

**Contains:**
- Tool Output (success or error)
- Step metadata (timing, parameters)
- Side effects (triggered actions, backlog items)
- Progress information

**Structure:**
```python
@dataclass
class OperationOutcome:
    action: StepDefinition        # What was executed
    status: StepStatus            # SUCCESS, FAILED, CANCELLED
    results: list[str]            # Output from tool
    error_message: str | None     # Error if failed
    triggered_actions: list       # Side effects
    backlog_items: list[str]      # Added backlog items
    started_at: datetime | None
    completed_at: datetime | None
```

---

## Tool State

Tools can be in two visibility states within the Determination process:

**Collapsed State:**
- Only name and description visible to LLM
- Minimal context consumption
- Default state for all tools

**Expanded State:**
- Full schema with parameters, types, descriptions visible
- Higher context consumption but necessary for execution planning
- Dynamically expanded/collapsed during Determination

---

## Agent System Decision Matrix

Use this table to choose the correct term:

| Scenario | Use Term | Example |
|----------|----------|---------|
| Autonomous intelligent entity | **Agent** | "Agent pursues goals autonomously" |
| Guiding principles/constraints | **Belief** | "Agent's beliefs constrain its approach" |
| User-initiated input | **Request** | "Request is converted to a Goal" |
| Autonomous goal generation | **Goal Decision** | "Goal Decision considers backlog and beliefs" |
| Observable target outcome | **Goal** | "Goal: Create protagonist character" |
| Goal completion process | **Pursuit** | "Pursuit contains multiple Steps" |
| Atomic behavior unit | **Step** | "Each Step approaches the goal state" |
| Pursuit progress description | **Pursuit State** | "Pursuit State tracks remaining work" |
| Step-level decision loop | **Determination** | "Determination discovers and refines context" |
| Decision product | **Directive** | "Directive can be Operation or Resolution" |
| Continue-execution directive | **Operation** | "Operation invokes a tool" |
| Terminate-pursuit directive | **Resolution** | "Resolution ends with success/failure" |
| Environment interaction unit | **Tool** | "Tool performs resource creation" |
| Tool execution interface | **Tool Runtime** | "Tool Runtime enables user communication" |
| Tool result (success/error) | **Tool Output** | "Tool Output indicates success or error" |
| Wrapped execution result | **Operation Outcome** | "Operation Outcome includes timing and side effects" |
| External communication | **Agent Channel** | "Agent Channel abstracts Shell/HTTP" |
| Determination workspace | **Context** | "Context manages resource state" |

---

## Agent System Naming Conventions

### Class Names
- `Agent` - Autonomous goal pursuit controller
- `Belief` - Agent's guiding principles (planned)
- `Goal` - Target outcome description
- `Pursuit` - Goal completion process
- `PursuitState` - Progress and evolution description
- `Step` - Atomic behavior unit
- `Determination` - Action determination loop (planned refactor from `OrchestrationLoop`)
- `Directive` - Decision product base class
- `Operation` - Continue-execution directive
- `Resolution` - Terminate-pursuit directive
- `Tool`, `SchematicTool` - Tool implementations
- `ToolRuntime` - Tool execution interface
- `ToolOutput`, `ToolResult`, `ToolError` - Tool result types
- `OperationOutcome` - Wrapped execution result (refactor from `StepOutcome`)
- `AgentChannel` - Communication interface
- `ResourceContext` - Context implementation for Resource System

### Method Names
- `handle_request(request)` - Agent's main entry point
- `pursue(goal)` - Start goal pursuit
- `advance_step()` - Execute next step in pursuit
- `determine()` - Run determination loop
- `call(runtime, **params)` - Tool execution

### Variable Names
```python
# Good - Clear semantic meaning
agent: Agent = create_agent(...)
belief: Belief = agent.belief
goal: Goal = Goal("Create protagonist named 余归")
pursuit: Pursuit = agent.pursue(goal)
step: Step = pursuit.current_step
directive: Directive = step.determine()

# Good - Directive types
operation: Operation = Operation(tool="create_resource", ...)
resolution: Resolution = Resolution(status=SUCCESS, response="Done")

# Good - Execution tracking
outcome: OperationOutcome = await execute_operation(operation)
tool_output: ToolOutput = await tool.call(runtime, **params)
```

---

## Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Agent Execution Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐                          │
│  │   Request   │     │    Belief   │                          │
│  │  (User or   │     │  (Guiding   │                          │
│  │  Autonomous)│     │  Principles)│                          │
│  └──────┬──────┘     └──────┬──────┘                          │
│         │                    │                                  │
│         └────────┬───────────┘                                  │
│                  ↓                                              │
│         ┌────────────────┐                                     │
│         │  Goal Decision │  (autonomous mode)                  │
│         │       or       │                                     │
│         │  Goal Builder  │  (request-driven mode)              │
│         └───────┬────────┘                                     │
│                 ↓                                              │
│         ┌───────────────┐                                      │
│         │     Goal      │                                      │
│         │  (Observable  │                                      │
│         │   Outcome)    │                                      │
│         └───────┬───────┘                                      │
│                 ↓                                              │
│         ┌───────────────┐                                      │
│         │    Pursuit    │◄────────────────────────┐           │
│         │  (Dynamic     │                         │           │
│         │   Process)    │                         │           │
│         └───────┬───────┘                         │           │
│                 ↓                                 │           │
│         ┌───────────────┐                         │           │
│         │ Pursuit State │                         │           │
│         │  (Progress    │                         │           │
│         │  Description) │                         │           │
│         └───────┬───────┘                         │           │
│                 ↓                                 │           │
│         ┌───────────────┐                         │           │
│         │     Step      │                         │           │
│         │   (Atomic     │                         │           │
│         │   Behavior)   │                         │           │
│         └───────┬───────┘                         │           │
│                 ↓                                 │           │
│     ┌───────────────────────┐                     │           │
│     │    Determination      │                     │           │
│     │  ┌─────────────────┐  │                     │           │
│     │  │ Context Loop    │  │                     │           │
│     │  │ (discover →     │  │                     │           │
│     │  │  refine →       │  │                     │           │
│     │  │  decide)        │  │                     │           │
│     │  └────────┬────────┘  │                     │           │
│     └───────────┼───────────┘                     │           │
│                 ↓                                 │           │
│         ┌───────────────┐                         │           │
│         │   Directive   │                         │           │
│         └───────┬───────┘                         │           │
│                 │                                 │           │
│       ┌─────────┴─────────┐                       │           │
│       ↓                   ↓                       │           │
│  ┌──────────┐      ┌────────────┐                │           │
│  │Operation │      │ Resolution │                │           │
│  │(continue)│      │(terminate) │                │           │
│  └────┬─────┘      └─────┬──────┘                │           │
│       │                  │                        │           │
│       ↓                  ↓                        │           │
│  ┌──────────┐      ┌────────────┐                │           │
│  │   Tool   │      │  Final     │                │           │
│  │Execution │      │  Response  │                │           │
│  └────┬─────┘      └────────────┘                │           │
│       ↓                                          │           │
│  ┌────────────┐                                  │           │
│  │ Operation  │──────────────────────────────────┘           │
│  │  Outcome   │  (next step)                                 │
│  └────────────┘                                              │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

---

## Integrated Conceptual Model

```
┌─────────────────────────────────────────────────────────────┐
│                    NovelRAG System                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                   Agent System                        │  │
│  │                                                       │  │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────────────────┐│  │
│  │  │  Agent  │──→│  Goal   │──→│      Pursuit        ││  │
│  │  │(Belief) │   │         │   │  ┌───────────────┐  ││  │
│  │  └────┬────┘   └─────────┘   │  │     Step      │  ││  │
│  │       │                      │  │ ┌───────────┐ │  ││  │
│  │       │                      │  │ │Determina- │ │  ││  │
│  │       │                      │  │ │   tion    │ │  ││  │
│  │       │                      │  │ └─────┬─────┘ │  ││  │
│  │       │                      │  │       ↓       │  ││  │
│  │       │                      │  │  ┌─────────┐  │  ││  │
│  │       │                      │  │  │Directive│  │  ││  │
│  │       │                      │  │  └────┬────┘  │  ││  │
│  │       │                      │  └───────┼───────┘  ││  │
│  │       │                      └──────────┼──────────┘│  │
│  │       │                                 │           │  │
│  │       │      ┌──────────────────────────┘           │  │
│  │       │      ↓                                      │  │
│  │       │  ┌───────────────┐                          │  │
│  │       │  │     Tools     │←── Tool Runtime          │  │
│  │       │  └───────┬───────┘                          │  │
│  │       │          │                                  │  │
│  │       │          │ (via Context)                    │  │
│  └───────┼──────────┼──────────────────────────────────┘  │
│          │          │                                      │
│  ┌───────▼──────────▼──────────────────────────────────┐  │
│  │              Resource System                         │  │
│  │  ┌─────────┐   ┌──────────────┐   ┌─────────────┐  │  │
│  │  │ Aspects │──→│  Resources   │──→│  Elements   │  │  │
│  │  │         │   │  (Entities)  │   │  (Data)     │  │  │
│  │  └─────────┘   └──────────────┘   └─────────────┘  │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Migration Notes

For developers migrating from the previous terminology:

### Term Mappings

| Old Term | New Term | Notes |
|----------|----------|-------|
| `OrchestrationLoop` | `Determination` | Per-step decision loop |
| `OrchestrationExecutionPlan` | `Operation` | Directive to continue |
| `OrchestrationFinalization` | `Resolution` | Directive to terminate |
| `StepOutcome` | `OperationOutcome` | Wrapped execution result |
| `Iteration` | (within Pursuit) | Now part of Step sequence |
| `Phase` | (within Determination) | Internal to determination |
| `Context Loop` | (within Determination) | Part of determine() |

### What Changed
- `OrchestrationLoop` → Conceptually replaced by `Determination` (step-level)
- `StepOutcome` → `OperationOutcome` (clarifies it wraps Operation results)
- Execution flow restructured around Pursuit → Step → Determination hierarchy

### What Did NOT Change
- `Agent` class (main controller)
- `Tool`, `SchematicTool` classes
- `ToolRuntime` interface
- `AgentChannel` interface
- `ResourceContext` (now explicitly decoupled via constructor injection)

### Rationale
The new terminology:
- Clarifies the hierarchical structure: Goal → Pursuit → Step → Determination → Directive
- Distinguishes between Directives (decisions) and their outcomes
- Makes the autonomous capability explicit with Belief and Goal Decision
- Decouples Agent framework from Resource System via Context injection

---

This terminology guide ensures consistent understanding across the Agent System components, from high-level goal management to low-level tool execution.
