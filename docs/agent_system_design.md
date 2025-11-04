# NovelRAG Agent System Design Document

## 1. System Overview

### 1.1 Project Background
The NovelRAG Agent System is a context-driven intelligent execution framework designed to achieve automated goal pursuit through dynamic context management and multi-phase strategic orchestration. The system uses a sophisticated OrchestrationLoop architecture with four distinct phases for context discovery, refinement, decision-making, and quality analysis.

### 1.2 Core Design Principles
- **Context-Driven**: Decision-making based on dynamically discovered and refined resource contexts
- **Multi-Phase Orchestration**: Strategic decisions through four distinct phases with dedicated LLM templates
- **Iterative Refinement**: Context and goals evolve through multiple iterations to achieve optimal execution
- **Tool Atomization**: Each tool is an independent, composable functional unit
- **Dynamic Tool Management**: Tools expand/collapse based on relevance to reduce context overhead
- **State Transparency**: All execution states are traceable and debuggable

### 1.3 Architectural Advantages
Compared to traditional step-based planning approaches, the multi-phase architecture offers:
- **Enhanced Context Discovery**: Dedicated phase for aggressive context exploration
- **Intelligent Filtering**: Separate refinement phase ensures only relevant context is retained
- **Strategic Oversight**: Refinement analysis phase catches planning errors and missing prerequisites
- **Goal Evolution**: Goals naturally evolve to incorporate discovered requirements
- **Adaptive Tool Management**: Dynamic expansion/collapse reduces LLM context size
- **Better Scalability**: Clear separation of concerns across four phases
- **Higher Controllability**: Each phase has specific responsibility and validation
- **Stronger Debugging**: Complete execution traces with phase-by-phase decision logging

## 2. System Architecture

### 2.1 Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    NovelRAG Agent System                   │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Agent     │  │ OrchestrationLoop│  │ ResourceContext │ │
│  │             │──│                 │──│                 │ │
│  │ • pursue_goal│  │ • execution_advance│ • refine       │ │
│  │ • _execute_ │  │ • _context_advance │ • build_context│ │
│  │   tool      │  │                 │  │                 │ │
│  └─────────────┘  └─────────────────┘  └─────────────────┘ │
│         │                    │                    │         │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │AgentToolRuntime│ │   Tool System   │  │ Resource System │ │
│  │             │  │                 │  │                 │ │
│  │ • debug     │  │ • SchematicTool │  │ • Repository    │ │
│  │ • message   │  │ • ToolRuntime   │  │ • Aspects       │ │
│  │ • progress  │  │ • ToolOutput    │  │ • Elements      │ │
│  └─────────────┘  └─────────────────┘  └─────────────────┘ │
│         │                    │                    │         │
│  ┌─────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ AgentChannel│  │ Template System │  │   LLM System    │ │
│  │             │  │                 │  │                 │ │
│  │ • info      │  │ • Jinja2        │  │ • ChatLLM       │ │
│  │ • error     │  │ • Structured    │  │ • JSON Schema   │ │
│  │ • confirm   │  │   Output        │  │ • Validation    │ │
│  └─────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Core Component Relationships

#### 2.2.1 Agent
- **Responsibility**: Main controller for goal pursuit
- **Core Method**: `pursue_goal()` - Receives user goals and coordinates execution
- **Execution Flow**: 
  1. Create new ResourceContext
  2. Loop call OrchestrationLoop.execution_advance()
  3. Execute recommended tools
  4. Update execution state
  5. Return final result

#### 2.2.2 OrchestrationLoop
- **Responsibility**: Multi-phase strategic decision-making and execution orchestration
- **Core Method**: `execution_advance()` - Orchestrates four-phase decision process
- **Four Phases**:
  1. **Context Discovery** (`_discover_and_expand_context()`) - Find relevant context
  2. **Context Refinement** (`_filter_and_refine_context()`) - Filter and prioritize context
  3. **Action Decision** (`_make_action_decision()`) - Decide to execute tool or finalize
  4. **Refinement Analysis** (`_analyze_and_refine()`) - Validate decision or refine goal
- **Decision Types**:
  - `OrchestrationExecutionPlan`: Execute specific tool
  - `OrchestrationFinalization`: Complete goal pursuit
- **Key Features**:
  - Dynamic tool expansion/collapse
  - Iterative goal refinement
  - Min/max iteration control
  - Last action tracking for graceful fallback

#### 2.2.3 ResourceContext
- **Responsibility**: Dynamic construction and management of execution context
- **Core Functions**:
  - Selective loading of resource aspects
  - Filter and sort resource segments
  - Search history management
  - Context refinement

## 3. Detailed Design

### 3.1 OrchestrationLoop Design

#### 3.1.1 Core Algorithm

The orchestration loop implements a sophisticated multi-phase architecture:

```python
async def execution_advance(self, user_request, goal, completed_steps, pending_steps, available_tools):
    iter_num = 0
    current_goal = goal
    last_planned_action = OrchestrationFinalization(...)  # Default fallback
    
    while iter_num < max_iter:
        # INNER LOOP: Context Discovery & Refinement
        while True:
            # Phase 1: Context Discovery
            discovery_plan = await self._discover_and_expand_context(
                user_request, current_goal, available_tools
            )
            await self._apply_discovery_plan(discovery_plan)
            
            if not discovery_plan.refinement_needed:
                break
            
            # Phase 2: Context Refinement
            refinement_plan = await self._filter_and_refine_context(
                current_goal, available_tools, discovery_plan.discovery_analysis
            )
            await self._apply_refinement_plan(refinement_plan)
        
        # Phase 3: Action Decision
        action_decision = await self._make_action_decision(
            user_request, current_goal, completed_steps, available_tools
        )
        
        # Phase 4: Refinement Analysis
        refinement_decision = await self._analyze_and_refine(
            user_request, current_goal, action_decision, completed_steps, available_tools
        )
        
        # Store last planned action for fallback
        last_planned_action = self._convert_to_orchestration_action(action_decision)
        
        # Process refinement verdict
        if refinement_decision.verdict == "approve":
            if isinstance(last_planned_action, OrchestrationExecutionPlan) and iter_num >= min_iter:
                return last_planned_action
            elif isinstance(last_planned_action, OrchestrationFinalization):
                return last_planned_action
        else:
            # Refine goal and apply exploration hints
            if refinement_decision.refinement:
                current_goal = refinement_decision.refinement.get("refined_goal", current_goal)
                await self._apply_context_gap_suggestions(
                    refinement_decision.refinement.get("exploration_hints", {})
                )
        
        iter_num += 1
    
    # Graceful fallback: return last planned action
    return last_planned_action
```

#### 3.1.2 Multi-Phase Template Integration

The system uses dedicated Jinja2 templates for each phase:

**Phase 1: Context Discovery**
- **Template**: `context_discovery.jinja2`
- **Output**: `DiscoveryPlan` with search queries, resource URIs, tools to expand
- **Purpose**: Aggressively explore and identify relevant context

**Phase 2: Context Refinement**
- **Template**: `context_relevance.jinja2` (used via `refine_context_for_execution.jinja2`)
- **Output**: `RefinementPlan` with exclusions, collapses, sorted segments
- **Purpose**: Filter and prioritize discovered context

**Phase 3: Action Decision**
- **Template**: `action_decision.jinja2`
- **Output**: `ActionDecision` with execution or finalization decision
- **Purpose**: Make decisive action choice (no ambiguous states)

**Phase 4: Refinement Analysis**
- **Template**: `refinement_analysis.jinja2`
- **Output**: `RefinementDecision` with approval or goal refinement
- **Purpose**: Strategic oversight and goal evolution

#### 3.1.3 Data Structures

```python
@dataclass(frozen=True)
class DiscoveryPlan:
    discovery_analysis: str
    search_queries: list[str]
    query_resources: list[str]
    expand_tools: list[str]

@dataclass(frozen=True)
class RefinementPlan:
    exclude_resources: list[str]
    exclude_properties: list[dict]
    collapse_tools: list[str]
    sorted_segments: list[str]

@dataclass(frozen=True)
class ActionDecision:
    situation_analysis: str
    decision_type: str  # "execute" or "finalize"
    execution: dict | None
    finalization: dict | None
    context_verification: dict

@dataclass(frozen=True)
class RefinementDecision:
    analysis: dict
    verdict: str  # "approve" or "refine"
    approval: dict | None
    refinement: dict | None
```

#### 3.1.4 Context Management Mechanisms

**Dynamic Resource Loading**
- Context discovery phase identifies needed resources
- Resources loaded incrementally via search and direct queries
- Search history tracked to avoid redundant queries

**Tool Expansion/Collapse**
- Tools start collapsed (summary only)
- Discovery phase expands relevant tools (full schema)
- Refinement phase collapses irrelevant tools
- Reduces LLM context size while maintaining necessary detail

**Goal Evolution**
- Goals accumulate discovered prerequisites across iterations
- Refinement analysis adds exploration hints
- Each iteration builds on previous knowledge

**Graceful Degradation**
- `last_planned_action` tracks best available decision
- Min iteration requirement prevents premature execution
- Max iteration limit ensures termination
- Always returns meaningful action (execute or finalize)

### 3.2 ResourceContext Design

#### 3.2.1 Data Structures

```python
@dataclass
class ResourceSegment:
    uri: str
    included_properties: set[str]
    excluded_properties: set[str]

@dataclass 
class ContextWorkspace:
    segments: dict[str, ResourceSegment]
    sorted_uris: list[str]
    excluded_uris: set[str]
```

#### 3.2.2 Core Functions

##### Property Selection Mechanism
- **include_properties**: Explicitly included properties (dictionary format: `{"uri": "property"}`)
- **exclude_properties**: Explicitly excluded properties (dictionary format: `{"uri": "property"}`)
- **pending_properties**: Properties not yet decided

##### Resource Filtering
- **URI-level filtering**: Complete exclusion of certain resources
- **Property-level filtering**: Selective inclusion/exclusion of specific resource properties
- **Relationship filtering**: Filter out relationships to excluded resources

##### Context Construction
```python
async def build_context_for_planning(self):
    segments_data = []
    for segment in self.workspace.sorted_segments():
        data = await self.build_segment_data(segment)
        if data:
            segments_data.append(data)
    
    return {
        "resource_segments": segments_data,
        "search_history": self.search_history
    }
```

### 3.3 Tool System Design

#### 3.3.1 Tool Interface Hierarchy

```python
# Base tool abstraction
class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    def description(self) -> str | None: ...

# Schematic tool
class SchematicTool(BaseTool):
    @property
    @abstractmethod
    def input_schema(self) -> dict: ...
    
    @abstractmethod
    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput: ...
```

#### 3.3.2 ToolRuntime Interface

ToolRuntime provides interaction interfaces during tool execution:

```python
class ToolRuntime(ABC):
    # Message output
    async def debug(self, content: str): ...
    async def message(self, content: str): ...
    async def warning(self, content: str): ...
    async def error(self, content: str): ...
    
    # User interaction
    async def confirmation(self, prompt: str) -> bool: ...
    async def user_input(self, prompt: str) -> str: ...
    
    # State management
    async def progress(self, key: str, value: str, description: str | None = None): ...
    async def trigger_action(self, action: dict[str, str]): ...
    async def backlog(self, content: dict, priority: str | None = None): ...
```

#### 3.3.3 AgentToolRuntime Implementation

Routes tool runtime calls to Agent's communication channel:

```python
class AgentToolRuntime(ToolRuntime):
    def __init__(self, channel: AgentChannel):
        self.channel = channel
        self._backlog: list[str] = []
        self._progress: dict[str, list[str]] = {}
        self._triggered_actions: list[dict[str, str]] = []
    
    async def message(self, content: str):
        await self.channel.info(content)
    
    # ... other method implementations
```

### 3.4 Communication System Design

#### 3.4.1 AgentChannel Abstraction

```python
class AgentChannel(ABC):
    @abstractmethod
    async def info(self, message: str): ...
    
    @abstractmethod  
    async def error(self, message: str): ...
    
    @abstractmethod
    async def debug(self, message: str): ...
    
    @abstractmethod
    async def confirm(self, prompt: str) -> bool: ...
    
    @abstractmethod
    async def request(self, prompt: str) -> str: ...
```

#### 3.4.2 Implementation Types
- **SessionChannel**: Session-based communication
- **ShellSessionChannel**: User interaction in Shell environment

## 4. Execution Flow

### 4.1 Goal Pursuit Flow

```
User Goal → Agent.handle_request()
    ↓
Create Goal via GoalBuilder
    ↓
Create OrchestrationLoop
    ↓
┌────────────────────────────────────────────────────────────┐
│              Main Orchestration Loop                       │
│                                                            │
│  ┌──────────────────────────────────────────────────┐   │
│  │       Context Discovery & Refinement Loop        │   │
│  │                                                  │   │
│  │  1. Phase 1: Context Discovery                  │   │
│  │     ├─ _discover_and_expand_context()          │   │
│  │     ├─ Identify search queries                  │   │
│  │     ├─ Identify resources to load               │   │
│  │     └─ Identify tools to expand                 │   │
│  │                                                  │   │
│  │  2. Apply Discovery Plan                        │   │
│  │     ├─ Execute searches                         │   │
│  │     ├─ Load resources                           │   │
│  │     └─ Expand tools                             │   │
│  │                                                  │   │
│  │  3. Check if refinement needed                  │   │
│  │     └─ If no: break to action decision          │   │
│  │                                                  │   │
│  │  4. Phase 2: Context Refinement                 │   │
│  │     ├─ _filter_and_refine_context()            │   │
│  │     ├─ Identify resources to exclude            │   │
│  │     ├─ Identify properties to exclude           │   │
│  │     ├─ Identify tools to collapse               │   │
│  │     └─ Sort context segments                    │   │
│  │                                                  │   │
│  │  5. Apply Refinement Plan                       │   │
│  │     ├─ Exclude resources/properties             │   │
│  │     ├─ Collapse tools                           │   │
│  │     └─ Sort segments                            │   │
│  │                                                  │   │
│  │  6. Loop until context adequate                 │   │
│  └──────────────────────────────────────────────────┘   │
│                                                            │
│  7. Phase 3: Action Decision                              │
│     ├─ _make_action_decision()                           │
│     ├─ Analyze situation                                 │
│     └─ Return execute or finalize decision               │
│                                                            │
│  8. Store as last_planned_action                          │
│                                                            │
│  9. Phase 4: Refinement Analysis                          │
│     ├─ _analyze_and_refine()                             │
│     ├─ Quality assessment                                 │
│     └─ Return approve or refine verdict                  │
│                                                            │
│  10. Process Verdict:                                     │
│      ├─ If approve + execute + min_iter_met:             │
│      │   └─ Return execution plan to Agent               │
│      ├─ If approve + finalize:                           │
│      │   └─ Return finalization to Agent                 │
│      └─ If refine:                                        │
│          ├─ Update goal with discoveries                 │
│          ├─ Apply exploration hints                      │
│          └─ Continue main loop                           │
│                                                            │
│  11. Check iteration limits                               │
│      └─ If max_iter reached: return last_planned_action  │
└────────────────────────────────────────────────────────────┘
    ↓
If ExecutionPlan:
    ├─ Agent._execute_tool()
    ├─ Update completed_steps
    └─ Continue orchestration loop
    
If Finalization:
    └─ Return final response to user
```

### 4.2 Context Management Flow

```
Context Discovery & Refinement Loop
    ↓
┌──────────────────────────────────────────────────┐
│          Phase 1: Context Discovery              │
│                                                  │
│  Input: user_request, goal, available_tools      │
│  Template: context_discovery.jinja2              │
│                                                  │
│  Analysis:                                       │
│  ├─ What context is needed?                     │
│  ├─ What can be searched?                       │
│  ├─ What resources should be loaded?            │
│  └─ What tools need full schemas?               │
│                                                  │
│  Output: DiscoveryPlan                           │
│  ├─ discovery_analysis                          │
│  ├─ search_queries: ["protagonist", ...]        │
│  ├─ query_resources: ["/character/john", ...]   │
│  └─ expand_tools: ["create_aspect", ...]        │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│         Apply Discovery Plan                     │
│                                                  │
│  For each search_query:                          │
│    └─ context.search_resources(query)           │
│                                                  │
│  For each query_resource:                        │
│    └─ context.query_resource(uri)               │
│                                                  │
│  For each expand_tool:                           │
│    └─ expanded_tools.add(tool_name)             │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│    Check if Refinement Needed                    │
│                                                  │
│  If search_queries or query_resources empty:     │
│    └─ Break to Action Decision                  │
│  Else:                                           │
│    └─ Continue to Refinement Phase              │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│         Phase 2: Context Refinement              │
│                                                  │
│  Input: goal, available_tools, discovery_analysis│
│  Template: refine_context_for_execution.jinja2   │
│                                                  │
│  Analysis:                                       │
│  ├─ What context is irrelevant?                 │
│  ├─ What properties are unnecessary?            │
│  ├─ What tools can be collapsed?                │
│  └─ How should segments be prioritized?         │
│                                                  │
│  Output: RefinementPlan                          │
│  ├─ exclude_resources: ["/location/x", ...]     │
│  ├─ exclude_properties: [{"uri": "...", ...}]   │
│  ├─ collapse_tools: ["tool1", ...]              │
│  └─ sorted_segments: [ordered URIs]             │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│        Apply Refinement Plan                     │
│                                                  │
│  For each exclude_resource:                      │
│    └─ context.exclude_resource(uri)             │
│                                                  │
│  For each exclude_property:                      │
│    └─ context.exclude_property(uri, property)   │
│                                                  │
│  For each collapse_tool:                         │
│    └─ expanded_tools.remove(tool_name)          │
│                                                  │
│  If sorted_segments provided:                    │
│    └─ context.sort_resources(sorted_segments)   │
└──────────────────────────────────────────────────┘
    ↓
Loop back to Phase 1 if more discovery needed
```

### 4.3 Decision and Refinement Flow

```
┌──────────────────────────────────────────────────┐
│        Phase 3: Action Decision                  │
│                                                  │
│  Input: user_request, goal, completed_steps      │
│  Template: action_decision.jinja2                │
│                                                  │
│  Analysis:                                       │
│  ├─ Assess current situation                    │
│  ├─ Verify context completeness                 │
│  ├─ Check tool prerequisites                    │
│  └─ Make decisive choice                        │
│                                                  │
│  Output: ActionDecision (ALWAYS decisive)        │
│  ├─ situation_analysis                          │
│  ├─ decision_type: "execute" or "finalize"      │
│  ├─ execution: {tool, params, confidence}       │
│  ├─ finalization: {status, response, gaps}      │
│  └─ context_verification                        │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│    Store as last_planned_action                  │
│                                                  │
│  Convert ActionDecision to orchestration type:   │
│  ├─ If execute → OrchestrationExecutionPlan     │
│  └─ If finalize → OrchestrationFinalization     │
│                                                  │
│  Purpose: Graceful fallback if max_iter reached  │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│       Phase 4: Refinement Analysis               │
│                                                  │
│  Input: user_request, goal, action_decision      │
│  Template: refinement_analysis.jinja2            │
│                                                  │
│  Analysis:                                       │
│  ├─ Assess decision quality                     │
│  ├─ Identify missing prerequisites              │
│  ├─ Check for better alternatives               │
│  └─ Determine if goal needs enhancement         │
│                                                  │
│  Output: RefinementDecision                      │
│  ├─ analysis: {quality, issues}                 │
│  ├─ verdict: "approve" or "refine"              │
│  ├─ approval: {ready, confidence, notes}        │
│  └─ refinement: {                               │
│      ├─ refined_goal: "Enhanced goal..."        │
│      ├─ additions: [discovered requirements]    │
│      ├─ exploration_hints: {                    │
│      │   ├─ search_terms: [...]                 │
│      │   ├─ resource_paths: [...]               │
│      │   └─ tools_to_expand: [...]              │
│      │ }                                         │
│      └─ rationale                               │
│    }                                             │
└──────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────┐
│         Process Refinement Verdict               │
│                                                  │
│  If verdict == "approve":                        │
│    ├─ If ExecutionPlan && iter >= min_iter:     │
│    │   └─ Return plan → Agent executes tool     │
│    └─ If Finalization:                          │
│        └─ Return response → User receives       │
│                                                  │
│  If verdict == "refine":                         │
│    ├─ Update current_goal with refined_goal     │
│    ├─ Apply exploration hints:                  │
│    │   ├─ Search for suggested terms            │
│    │   ├─ Load suggested resources              │
│    │   └─ Expand suggested tools                │
│    └─ Continue to next iteration                │
└──────────────────────────────────────────────────┘
    ↓
Check iteration limits → If max_iter: return last_planned_action
Otherwise: Loop back to Context Discovery
```

## 5. Key Technical Points

### 5.1 Multi-Phase Orchestration Strategy

#### 5.1.1 Phase Separation of Concerns
- **Phase 1 (Discovery)**: Explores broadly without filtering - finds all potentially relevant context
- **Phase 2 (Refinement)**: Filters aggressively - removes noise and prioritizes important context
- **Phase 3 (Decision)**: Acts decisively - always chooses execute or finalize, never ambiguous
- **Phase 4 (Analysis)**: Thinks strategically - catches errors and evolves goals with prerequisites

#### 5.1.2 Context Loop vs Iteration Loop
- **Inner Context Loop**: Repeatedly discovers and refines until context is adequate
- **Outer Iteration Loop**: Tests decisions and refines goals across multiple attempts
- Allows multiple discovery/refinement cycles per action decision
- Ensures comprehensive context before committing to execution

#### 5.1.3 Structured Output with Validation
- **JSON Schema**: Ensures consistency of LLM output format for all phases
- **Pydantic Models**: Type-safe data structures for each phase output
- **Error Handling**: Graceful handling of malformed LLM outputs
- **Template Variables**: Rich context passed to each template

### 5.2 Dynamic Tool Management

#### 5.2.1 Expansion/Collapse Mechanism
- **Collapsed State**: Only tool name and description visible (minimal context)
- **Expanded State**: Full schema with parameters, types, descriptions (detailed context)
- **Discovery Phase**: Identifies tools to expand based on relevance
- **Refinement Phase**: Collapses tools no longer needed
- **Benefit**: Significantly reduces LLM context size while maintaining necessary detail

#### 5.2.2 Tool Context Requirements
- Tools declare `require_context` flag
- If true, Agent builds current context and injects as parameter
- Context built from ResourceContext's current workspace state
- Allows tools to make context-aware decisions

### 5.3 Goal Evolution Strategy

#### 5.3.1 Living Goal Document
Goals accumulate discovered knowledge across iterations:
```
Iteration 1: "Create protagonist named 余归"
Iteration 2: "Create protagonist named 余归 (Prerequisites: Check if character aspect exists. Context: Need aspect configuration and file paths.)"
Iteration 3: "Create protagonist named 余归 (Prerequisites: 1. Verify character aspect, 2. Check for existing character with same name...)"
```

#### 5.3.2 Exploration Hints
Refinement analysis provides specific guidance:
- **search_terms**: What to search for in next iteration
- **resource_paths**: Specific resources to load
- **tools_to_expand**: Additional tools that might be needed
- **focus_areas**: Conceptual areas to explore

### 5.4 Graceful Degradation

#### 5.4.1 Last Action Tracking
- System maintains `last_planned_action` throughout execution
- Updated with every action decision (approved or refined)
- Serves as fallback when max iterations reached
- Ensures meaningful response in all scenarios

#### 5.4.2 Min/Max Iteration Control
- **min_iter**: Prevents premature execution, ensures thorough analysis
- **max_iter**: Prevents infinite loops, guarantees termination
- Balance between thoroughness and practicality
- Configurable per Agent instance

### 5.5 LLM Integration Strategy

#### 5.5.1 Template System
- **Jinja2 Templates**: Flexible prompt construction for each phase
- **Multi-language Support**: `zh/` and `en/` template directories
- **Context Injection**: Dynamic information passing with rich variables
- **Structured Prompts**: Clear instructions for each phase's responsibility

#### 5.5.2 Template Context Variables
Different variables for different phases:
- **Discovery**: user_request, goal, workspace_segments, search_history, expanded_tools, collapsed_tools
- **Refinement**: goal, discovery_analysis, workspace_segments, expanded_tools, collapsed_tools
- **Decision**: user_request, goal, completed_steps, workspace_segments, available_tools
- **Analysis**: user_request, original_goal, action_decision, completed_steps, workspace_segments, available_tools

### 5.6 Resource Management Strategy

#### 5.6.1 Selective Loading
- **Lazy Loading**: Resources loaded only when discovered as relevant
- **Incremental Loading**: Context expands gradually through discovery cycles
- **Search-Based Discovery**: Semantic search identifies relevant resources
- **Direct Queries**: Load specific resources by URI when known
- **Memory Optimization**: Avoid loading unnecessary large datasets

#### 5.6.2 Property Filtering
- **Include/Exclude Lists**: Control which properties are visible
- **Dynamic Filtering**: Adjusted through refinement phase
- **Relationship Filtering**: Filter relations to excluded resources
- **Reduces Context Size**: Only relevant properties consume LLM context

#### 5.6.3 Segment Sorting
- **Priority-Based Ordering**: Most relevant resources first
- **Refinement Phase Control**: LLM determines optimal ordering
- **Improves LLM Attention**: Important context appears early
- **Maintains Relationships**: Parent-child structure preserved

### 5.7 Error Handling and Fault Tolerance

#### 5.7.1 Tool Execution Errors
- **Type-safe Errors**: `ToolResult` vs `ToolError` distinction
- **Error Recovery**: System continues with finalization on errors
- **User Notification**: Clear error messages through AgentChannel
- **StepOutcome Tracking**: Complete error history maintained

#### 5.7.2 LLM Response Errors
- **Format Validation**: JSON Schema validation for all phases
- **Malformed Decision Handling**: Fallback to finalization on invalid decisions
- **Template Robustness**: Clear instructions reduce format errors
- **Graceful Degradation**: System never crashes, always provides response

#### 5.7.3 Iteration Limit Handling
- **Max Iterations**: Prevents infinite refinement loops
- **Last Action Fallback**: Returns best planned action when limit hit
- **Meaningful Responses**: Even incomplete scenarios provide useful feedback
- **Status Indicators**: Finalization includes status (success/failed/abandoned)

## 6. Configuration and Extension

### 6.1 Agent Factory Pattern

```python
def create_agent(
    tools: dict[str, SchematicTool],
    resource_repo: ResourceRepository,
    template_env: TemplateEnvironment,
    chat_llm: ChatLLM,
    channel: AgentChannel,
    max_iterations: int = 10,
    min_iterations: int | None = 2
) -> Agent:
    """Create a configured Agent instance with multi-phase orchestration."""
    
    return Agent(
        tools=tools,
        resource_repo=resource_repo,
        template_env=template_env,
        chat_llm=chat_llm,
        channel=channel,
        max_iterations=max_iterations,
        min_iterations=min_iterations
    )
```

**Key Parameters:**
- `max_iterations`: Maximum orchestration loops (default: 10)
- `min_iterations`: Minimum loops before allowing execution (default: 2)
- OrchestrationLoop created fresh for each request
- ResourceContext shared across Agent lifetime
```

### 6.2 Tool Registration and Discovery

#### 6.2.1 SchematicTool Implementation Example
```python
class CreateAspectTool(SchematicTool):
    @property
    def name(self) -> str:
        return "create_aspect"
    
    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"}
            },
            "required": ["name", "description"]
        }
    
    async def call(self, runtime: ToolRuntime, **kwargs) -> ToolOutput:
        # Tool logic implementation
        ...
```

#### 6.2.2 Tool Integration Patterns
- **Modularization**: Each tool as independent module
- **Registration Mechanism**: Automatic tool discovery and registration
- **Dependency Injection**: Automatic injection of tool dependencies

### 6.3 Template Customization

#### 6.3.1 Directory Structure
```
templates/
├── zh/                     # Chinese templates
│   ├── strategic_context_orchestrator.jinja2
│   └── ...
└── en/                     # English templates  
    ├── strategic_context_orchestrator.jinja2
    └── ...
```

#### 6.3.2 Template Variables
- **goal**: User goal
- **completed_steps**: Completed steps
- **pending_steps**: Pending steps  
- **available_tools**: Available tools
- **context**: Current resource context

## 7. Performance and Monitoring

### 7.1 Performance Optimization Strategies

#### 7.1.1 Context Management Optimization
- **Segment Caching**: Avoid rebuilding identical resource segments
- **Incremental Updates**: Only update changed parts
- **Size Limits**: Control total context size

#### 7.1.2 LLM Call Optimization
- **Batch Requests**: Batch processing when possible
- **Caching Strategy**: Cache results for similar requests
- **Timeout Control**: Avoid long waits

### 7.2 Monitoring and Debugging

#### 7.2.1 Execution Trace
- **StepOutcome Records**: Detailed information for each execution step
- **Timestamps**: Precise execution time recording
- **State Tracking**: Complete history of state changes

#### 7.2.2 Logging System
- **Structured Logs**: Log format convenient for analysis
- **Level Control**: debug/info/warning/error
- **Context Information**: Rich contextual logs

## 8. Testing Strategy

### 8.1 Unit Testing
- **Component Isolation**: Independent testing of each component
- **Mock Strategy**: Simulation of LLM and external dependencies
- **Boundary Testing**: Exception cases and boundary conditions

### 8.2 Integration Testing
- **End-to-End Testing**: Complete goal pursuit processes
- **Multi-tool Collaboration**: Tool combinations in complex scenarios
- **Performance Benchmarks**: Execution time and resource consumption

### 8.3 Testing Tools
- **MockTool**: Mock tools for testing
- **TestChannel**: Communication channels for test environments
- **ContextValidator**: Context state validators

## 9. Deployment and Operations

### 9.1 Dependency Management
- **Core Dependencies**: Pydantic, Jinja2, AsyncIO
- **LLM Dependencies**: Support for multiple LLM providers
- **Optional Dependencies**: Additional dependencies for specific tools

### 9.2 Configuration Management
- **Environment Variables**: Environment variable management for sensitive configurations
- **Configuration Files**: Structured configuration file support
- **Dynamic Configuration**: Runtime configuration updates

### 9.3 Monitoring Metrics
- **Execution Success Rate**: Success rate of goal pursuit
- **Average Execution Time**: Performance monitoring metrics
- **Tool Usage Statistics**: Tool usage frequency and success rates
- **Error Rate**: Frequency of various error types

## 10. Future Development

### 10.1 Architecture Evolution
- **Distributed Execution**: Support for multi-Agent collaboration
- **Streaming Processing**: Real-time streaming result returns
- **Plugin System**: More flexible plugin architecture

### 10.2 AI Capability Enhancement
- **Multi-modal Support**: Comprehensive processing of text, images, and audio
- **Memory System**: Long-term memory and learning capabilities
- **Adaptive Learning**: Strategy optimization based on historical execution

### 10.3 Ecosystem
- **Tool Marketplace**: Community-contributed tool ecosystem
- **Template Library**: Rich predefined template collections
- **Best Practices**: Industry application best practice libraries

---

## Appendix

### A. Data Structure Definitions

#### A.1 Core Data Structures
```python
@dataclass(frozen=True)
class StepDefinition:
    reason: str
    tool: str  
    parameters: dict

@dataclass(frozen=True) 
class StepOutcome:
    action: StepDefinition
    status: StepStatus
    results: list[str]
    error_message: str | None
    started_at: datetime
    completed_at: datetime
    triggered_actions: list[dict[str, str]]
    backlog_items: list[str]
    progress: dict[str, list[str]]

@dataclass(frozen=True)
class OrchestrationExecutionPlan:
    reason: str
    tool: str
    params: dict
    future_steps: list[str]

@dataclass(frozen=True)
class OrchestrationFinalization:
    reason: str
    response: str
    status: str
```

#### A.2 Tool Output Types
```python
class ToolResult(ToolOutputBase):
    type: Literal[ToolOutputType.OUTPUT] = ToolOutputType.OUTPUT
    result: str

class ToolError(ToolOutputBase):
    type: Literal[ToolOutputType.ERROR] = ToolOutputType.ERROR
    error_message: str
```

### B. Configuration Examples

#### B.1 Agent Creation Configuration
```python
# Create tools dictionary
tools = {
    "create_aspect": CreateAspectTool(resource_repo),
    "fetch_resource": FetchResourceTool(resource_repo),
    "search_resources": SearchResourceTool(resource_repo),
    "generate_content": GenerateContentTool(template_env, chat_llm)
}

# Create Agent
agent = create_agent(
    tools=tools,
    resource_repo=resource_repo,
    template_env=template_env,
    chat_llm=chat_llm,
    channel=session_channel,
    max_iterations=10,
    min_iterations=2
)
```

#### B.2 LLM Configuration Example
```python
chat_llm_config = {
    "provider": "azure_openai",
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "timeout": 30
}
```

### C. Common Issues and Solutions

#### C.1 Performance Issues
**Problem**: Agent execution too slow
**Solutions**: 
- Check context size, reduce unnecessary resource loading
- Optimize LLM call frequency and payload size
- Use tool expansion/contraction mechanism to reduce tool schema transmission

#### C.2 LLM Output Format Errors
**Problem**: LLM returns invalid JSON
**Solutions**:
- Add JSON Schema constraints
- Improve template prompts
- Implement retry mechanisms

#### C.3 Tool Execution Failures
**Problem**: Frequent tool call failures
**Solutions**:
- Check tool input parameter validation
- Add error handling and user-friendly error messages
- Implement tool health check mechanisms

This design document comprehensively describes the architecture, design principles, implementation details, and best practices of the NovelRAG Agent System, providing detailed technical guidance for system development, maintenance, and extension.