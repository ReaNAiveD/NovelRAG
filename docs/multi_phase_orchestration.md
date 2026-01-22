# Multi-Phase Orchestration Architecture

## Overview

The orchestration system uses a multi-phase decision architecture that handles goal pursuit through iterative context refinement, action planning, and execution with built-in refinement capabilities.

## Architecture

The system operates through multiple distinct phases within each iteration:

### Phase 1: Context Discovery
**Template**: `context_discovery.jinja2`

**Responsibility**: Discover and expand context relevant to the current goal.

**Returns** (`DiscoveryPlan`):
- `discovery_analysis` - Analysis of what was discovered
- `search_queries` - Terms to search for in the resource repository
- `query_resources` - Specific resource URIs to load
- `expand_tools` - Tools to expand for detailed schemas

### Phase 2: Context Refinement  
**Template**: `context_relevance.jinja2`

**Responsibility**: Filter and refine the discovered context for relevance.

**Returns** (`RefinementPlan`):
- `exclude_resources` - Resources to remove from context
- `exclude_properties` - Specific properties to exclude
- `collapse_tools` - Tools to collapse back to summary form
- `sorted_segments` - Priority ordering of context segments

### Phase 3: Action Decision
**Template**: `action_decision.jinja2`

**Responsibility**: Make decisive choice between executing a tool or finalizing.

**Returns** (`ActionDecision`):
- `situation_analysis` - Comprehensive assessment of current state
- `decision_type` - Either "execute" or "finalize"
- `execution` - Tool and parameters if executing
- `finalization` - Response and status if finalizing
- `context_verification` - Detailed verification of prerequisites and parameters

### Phase 4: Refinement Analysis
**Template**: `refinement_analysis.jinja2`

**Responsibility**: Analyze the action decision for quality and determine if goal refinement is needed.

**Returns** (`RefinementDecision`):
- `analysis` - Quality assessment and discovered issues
- `verdict` - Either "approve" or "refine"
- `approval` - Confidence and notes if approving
- `refinement` - Enhanced assessment and exploration hints if refining

## Execution Flow

```
Goal -> PursuitProgress
    |
[Main Loop]
    |
    +--[Context Loop]
    |  +-> Phase 1: Context Discovery
    |  +-> Apply discovery (search, query resources, expand tools)
    |  +-> Check if refinement needed
    |  +-> Phase 2: Context Refinement (if needed)
    |  +-> Apply refinement (exclude, sort, collapse tools)
    |
    +-> Phase 3: Action Decision (execute/finalize)
    +-> Store as last_planned_action
    +-> Phase 4: Refinement Analysis (approve/refine)
    |
    +-> If APPROVE + EXECUTE: Return OperationPlan
    +-> If APPROVE + FINALIZE: Return Resolution
    +-> If REFINE: Update PursuitAssessment & continue loop
    +-> If MAX_ITER: Return last_planned_action
```

## Key Features

### 1. Iterative Context Refinement
The system can perform multiple rounds of context discovery and refinement within a single iteration.

### 2. Pursuit Assessment Evolution
The `PursuitAssessment` evolves through iterations to incorporate discovered prerequisites:
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

### 3. Flexible Tool Management
Tools can be dynamically expanded and collapsed:
- **Expanded**: Full schema visible for execution planning
- **Collapsed**: Summary only to reduce context size

### 4. Last Action Tracking
The system tracks `last_planned_action` throughout execution as a fallback when max iterations reached.

### 5. Min/Max Iteration Control
- **min_iter**: Minimum iterations before allowing execution
- **max_iter**: Maximum iterations to prevent infinite loops

## Data Flow

### Context Management
```python
ResourceContext
    +-- queried_resources: Loaded resource URIs
    +-- excluded_resources: Excluded URIs  
    +-- property_filters: Property inclusion/exclusion rules
    +-- search_history: Previous search queries and results
    +-- workspace_segments: Sorted, filtered context segments
```

### Tool State Management
```python
ActionDetermineLoop
    +-- expanded_tools: Set of tool names with full schemas
    +-- context: ResourceContext instance
    +-- min_iter: Minimum iteration requirement
    +-- max_iter: Maximum iteration limit
```

## Edge Case Handling

| Scenario | System Response |
|----------|-----------------|
| Direct answer available | Action Decision finalizes immediately with success |
| Tool ready with all prerequisites | Executes after min_iter requirement met |
| Missing prerequisites discovered | Refinement Analysis updates PursuitAssessment |
| No suitable tool exists | Action Decision finalizes with detailed explanation |
| Max iterations reached | Returns last planned action (OperationPlan or Resolution) |

## Configuration

### ActionDetermineLoop Parameters
- **context**: `ResourceContext` instance for resource management
- **chat_llm**: `ChatLLM` for LLM interactions
- **template_lang**: Language for templates (default: "en")
- **max_iter**: Maximum iterations (default: 5)
- **min_iter**: Minimum iterations before execution (default: 0)

## Implementation References

- **Core Loop**: `ActionDetermineLoop.determine_action()` in `novelrag/resource_agent/action_determine_loop.py`
- **Context Discovery**: `ActionDetermineLoop._discover_and_expand_context()`
- **Context Refinement**: `ActionDetermineLoop._filter_and_refine_context()`
- **Action Decision**: `ActionDetermineLoop._make_action_decision()`
- **Refinement Analysis**: `ActionDetermineLoop._analyze_and_refine()`
