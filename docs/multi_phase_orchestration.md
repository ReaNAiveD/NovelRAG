# Multi-Phase Orchestration Architecture

## Overview

The orchestration system uses a sophisticated multi-phase decision architecture that handles goal pursuit through iterative context refinement, action planning, and execution with built-in refinement capabilities.

## Architecture

The system operates through multiple distinct phases within each iteration:

### Phase 1: Context Discovery
**Template**: `context_discovery.jinja2`

**Responsibility**: Discover and expand context relevant to the current goal.

**Returns** ([`DiscoveryPlan`](../novelrag/agent/orchestrate.py)):
- `discovery_analysis` - Analysis of what was discovered
- `search_queries` - Terms to search for in the resource repository
- `query_resources` - Specific resource URIs to load
- `expand_tools` - Tools to expand for detailed schemas

**Key Principle**: Aggressively explore semantic relationships and gather comprehensive context.

### Phase 2: Context Refinement  
**Template**: `refine_context_for_execution.jinja2`

**Responsibility**: Filter and refine the discovered context for relevance.

**Returns** ([`RefinementPlan`](../novelrag/agent/orchestrate.py)):
- `exclude_resources` - Resources to remove from context
- `exclude_properties` - Specific properties to exclude
- `collapse_tools` - Tools to collapse back to summary form
- `sorted_segments` - Priority ordering of context segments

**Key Principle**: Focus context on what's most relevant for the current goal.

### Phase 3: Action Decision
**Template**: `action_decision.jinja2`

**Responsibility**: Make decisive choice between executing a tool or finalizing.

**Returns** ([`ActionDecision`](../novelrag/agent/orchestrate.py)):
- `situation_analysis` - Comprehensive assessment of current state
- `decision_type` - Either "execute" or "finalize"
- `execution` - Tool and parameters if executing
- `finalization` - Response and status if finalizing
- `context_verification` - Detailed verification of prerequisites and parameters

**Key Principle**: Always be decisive - no ambiguous "need more context" state.

### Phase 4: Refinement Analysis
**Template**: `refinement_analysis.jinja2`

**Responsibility**: Analyze the action decision for quality and determine if goal refinement is needed.

**Returns** ([`RefinementDecision`](../novelrag/agent/orchestrate.py)):
- `analysis` - Quality assessment and discovered issues
- `verdict` - Either "approve" or "refine"
- `approval` - Confidence and notes if approving
- `refinement` - Enhanced goal and exploration hints if refining

**Key Principle**: Strategic oversight to catch missed prerequisites and improve goal clarity.

## Execution Flow

```
User Request → Initial Goal
    ↓
[Main Iteration Loop]
    │
    ├─[Context Loop]
    │  ├→ Phase 1: Context Discovery
    │  ├→ Apply discovery (search, query resources, expand tools)
    │  ├→ Check if refinement needed
    │  ├→ Phase 2: Context Refinement (if needed)
    │  └→ Apply refinement (exclude, sort, collapse tools)
    │
    ├→ Phase 3: Action Decision (execute/finalize)
    ├→ Store as last_planned_action
    ├→ Phase 4: Refinement Analysis (approve/refine)
    │
    ├→ If APPROVE + EXECUTE + min_iter_met: Execute tool & return
    ├→ If APPROVE + FINALIZE: Return response to user
    ├→ If REFINE: 
    │   ├→ Update goal with discovered requirements
    │   └→ Apply exploration hints & continue loop
    └→ If MAX_ITER: Return last_planned_action
```

## Key Features

### 1. Iterative Context Refinement
The system can perform multiple rounds of context discovery and refinement within a single iteration:
- Discovery identifies what might be relevant
- Refinement filters to what is actually needed
- Process repeats until context is adequate

### 2. Goal Evolution
Goals evolve through iterations to incorporate discovered prerequisites:
```
Iteration 1: "Find protagonist's weapon"
Iteration 2: "Find protagonist's weapon (Prerequisites: Identify protagonist from /character resources first. Context: equipment structure needed.)"
```

### 3. Flexible Tool Management
Tools can be dynamically expanded and collapsed:
- **Expanded**: Full schema visible for execution planning
- **Collapsed**: Summary only to reduce context size
- System manages which tools need expansion based on analysis

### 4. Last Action Tracking
The system tracks `last_planned_action` throughout execution:
- Initialized with a safe default finalization
- Updated with each action decision (whether approved or refined)
- Serves as fallback when max iterations reached
- Ensures meaningful response in all scenarios

### 5. Min/Max Iteration Control
- **min_iter**: Minimum iterations before allowing execution (ensures thorough analysis)
- **max_iter**: Maximum iterations to prevent infinite loops
- Balances thoroughness with practical limits

## Data Flow

### Context Management
```python
ResourceContext
    ├─ queried_resources: Set of loaded resource URIs
    ├─ excluded_resources: Set of excluded URIs  
    ├─ property_filters: Property inclusion/exclusion rules
    ├─ search_history: Previous search queries and results
    └─ workspace_segments: Sorted, filtered context segments
```

### Tool State Management
```python
OrchestrationLoop
    ├─ expanded_tools: Set of tool names with full schemas
    ├─ context: ResourceContext instance
    ├─ min_iter: Minimum iteration requirement
    └─ max_iter: Maximum iteration limit
```

## Edge Case Handling

| Scenario | System Response |
|----------|-----------------|
| Direct answer available | Action Decision finalizes immediately with success |
| Tool ready with all prerequisites | Executes after min_iter requirement met |
| Missing prerequisites discovered | Refinement Analysis updates goal with requirements |
| Wrong tool initially selected | Refinement suggests alternatives via exploration hints |
| No suitable tool exists | Action Decision finalizes with detailed explanation |
| Context discovery finds nothing | Proceeds to action decision with available context |
| Max iterations reached | Returns last planned action (execute or finalize) |
| Circular dependencies detected | Finalizes with incomplete status and explanation |

## Configuration

### OrchestrationLoop Parameters
- **context**: [`ResourceContext`](../novelrag/agent/workspace.py) instance for resource management
- **template_env**: [`TemplateEnvironment`](../novelrag/template.py) for prompt generation
- **chat_llm**: [`ChatLLM`](../novelrag/llm/types.py) for LLM interactions
- **max_iter**: Maximum iterations (default: 5, None for unlimited)
- **min_iter**: Minimum iterations before execution (default: None)

## Template Integration

Each phase uses sophisticated Jinja2 templates that receive:
- Current goal (potentially refined from previous iterations)
- Workspace segments (filtered, sorted context)
- Tool information (expanded or collapsed)
- Execution history
- Search history
- Discovery analysis from previous phases

## Design Principles

1. **Separation of Concerns**: Each phase has a single, well-defined responsibility
2. **Progressive Enhancement**: Context and goals improve through iterations
3. **Always Decisive**: No ambiguous states - always execute or finalize
4. **Fail Gracefully**: Clear communication even when goals cannot be achieved
5. **Strategic Depth**: Multiple analysis layers catch different types of issues
6. **Resource Efficiency**: Smart context management prevents information overload
7. **User-Centric**: Clear responses that explain decisions and outcomes

## Implementation References

- **Core Loop**: [`OrchestrationLoop.execution_advance()`](../novelrag/agent/orchestrate.py)
- **Context Discovery**: [`OrchestrationLoop._discover_and_expand_context()`](../novelrag/agent/orchestrate.py)
- **Context Refinement**: [`OrchestrationLoop._filter_and_refine_context()`](../novelrag/agent/orchestrate.py)
- **Action Decision**: [`OrchestrationLoop._make_action_decision()`](../novelrag/agent/orchestrate.py)
- **Refinement Analysis**: [`OrchestrationLoop._analyze_and_refine()`](../novelrag/agent/orchestrate.py)

## Future Enhancements

- **Context Caching**: Reuse context across similar goals
- **Learning Patterns**: Identify common prerequisite chains for faster resolution
- **Parallel Discovery**: Explore multiple context paths simultaneously
- **Confidence Tracking**: Maintain confidence scores across all phases
- **Goal Templates**: Pre-defined goal patterns for common requests
- **Adaptive Iteration Limits**: Adjust min/max based on goal complexity
