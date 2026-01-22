# NovelRAG Agent System Design Document

## 1. System Overview

### 1.1 Project Background
The NovelRAG Agent System is a context-driven intelligent execution framework designed to achieve automated goal pursuit through dynamic context management and multi-phase strategic orchestration. The system uses a sophisticated `ActionDetermineLoop` architecture with phases for context discovery, refinement, action decision-making, and refinement analysis.

### 1.2 Core Design Principles
- **Context-Driven**: Decision-making based on dynamically discovered and refined resource contexts
- **Multi-Phase Orchestration**: Strategic decisions through distinct phases with dedicated LLM templates
- **Iterative Refinement**: Context and goals evolve through multiple iterations to achieve optimal execution
- **Tool Atomization**: Each tool is an independent, composable functional unit
- **Dynamic Tool Management**: Tools expand/collapse based on relevance to reduce context overhead
- **State Transparency**: All execution states are traceable and debuggable

### 1.3 Architectural Advantages
- **Enhanced Context Discovery**: Dedicated phase for aggressive context exploration
- **Intelligent Filtering**: Separate refinement phase ensures only relevant context is retained
- **Strategic Oversight**: Refinement analysis phase catches planning errors and missing prerequisites
- **Goal Evolution**: Goals naturally evolve to incorporate discovered requirements
- **Adaptive Tool Management**: Dynamic expansion/collapse reduces LLM context size

## 2. System Architecture

### 2.1 Overall Architecture Diagram

```
+-----------------------------------------------------------------+
|                    NovelRAG Agent System                        |
+-----------------------------------------------------------------+
|  +-------------+  +-----------------+  +-----------------+      |
|  |GoalExecutor |  |ActionDetermine- |  | ResourceContext |      |
|  |             |--|     Loop        |--|                 |      |
|  | - handle_   |  | - determine_    |  | - refine        |      |
|  |   goal      |  |   action        |  | - build_context |      |
|  | - _execute_ |  |                 |  |                 |      |
|  |   tool      |  |                 |  |                 |      |
|  +-------------+  +-----------------+  +-----------------+      |
|         |                    |                    |              |
|  +-------------+  +-----------------+  +-----------------+      |
|  |AgentTool-   |  |   Tool System   |  | Resource System |      |
|  |   Runtime   |  |                 |  |                 |      |
|  | - debug     |  | - SchematicTool |  | - Repository    |      |
|  | - message   |  | - ToolRuntime   |  | - Aspects       |      |
|  | - progress  |  | - ToolOutput    |  | - Elements      |      |
|  +-------------+  +-----------------+  +-----------------+      |
|         |                    |                    |              |
|  +-------------+  +-----------------+  +-----------------+      |
|  | AgentChannel|  | Template System |  |   LLM System    |      |
|  |             |  |                 |  |                 |      |
|  | - info      |  | - Jinja2        |  | - ChatLLM       |      |
|  | - error     |  | - Structured    |  | - JSON Schema   |      |
|  | - confirm   |  |   Output        |  | - Validation    |      |
|  +-------------+  +-----------------+  +-----------------+      |
+-----------------------------------------------------------------+
```

### 2.2 Core Component Relationships

#### 2.2.1 GoalExecutor
- **Responsibility**: Main controller for goal pursuit
- **Location**: `novelrag/agenturn/agent.py`
- **Core Method**: `handle_goal()` - Receives goals and coordinates execution
- **Execution Flow**: 
  1. Create new PursuitProgress
  2. Loop call ActionDeterminer.determine_action()
  3. Execute recommended tools via `_execute_tool()`
  4. Update execution state
  5. Return PursuitOutcome

#### 2.2.2 ActionDetermineLoop
- **Responsibility**: Multi-phase strategic decision-making and execution orchestration
- **Location**: `novelrag/resource_agent/action_determine_loop.py`
- **Core Method**: `determine_action()` - Orchestrates phased decision process
- **Implements**: `ActionDeterminer` protocol from `novelrag/agenturn/pursuit.py`
- **Return Types**:
  - `OperationPlan`: Execute specific tool
  - `Resolution`: Complete goal pursuit

#### 2.2.3 ResourceContext
- **Responsibility**: Dynamic construction and management of execution context
- **Location**: `novelrag/resource_agent/workspace.py`
- **Core Functions**:
  - Selective loading of resource aspects
  - Filter and sort resource segments
  - Search history management
  - Context refinement

## 3. Detailed Design

### 3.1 ActionDetermineLoop Design

#### 3.1.1 Core Algorithm

The orchestration loop implements a multi-phase architecture:

```python
async def determine_action(self, beliefs, pursuit_progress, available_tools):
    iter_num = 0
    goal = pursuit_progress.goal
    pursuit_assessment = await self.assessor.assess_progress(pursuit_progress, beliefs)
    last_planned_action = Resolution(...)  # Default fallback
    
    while True:
        # Context Discovery & Refinement Loop
        while True:
            iter_num += 1
            discovery_plan = await self._discover_and_expand_context(
                goal, pursuit_assessment, available_tools
            )
            await self._apply_discovery_plan(discovery_plan)
            
            if not discovery_plan.refinement_needed:
                break
            if self.max_iter and iter_num >= self.max_iter:
                break
            
            refinement_plan = await self._filter_and_refine_context(
                goal, pursuit_assessment, available_tools, discovery_plan.discovery_analysis
            )
            await self._apply_refinement_plan(refinement_plan)
        
        # Action Decision
        action_decision = await self._make_action_decision(
            goal, pursuit_assessment, pursuit_progress.executed_steps, available_tools
        )
        planned_action = self._convert_to_orchestration_action(action_decision)
        last_planned_action = planned_action
        
        # Refinement Analysis
        refinement_decision = await self._analyze_and_refine(
            goal, pursuit_assessment, action_decision, pursuit_progress.executed_steps, available_tools
        )
        
        # Process verdict
        if refinement_decision.verdict == "approve":
            return planned_action
        else:
            # Update pursuit_assessment with refinements
            ...
        
        if self.max_iter and iter_num >= self.max_iter:
            break
    
    return last_planned_action
```

#### 3.1.2 Data Structures

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

### 3.2 Tool System Design

#### 3.2.1 Tool Interface Hierarchy

```python
# Base tool abstraction (novelrag/agenturn/tool/)
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

#### 3.2.2 ToolRuntime Interface

```python
class ToolRuntime(ABC):
    async def debug(self, content: str): ...
    async def message(self, content: str): ...
    async def warning(self, content: str): ...
    async def error(self, content: str): ...
    async def confirmation(self, prompt: str) -> bool: ...
    async def user_input(self, prompt: str) -> str: ...
    async def progress(self, key: str, value: str, description: str | None = None): ...
    async def backlog(self, content: dict, priority: str | None = None): ...
```

### 3.3 Communication System Design

#### 3.3.1 AgentChannel Protocol

```python
class AgentChannel(Protocol):
    async def send_message(self, content: str, level: AgentMessageLevel) -> None: ...
    async def confirm(self, prompt: str) -> bool: ...
    async def request(self, prompt: str) -> str: ...
```

## 4. Execution Flow

### 4.1 Goal Pursuit Flow

```
User Request -> RequestHandler.handle_request()
    |
GoalTranslator.translate() -> Goal
    |
GoalExecutor.handle_goal()
    |
+------------------------------------------------------------+
|              Main Execution Loop                           |
|                                                            |
|  ActionDeterminer.determine_action()                       |
|      |                                                     |
|  +------------------------------------------------------+  |
|  |       Context Discovery & Refinement Loop            |  |
|  |                                                      |  |
|  |  1. _discover_and_expand_context()                   |  |
|  |  2. _apply_discovery_plan()                          |  |
|  |  3. If refinement needed:                            |  |
|  |     - _filter_and_refine_context()                   |  |
|  |     - _apply_refinement_plan()                       |  |
|  +------------------------------------------------------+  |
|      |                                                     |
|  _make_action_decision()                                   |
|      |                                                     |
|  _analyze_and_refine()                                     |
|      |                                                     |
|  Return OperationPlan or Resolution                        |
+------------------------------------------------------------+
    |
If OperationPlan:
    +- GoalExecutor._execute_tool()
    +- Update executed_steps
    +- Continue loop
    
If Resolution:
    +- Return PursuitOutcome to user
```

## 5. Key Technical Points

### 5.1 Dynamic Tool Management

- **Collapsed State**: Only tool name and description visible (minimal context)
- **Expanded State**: Full schema with parameters, types, descriptions
- **Discovery Phase**: Identifies tools to expand based on relevance
- **Refinement Phase**: Collapses tools no longer needed

### 5.2 Pursuit Assessment

The `PursuitAssessor` evaluates progress toward goals:

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

### 5.3 Error Handling

- **ToolResult vs ToolError**: Type-safe distinction for tool outputs
- **OperationOutcome**: Tracks execution status, results, and error messages
- **Graceful Degradation**: System returns `last_planned_action` when max iterations reached

## 6. Configuration and Extension

### 6.1 Factory Pattern

```python
# novelrag/resource_agent/__init__.py
def create_executor(
    resource_repo,
    channel,
    chat_llm,
    beliefs: list[str] | None = None,
    lang: str | None = None,
) -> GoalExecutor:
    """Create a GoalExecutor configured for resource operations."""
    context = ResourceContext(resource_repo, template_env, chat_llm)
    orchestrator = ActionDetermineLoop(context, chat_llm, template_lang=lang or "en")
    
    tools = {
        "ResourceFetchTool": ResourceFetchTool(resource_repo),
        "ResourceSearchTool": ResourceSearchTool(resource_repo),
        "AspectCreateTool": AspectCreateTool(resource_repo, template_env, chat_llm),
        "ResourceRelationWriteTool": ResourceRelationWriteTool(resource_repo, template_env, chat_llm),
    }
    
    return GoalExecutor(
        beliefs=beliefs or [],
        tools=tools,
        determiner=orchestrator,
        channel=channel,
    )
```

### 6.2 Request Handler and Autonomous Agent

```python
# Create request-driven handler
request_handler = executor.create_request_handler(goal_translator)
response = await request_handler.handle_request("Find the protagonist")

# Create autonomous agent
autonomous_agent = executor.create_autonomous_agent(goal_decider)
outcome = await autonomous_agent.pursue_next_goal()
```

## 7. Package Structure

```
novelrag/
+-- agenturn/                    # Generic agent framework
|   +-- agent.py                 # GoalExecutor, AgentToolRuntime, RequestHandler, AutonomousAgent
|   +-- channel.py               # AgentChannel protocol
|   +-- goal.py                  # Goal, GoalTranslator, LLMGoalTranslator
|   +-- pursuit.py               # PursuitProgress, PursuitOutcome, ActionDeterminer, PursuitAssessor
|   +-- step.py                  # OperationPlan, OperationOutcome, Resolution, StepStatus
|   +-- tool/                    # Tool abstractions
|       +-- schematic.py         # SchematicTool, BaseTool
|       +-- types.py             # ToolRuntime, ToolResult, ToolError
|
+-- resource_agent/              # Resource-specific agent implementation
|   +-- action_determine_loop.py # ActionDetermineLoop (implements ActionDeterminer)
|   +-- workspace.py             # ResourceContext, ContextWorkspace
|   +-- tool/                    # Resource tools
|   |   +-- fetch.py             # ResourceFetchTool
|   |   +-- search.py            # ResourceSearchTool
|   |   +-- aspect.py            # AspectCreateTool
|   |   +-- relation.py          # ResourceRelationWriteTool
|   +-- templates/               # LLM prompt templates
|
+-- resource/                    # Resource system
|   +-- repository.py            # ResourceRepository, LanceDBResourceRepository
|   +-- aspect.py                # ResourceAspect
|   +-- element.py               # Element, DirectiveElement
|   +-- vector.py                # LanceDBStore
|
+-- cli/                         # Command-line interface
    +-- session.py               # Session, SessionChannel
    +-- shell.py                 # NovelShell
    +-- handler/                 # Command handlers
        +-- builtin/             # Built-in handlers (agent, quit, undo, redo)
```
