# NovelRAG Agent System Design Document

## 1. System Overview

### 1.1 Project Background
The NovelRAG Agent System is a context-driven intelligent execution framework designed to achieve automated goal pursuit through dynamic context management and strategic orchestration. The system transitions from traditional step-based planning approaches to a more flexible OrchestrationLoop architecture.

### 1.2 Core Design Principles
- **Context-Driven**: Decision-making based on dynamically constructed and refined resource contexts
- **Iterative Orchestration**: Strategic decisions through LLM templates and structured output
- **Tool Atomization**: Each tool is an independent, composable functional unit
- **State Transparency**: All execution states are traceable and debuggable

### 1.3 Architectural Advantages
Compared to traditional step-based planning approaches, the new architecture offers the following advantages:
- Enhanced Adaptability: Ability to adjust strategies based on new information during execution
- Better Scalability: Decoupling of tools and context management
- Higher Controllability: Each decision step is rationally analyzed through LLM
- Stronger Debugging Capabilities: Complete execution traces and context states

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
- **Responsibility**: Strategic decision-making and execution orchestration
- **Core Method**: `execution_advance()` - Decides next execution plan or goal completion
- **Decision Types**:
  - `OrchestrationExecutionPlan`: Execute specific tool
  - `OrchestrationFinalization`: Complete goal pursuit

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
```python
async def execution_advance(self, goal, completed_steps, pending_steps, available_tools, context):
    for iteration in range(max_iterations):
        # 1. Context analysis
        orchestration_result = await self._context_advance(...)
        
        # 2. Decision judgment
        if orchestration_result.execution:
            return orchestration_result.execution
        elif orchestration_result.finalize:
            return orchestration_result.finalize
            
        # 3. Context refinement
        await context.refine(...)
        
        # 4. Tool expansion/contraction
        expanded_tools.update(orchestration_result.expand_tools)
        expanded_tools -= set(orchestration_result.collapse_tools)
```

#### 3.1.2 LLM Template Integration
- **Template Path**: `agent/templates/zh/strategic_context_orchestrator.jinja2`
- **Output Format**: Structured JSON, including analysis, resource queries, execution plans, etc.
- **JSON Schema**: Ensures output format consistency and parseability

#### 3.1.3 Context Refinement Mechanism
- **Dynamic Resource Loading**: Selective loading of resource attributes as needed
- **Relationship Filtering**: Exclude irrelevant resource relationships
- **Search History**: Maintain history of search queries
- **Segment Sorting**: Sort resource segments by importance

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
User Goal → Agent.pursue_goal()
    ↓
Create ResourceContext
    ↓
┌─────────────────────────────────────┐
│         Orchestration Loop          │
│                                     │
│ 1. OrchestrationLoop.execution_advance()
│    ├─ Call _context_advance()       │
│    ├─ LLM analyzes current state    │
│    └─ Return execution plan or completion decision
│                                     │
│ 2. If execution plan:               │
│    ├─ Agent._execute_tool()         │
│    ├─ Update completed_steps        │
│    └─ Continue loop                 │
│                                     │
│ 3. If completion decision:          │
│    └─ Return final response         │
│                                     │
│ 4. Context refinement:              │
│    ├─ ResourceContext.refine()      │
│    ├─ Update resource segments      │
│    └─ Adjust tool expansion state   │
└─────────────────────────────────────┘
    ↓
Return user response
```

### 4.2 Context Refinement Flow

```
OrchestrationResult
    ↓
ResourceContext.refine()
    ├─ Sort resource segments (sorted_segments)
    ├─ Query new resources (query_resources)  
    ├─ Exclude resources (exclude_resources)
    ├─ Include properties (include_properties)
    ├─ Exclude properties (exclude_properties)
    ├─ Execute searches (search_queries)
    └─ Reset exclusion state (if requested)
```

### 4.3 Tool Execution Flow

```
Agent._execute_tool()
    ↓
1. Validate tool existence
    ↓
2. Create AgentToolRuntime
    ↓
3. Call tool.call(runtime, **params)
    ├─ Tool internal logic execution
    ├─ User interaction through runtime
    └─ Return ToolResult or ToolError
    ↓
4. Construct StepOutcome
    ├─ Record execution time
    ├─ Collect progress information
    ├─ Collect triggered actions
    └─ Record backlog items
    ↓
5. Return execution result
```

## 5. Key Technical Points

### 5.1 LLM Integration Strategy

#### 5.1.1 Structured Output
- **JSON Schema**: Ensures consistency of LLM output format
- **Pydantic Validation**: Type-safe data parsing
- **Error Handling**: Graceful handling of LLM output format errors

#### 5.1.2 Template System
- **Jinja2 Templates**: Flexible prompt construction
- **Multi-language Support**: zh/en template directory structure
- **Context Injection**: Dynamic context information passing

### 5.2 Resource Management Strategy

#### 5.2.1 Selective Loading
- **Lazy Loading**: Load resource properties only when needed
- **Incremental Loading**: Gradually expand context based on analysis results
- **Memory Optimization**: Avoid loading unnecessary large datasets

#### 5.2.2 Relationship Management
- **Dynamic Filtering**: Filter resource relationships based on context needs
- **Hierarchical Navigation**: Support parent-child relationship navigation for resources
- **Cycle Detection**: Avoid infinite loops in resource relationships

### 5.3 Error Handling and Fault Tolerance

#### 5.3.1 Tool Execution Errors
- **Type-safe Errors**: `ToolResult` vs `ToolError`
- **Error Recovery**: Continue execution of other possible paths
- **User Notification**: Clear error message delivery

#### 5.3.2 LLM Response Errors
- **Format Validation**: JSON Schema validation
- **Retry Mechanism**: Retry strategy for format errors
- **Graceful Degradation**: Fallback options when valid responses cannot be obtained

## 6. Configuration and Extension

### 6.1 Agent Factory Pattern

```python
def create_agent(
    tools: dict[str, SchematicTool],
    resource_repo: ResourceRepository,
    template_env: TemplateEnvironment,
    chat_llm: ChatLLM,
    channel: AgentChannel,
    max_iterations: int = 5,
    min_iterations: int | None = None
) -> Agent:
    """Create a configured Agent instance"""
    orchestrator = OrchestrationLoop(
        template_env=template_env,
        chat_llm=chat_llm,
        max_iter=max_iterations,
        min_iter=min_iterations
    )
    
    return Agent(
        tools=tools,
        orchestrator=orchestrator,
        resource_repo=resource_repo,
        template_env=template_env,
        chat_llm=chat_llm,
        channel=channel
    )
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