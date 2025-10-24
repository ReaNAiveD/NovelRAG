import json
from dataclasses import dataclass, field
from novelrag.agent.steps import StepOutcome
from novelrag.agent.tool import LLMToolMixin, SchematicTool
from novelrag.agent.workspace import ResourceContext
from novelrag.llm.types import ChatLLM
from novelrag.template import TemplateEnvironment


@dataclass(frozen=True)
class OrchestrationExecutionPlan:
    reason: str
    tool: str
    params: dict = field(default_factory=dict)
    future_steps: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OrchestrationFinalization:
    reason: str
    response: str
    status: str  # success, failed, abandoned


@dataclass(frozen=True)
class OrchestrationResult:
    analysis: str
    query_resources: list[str] = field(default_factory=list)
    exclude_resources: list[str] = field(default_factory=list)
    include_properties: list[dict[str, str]] = field(default_factory=list)
    exclude_properties: list[dict[str, str]] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    sorted_segments: list[str] = field(default_factory=list)
    expand_tools: list[str] = field(default_factory=list)
    collapse_tools: list[str] = field(default_factory=list)
    execution: OrchestrationExecutionPlan | None = None
    finalize: OrchestrationFinalization | None = None


class OrchestrationLoop(LLMToolMixin):
    def __init__(self, template_env: TemplateEnvironment, chat_llm: ChatLLM, max_iter: int | None = 5, min_iter: int | None = None):
        super().__init__(template_env, chat_llm)
        self.max_iter = max_iter
        self.min_iter: int = min_iter or 0

    async def execution_advance(
            self,
            user_request: str,
            goal: str,
            completed_steps: list[StepOutcome],
            pending_steps: list[str],
            available_tools: dict[str, SchematicTool],
            context: ResourceContext,
    ) -> OrchestrationExecutionPlan | OrchestrationFinalization:
        """
        Advance the execution orchestration by determining the next tool to execute
        or finalizing the goal pursuit.

        Uses the execution orchestrator template with structured JSON schema
        to ensure consistent response format for execution decisions.
        """
        last_result: OrchestrationExecutionPlan | OrchestrationFinalization | None = None
        iter_num = 0
        expanded_tools = set()
        while True:
            iter_num += 1
            expanded_tool_dict = {name: {
                "description": available_tools[name].description,
                "input_schema": available_tools[name].input_schema,
                "output_description": available_tools[name].output_description,
            } for name in expanded_tools if name in available_tools}
            collapsed_tool_dict = {name: {
                "description": available_tools[name].description,
            } for name in available_tools if name not in expanded_tools}
            orchestration_result = await self._context_advance(
                user_request,
                goal,
                iter_num,
                completed_steps,
                pending_steps,
                {
                    **expanded_tool_dict,
                    **collapsed_tool_dict
                },
                context
            )

            if orchestration_result.execution:
                last_result = orchestration_result.execution
                if iter_num >= self.min_iter:
                    break
            elif orchestration_result.finalize:
                last_result = orchestration_result.finalize
                if iter_num >= self.min_iter:
                    break
            if iter_num == self.max_iter and orchestration_result.finalize:
                last_result = orchestration_result.finalize
                break
            if self.max_iter and iter_num >= self.max_iter:
                break

            await context.refine(
                orchestration_result.sorted_segments,
                orchestration_result.query_resources,
                orchestration_result.exclude_resources,
                orchestration_result.include_properties,
                orchestration_result.exclude_properties,
                orchestration_result.search_queries,
            )
            expanded_tools.update(orchestration_result.expand_tools)
            expanded_tools.difference_update(orchestration_result.collapse_tools)
        if not last_result:
            raise ValueError("Orchestration did not produce any execution or finalization result.")
        return last_result
        

    async def _context_advance(
            self,
            user_request: str,
            goal: str,
            iter_num: int,
            completed_steps: list[StepOutcome],
            pending_steps: list[str],
            available_tools: dict[str, dict],
            context: ResourceContext,
    ) -> OrchestrationResult:
        """
        Advance the context orchestration by determining the next strategic actions.
        
        Uses the strategic context orchestrator template with structured JSON schema
        to ensure consistent response format for orchestration decisions.
        """
        # Build workspace segments data for the template
        workspace_segments = []
        nonexisted_uris = []
        for segment in context.workspace.sorted_segments():
            if segment_data := await context.build_segment_data(segment):
                workspace_segments.append(segment_data)
            else:
                nonexisted_uris.append(segment.uri)
        
        # Prepare last step data if available
        last_step = None
        if completed_steps:
            last_completed = completed_steps[-1]
            last_step = {
                "tool": last_completed.action.tool,
                "intent": last_completed.action.reason,
                "status": last_completed.status.value,
                "results": last_completed.results
            }
        
        # Define the JSON schema for structured response with comprehensive descriptions
        # TODO: At present, the `execution.params` is not strictly defined and can be any dict. The schema cannot be accepted by LLMs
        response_schema = {
            "type": "object",
            "description": "Strategic orchestration decision for context management and execution planning",
            "properties": {
                "analysis": {
                    "type": "string",
                    "description": "Comprehensive strategic analysis of the current situation, including context assessment, goal alignment, execution readiness, and reasoning for the orchestration decisions made"
                },
                "query_resources": {
                    "type": "array",
                    "description": "List of resource URIs to fetch and include in the workspace context. These should be strategically relevant resources that will help with planning or execution",
                    "items": {
                        "type": "string",
                        "description": "Resource URI in the format /aspect/resource_id or nested /aspect/parent/child"
                    }
                },
                "exclude_resources": {
                    "type": "array", 
                    "description": "List of resource URIs to remove from the workspace context to reduce noise and focus on relevant information",
                    "items": {
                        "type": "string",
                        "description": "Resource URI to exclude from the context"
                    }
                },
                "include_properties": {
                    "type": "array",
                    "description": "Specific properties to include for resources in the workspace. Use when you need specific data fields for planning or execution",
                    "items": {
                        "type": "object",
                        "description": "Property inclusion specification for a specific resource",
                        "properties": {
                            "uri": {
                                "type": "string",
                                "description": "The resource URI for which to include the property"
                            },
                            "property": {
                                "type": "string", 
                                "description": "The specific property name to include (e.g., 'name', 'description', 'motivation')"
                            }
                        },
                        "required": ["uri", "property"],
                        "additionalProperties": False
                    }
                },
                "exclude_properties": {
                    "type": "array",
                    "description": "Specific irrelated properties to exclude from resources to reduce context size while maintaining relevant information",
                    "items": {
                        "type": "object",
                        "description": "Property exclusion specification for a specific resource", 
                        "properties": {
                            "uri": {
                                "type": "string",
                                "description": "The resource URI for which to exclude the property"
                            },
                            "property": {
                                "type": "string",
                                "description": "The specific property name to exclude from the context"
                            }
                        },
                        "required": ["uri", "property"],
                        "additionalProperties": False
                    }
                },
                "search_queries": {
                    "type": "array",
                    "description": "Semantic search queries to find relevant resources not currently in the workspace. Use when you need to discover resources related to the goal",
                    "items": {
                        "type": "string",
                        "description": "Natural language search query for finding semantically similar resources"
                    }
                },
                "sorted_segments": {
                    "type": "array",
                    "description": "Ordered list of resource URIs indicating the priority/relevance for the current goal. Most important resources should be listed first",
                    "items": {
                        "type": "string",
                        "description": "Resource URI in order of strategic importance"
                    }
                },
                "expand_tools": {
                    "type": "array",
                    "description": "List of tool names to show detailed schemas for. Use when planning requires understanding specific tool capabilities and parameters",
                    "items": {
                        "type": "string",
                        "description": "Tool name to expand and show detailed schema information"
                    }
                },
                "collapse_tools": {
                    "type": "array",
                    "description": "List of tool names to hide detailed schemas for. Use to reduce context length when tool is not related to immediate execution",
                    "items": {
                        "type": "string",
                        "description": "Tool name to collapse and hide detailed schema information"
                    }
                },
                "execution": {
                    "type": "object",
                    "description": "Immediate execution plan when sufficient context is available for the task, even if additional related context could potentially be gathered. Use when current context provides enough information to proceed confidently with tool execution",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Explanation of why this tool execution is recommended and how it advances toward the goal"
                        },
                        "tool": {
                            "type": "string",
                            "description": "Name of the tool to execute (must match one of the available tools)"
                        },
                        "params": {
                            "type": "object",
                            "description": "Parameters to pass to the tool. Should match the tool's expected input schema",
                            "additionalProperties": True
                        },
                        "future_steps": {
                            "type": "array",
                            "description": "Anticipated future steps that may be needed after this execution completes",
                            "items": {
                                "type": "string",
                                "description": "Description of a potential future step or action"
                            }
                        }
                    },
                    "required": ["reason", "tool", "params"],
                    "additionalProperties": False
                },
                "finalize": {
                    "type": "object",
                    "description": "Goal completion or termination only when no further actions are needed or possible",
                    "properties": {
                        "reason": {
                            "type": "string",
                            "description": "Explanation of why the goal is being finalized (completed successfully, failed, or abandoned)"
                        },
                        "response": {
                            "type": "string",
                            "description": "Final response to provide to the user summarizing the outcome"
                        },
                        "status": {
                            "type": "string",
                            "description": "Final status of the goal pursuit",
                            "enum": ["success", "failed", "abandoned"]
                        }
                    },
                    "required": ["reason", "response", "status"],
                    "additionalProperties": False
                }
            },
            "required": ["analysis", "sorted_segments"],
            "additionalProperties": False
        }
        
        # Call the template with structured response
        response_json = await self.call_template(
        # response_json = await self.call_template_structured(
            "strategic_context_orchestrator.jinja2",
            # response_schema=response_schema,
            json_format=True,
            user_request=user_request,
            goal=goal,
            iter_num=iter_num,
            max_iter=self.max_iter,
            last_step=last_step,
            completed_steps=[{
                "tool": step.action.tool,
                "intent": step.action.reason,
                "status": step.status.value,
                "results": step.results,
                "error": step.error_message
            } for step in completed_steps],
            pending_steps=pending_steps,
            available_tools=available_tools,
            workspace_segments=workspace_segments,
            nonexisted_uris=nonexisted_uris,
            search_history=[{
                "query": item.query,
                "aspect": item.aspect,
                "uris": item.uris
            } for item in context.search_history]
        )
        
        # Parse the structured JSON response
        try:
            response_data = json.loads(response_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse orchestration response: {str(e)}")
        
        # Convert property operations from dict format directly (no conversion needed)
        include_properties = response_data.get("include_properties", [])
        exclude_properties = response_data.get("exclude_properties", [])
        
        # Build execution plan if present
        execution = None
        if exec_data := response_data.get("execution"):
            execution = OrchestrationExecutionPlan(
                reason=exec_data["reason"],
                tool=exec_data["tool"],
                params=exec_data["params"],
                future_steps=exec_data.get("future_steps", [])
            )
        
        # Build finalization if present
        finalize = None
        if fin_data := response_data.get("finalize"):
            finalize = OrchestrationFinalization(
                reason=fin_data["reason"],
                response=fin_data["response"],
                status=fin_data["status"]
            )
        
        # Return structured orchestration result
        return OrchestrationResult(
            analysis=response_data["analysis"],
            query_resources=response_data.get("query_resources", []),
            exclude_resources=response_data.get("exclude_resources", []),
            include_properties=include_properties,
            exclude_properties=exclude_properties,
            search_queries=response_data.get("search_queries", []),
            sorted_segments=response_data["sorted_segments"],
            expand_tools=response_data.get("expand_tools", []),
            collapse_tools=response_data.get("collapse_tools", []),
            execution=execution,
            finalize=finalize
        )
