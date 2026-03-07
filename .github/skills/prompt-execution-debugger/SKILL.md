---
name: prompt-execution-debugger
description: Diagnose, investigate, and fix issues where NovelRAG prompt execution produces unexpected LLM behavior. Use this skill when the user reports a problem with how the agent executed a request and wants to understand why and how to fix the underlying prompt template. 
argument-hint: "[issue description or trace log file link]"
user-invocable: true
---

# Prompt Execution Debugger Skill

This skill diagnoses issues in NovelRAG prompt execution, identifies root causes in prompt templates or pipeline logic, prescribes fixes, and optionally guides rollback of corrupted operations. It is invoked when a user observes unexpected LLM behavior during agent execution and wants to investigate and resolve the issue.

## When to Use This Skill

- The user reports that the agent produced unexpected, incorrect, or hallucinated output during a pursuit.
- The user provides a trace log file and asks to analyze what went wrong.
- The user notices corrupted resources, wrong relationships, or junk backlog entries after an agent run.
- The user wants to understand why a specific LLM call in the pipeline misbehaved.
- The user explicitly asks to roll back operations from a bad execution.

## Supported Actions

- **diagnose** — Analyze a reported issue or trace log to identify which LLM call and template caused unexpected behavior. Produce a structured diagnosis with root cause, affected resources, and the responsible template.
- **prescribe** — After diagnosis, recommend specific changes to the prompt template that would prevent the issue. Delegate the actual template edit to the `prompt-template-author` skill.
- **rollback** — **Only when the user explicitly requests it.** Guide or execute undo operations to revert the effects of a bad execution. Rollback is always the lowest priority action — never perform it without explicit user confirmation.
- **inspect** — Read and interpret workspace state files (undo stack, backlog, resource YAML files) to understand the current state and what damage a bad execution may have caused.

## Expected System Behavior

Use this section as a reference for what "correct" looks like. When diagnosing an issue, compare actual behavior against these expectations to identify where the system diverged.

### Agent Interaction Model

The agent follows a **request → goal → multi-step pursuit → outcome** pattern:

1. The user sends a natural-language request or the agent autonomously generates a goal from backlog/exploration.
2. The goal translator converts it into an **observable, judgeable, implementation-agnostic** goal — it describes WHAT to achieve, never HOW. Goals must not contain fallback strategies, conditional branching, tool names, or step-by-step plans.
3. The agent enters a pursuit loop: repeatedly determines the next action (execute a tool or finalize), executes it, and assesses progress until a resolution is reached.
4. A resolution ends the pursuit with a status (`success`, `failed`, or `abandoned`), a reason, and a user-facing response.

**Key expectation:** Each pursuit should make steady progress toward the goal. If the agent loops without advancing (e.g., repeated context discovery with no new information, or repeated refinement rejections), something is wrong in the action determination templates.

### Expected Resource Write Workflow

Think of the agent as a professional novelist's writing assistant maintaining a structured outline. When the goal is to add or update a resource, the workflow should feel like a disciplined editorial process — not freewheeling generation:

1. **Understand the intent.** The assistant first clarifies what the user wants — create a new character, revise a faction's political stance, add an event — and frames it as a clear editorial objective. It does not jump to writing immediately.

2. **Survey the existing outline.** Before touching anything, the assistant reads through the relevant parts of the existing outline: what characters already exist, what relationships are established, what events are in play. It searches for related entries to avoid contradictions and finds patterns or templates that similar entries follow. This is the context discovery phase — it should be thorough but converge, not loop endlessly.

3. **Draft the content.** With full context in hand, the assistant produces multiple creative drafts (proposals) for the new or updated content, each from a different narrative perspective. It then evaluates and ranks these drafts, selecting the strongest one — like a writer trying several approaches before committing to the best fit.

4. **Translate the draft into structured edits.** The selected content is translated into precise, structured operations on the outline — which fields to set, which entries to insert or replace. The assistant presents these planned edits to the user for confirmation before applying anything. This is the only point where the user is asked to approve.

5. **Propagate consistency.** After the edits are applied, the assistant immediately reviews neighboring entries that might be affected. If a character's role changed, related characters' descriptions may need subtle adjustments. If a new faction was introduced, existing political relationships may need updating. These cascade updates happen automatically — the assistant discovers what needs to change, applies perspective updates to affected entries, and updates relationship links bidirectionally (if A references B, B must also reference A).

6. **Note follow-up work.** Finally, the assistant identifies work items that fall outside the current task but should be addressed later — a newly mentioned location that doesn't have its own entry yet, a plot thread that needs a continuity check, a character whose motivations should be fleshed out. These go into the backlog for future sessions.

**Key expectation:** The overall workflow should feel like a careful editorial pass — survey first, draft creatively, edit precisely, propagate consistency, and flag loose ends. If any stage is skipped, over-represented (e.g., endless surveying), or produces output disconnected from the existing outline, the corresponding template needs investigation.

## Investigation Workflow

### Phase 1: Gather Context

Before diagnosing, collect evidence from up to three sources depending on what the user provides:

#### Source A: User Description
The user describes the problem in natural language. Extract:
- What was the user's original request to the agent?
- What was the expected behavior?
- What actually happened?
- Which resources or aspects are affected?

#### Source B: Trace Log File
If the user provides a trace log file path (e.g., `workspace/logs/llm_log_YYYYMMDD_HHMMSS.yaml`), read and analyze it. See [Trace Log Schema](#trace-log-schema) below for the structure.

#### Source C: Workspace State
If the agent knows the workspace config location (typically `workspace/config.yml`), it can derive the paths to workspace state files:

```yaml
# From workspace/config.yml:
resource_config: aspect.yml        # → workspace/aspect.yml (aspect definitions)
undo_path: undo.json               # → workspace/undo.json (undo stack)
backlog_path: backlog.json         # → workspace/backlog.json (pending tasks)
# Resource files:                  # → workspace/<aspect_name>.yml (one per aspect)
# Trace logs:                      # → workspace/logs/llm_log_*.yaml
```

If the user has not provided the workspace path, and they mention working locally, check whether the current working directory contains a `workspace/` folder with a `config.yml`.

If any evidence indicates that the user is not running the local session you found (e.g., mismatched config, different resource names, or the user explicitly states a different environment), stop using local workspace files for investigation and continue the diagnosis based solely on the user's description and trace log analysis.

### Phase 2: Identify the Failing LLM Call

Locate which LLM call in the pipeline produced the unexpected output.

#### Path A: Trace Log Available

Use the trace log's `llm_calls` list (ordered chronologically) and the `template_name` field to map each call to its pipeline stage.

1. Scan the `llm_calls` in the relevant pursuit from the trace log.
2. For each call, compare `response.content` against the intent described in `request.messages`. Find the **first call where the response diverges** from expectations.
3. Use the `template_name` to locate the source `.jinja2` file. Templates are organized under `templates/en/` directories within their owning module. Search the workspace for the template filename if the path is not obvious.
4. Read the template source and the Python caller that renders it (search for the template filename in `.py` files) to understand the expected input variables and output pattern.

#### Path B: No Trace Log — Diagnose from User-Provided Output

When no trace log is available, work backwards from the text the user can see. The user may provide:

- **The final agent response** — the completion message shown after a pursuit finishes. This text originates from a `Resolution.response` field, generated by the action decision template when the agent finalizes a goal. Search the action determination code for how resolutions are constructed.
- **Intermediate progress messages** — text printed to the console during execution via `ctx.output()`. These messages are emitted at specific pipeline stages and contain recognizable patterns:

  | Message Pattern | Pipeline Stage | Relevant Code |
  |---|---|---|
  | `"Querying resource: ..."` | Context discovery | Action determination loop |
  | `"Searching: ..."` | Context discovery | Action determination loop |
  | `"Expanding tools: ..."` | Context discovery | Action determination loop |
  | `"Operation planned: ..."` | Tool execution start | Resource write tool |
  | `"Generating content: ..."` | Content generation | Content generation procedure |
  | `"Generated content for N tasks."` | Content generation | Resource write tool |
  | `"Applied N operation(s) successfully."` | Operation application | Resource write tool |
  | `"Discovered N perspective cascade update(s)."` | Cascade updates | Cascade update procedure |
  | `"Updating perspective: ..."` | Cascade updates | Cascade update procedure |
  | `"Discovered N relationship cascade update(s)."` | Cascade updates | Cascade update procedure |
  | `"Updating relationship: ..."` | Cascade updates | Cascade update procedure |
  | `"Updated relations: X → Y"` | Relation application | Cascade update procedure |
  | `"Updated relations for source/target resource ..."` | Relation write tool | Relation write tool |
  | `"Discovered N backlog item(s): ..."` | Backlog discovery | Backlog discovery procedure |

**Investigation steps:**
1. Match the user's output text against the message patterns above to identify which pipeline stage produced or preceded the issue.
2. Search the codebase for the matched message string (grep for keywords from the output) to find the exact source location.
3. From the source location, trace upstream to find which template and LLM call feeds into that stage. Look for `self.template.render(...)` or `self.env.load_template(...)` calls nearby.
4. Read the template and its caller to understand the expected behavior.

**Pipeline stage categories to consider** (from outermost to innermost):
- **Goal translation** — converts user request to a structured goal
- **Context discovery & refinement** — iteratively gathers and filters relevant resources
- **Action decision & refinement** — chooses which tool to execute or whether to finalize
- **Tool execution** — content generation, operation building, cascade updates, backlog discovery, summarization
- **Pursuit assessment** — evaluates progress between steps
- **Autonomous goal decision** — (if applicable) selects the next goal from backlog or exploration

### Phase 3: Root Cause Analysis

Once the failing LLM call is identified, classify the root cause:

#### 1. Template Instruction Ambiguity
The template allowed or encouraged the wrong output because its instructions are vague, contradictory, or missing guardrails for the observed scenario.

**Indicators:** The LLM response is plausible given the template text but not the intended behavior; an edge case is unaddressed.
**Remedy:** Prescribe template edits → delegate to `prompt-template-author`.

#### 2. Insufficient Context
The rendered prompt lacked information the LLM needed — missing resources, properties, or prematurely terminated context gathering.

**Indicators:** The LLM references entities not present in the prompt; workspace snapshot is incomplete.
**Remedy:** Investigate context discovery/refinement templates or the caller's `.render()` kwargs.

#### 3. Output Format Mismatch
The LLM's output structure doesn't match what the caller parses — wrong fields, malformed JSON, or mismatched tool call signatures.

**Indicators:** Parse errors downstream; extra or missing fields in the response.
**Remedy:** Cross-reference the template instructions with the caller's expected output schema (Pydantic model, JSON schema, or tool definitions).

#### 4. Cascade / Side-Effect Corruption
The primary call was correct, but downstream operations (cascade updates, relationship writes, backlog discovery) produced bad results.

**Indicators:** The primary write looks correct but related resources are corrupted, backlog entries are nonsensical, or relationship directions are reversed.
**Remedy:** Investigate the cascade/backlog templates invoked after the primary operation.

#### 5. Goal Misinterpretation
The goal translation produced a goal that doesn't match the user's intent, sending the entire pursuit off track.

**Indicators:** The translated goal in the trace log diverges from the original user request.
**Remedy:** Investigate the goal translation template.

#### 6. Action Decision Error
The agent chose the wrong tool, wrong parameters, or finalized prematurely/never finalized.

**Indicators:** Wrong tool selected; premature "success" finalization; infinite context discovery loop.
**Remedy:** Investigate the action decision and refinement analysis templates.

### Phase 4: Prescribe Fix

After identifying the root cause:

1. **Describe the fix** — Write a clear, specific description of what needs to change in the template: which section, what to add/modify/remove, and why.
2. **Delegate to `prompt-template-author`** — Hand off the actual template edit to the `prompt-template-author` skill with the specific action (edit), template name, and the prescribed changes. The `prompt-template-author` skill handles Jinja2 syntax, variable validation, and the verification checklist.

### Phase 5: Rollback (Only on Explicit User Request)

**CRITICAL: Never perform rollback without the user explicitly asking for it.** Rollback is destructive and is always the lowest priority action. Even when the diagnosis clearly shows corrupted data, prefer prescribing template fixes first. Only proceed with rollback when the user says something like "roll back", "undo the changes", "revert", etc.

#### Rollback via CLI Undo

If the user has a running session:
- The `undo` command in the CLI pops the last undo group and reverses all operations in it.
- Each undo group corresponds to one tool execution (a single `resource_write`, `resource_relation_write`, or `aspect_create` call and its cascade updates).
- Multiple undo commands may be needed to revert a full pursuit with multiple tool executions.

#### Rollback via Workspace File Inspection

If the user is not in a running session and wants to understand or manually revert changes:

1. **Read `workspace/undo.json`** to see the current undo stack. Each entry has:
   ```json
   {
     "method": "apply|update_relationships|add_aspect|remove_aspect",
     "params": { ... },
     "group": "group_tag_or_null"
   }
   ```
2. **Identify which undo entries** correspond to the bad execution. Entries are ordered oldest-to-newest. Entries with the same `group` value were part of one tool execution.
3. **Describe what each entry would revert** using the `ReversibleAction.description` format:
   - `apply` with `target: property` → "Property update on `<uri>` (fields: `<keys>`)"
   - `apply` with `target: resource` → "Inserted/Removed N resource(s) at `<location>`"
   - `update_relationships` → "Updated relationships between `<source>` and `<target>`"
   - `add_aspect` → "Added aspect `<name>`"
   - `remove_aspect` → "Removed aspect `<name>`"
4. **For manual cleanup**, the user can:
   - Edit `workspace/undo.json` to remove specific entries from the undo stack
   - Edit `workspace/backlog.json` to remove poisoned backlog entries
   - Edit `workspace/<aspect>.yml` to manually fix corrupted resource data
   - Re-run the session and use the `undo` command for the most recent groups

#### Backlog Cleanup

Bad executions often produce invalid backlog entries via the `discover_backlog.jinja2` template. When investigating:

1. **Read `workspace/backlog.json`** and check for entries that:
   - Reference non-existent resources in `target_resources`
   - Have descriptions based on hallucinated content
   - Were generated during the bad execution (cross-reference with trace log timestamps)
2. **Remove** those entries by editing `workspace/backlog.json` if the user confirms.

## Trace Log Schema

Trace logs are YAML files at `workspace/logs/llm_log_<timestamp>.yaml`. Structure:

```yaml
pursuits:
  - goal: "<goal description string>"
    started_at: "<ISO 8601 timestamp>"
    completed_at: "<ISO 8601 timestamp>"
    llm_calls:
      - template_name: "<jinja2 template filename>"
        timestamp: "<ISO 8601 timestamp>"
        duration_ms: <integer>
        request:
          messages:
            - role: "system"
              content: "<rendered prompt template>"
            - role: "user"
              content: "<trigger message>"
          response_format: "json_object"  # optional, only present if set
        response:
          content: "<LLM response string or parsed JSON>"
      - template_name: ...
        ...
```

> IMPORTANT: The trace log file could be very large. Read it carefully and avoid loading the entire file into memory if possible.

### Reading a Trace Log

1. **Identify the pursuit**: Match the `goal` field to the user's reported request.
2. **Scan `llm_calls` sequentially**: They are ordered chronologically. Each entry shows:
   - `template_name` — which template was used (maps to the pipeline stage map above)
   - `request.messages[0].content` — the full rendered prompt (system message)
   - `response.content` — the LLM's actual response
3. **Look for the divergence point**: Find the first LLM call where the response deviates from expectations. Compare the response against what the template instructions asked for.
4. **Check context**: In orchestrator calls (`strategic_context_orchestrator.jinja2` or `context_discovery.jinja2`), examine the workspace state embedded in the system prompt — were the right resources loaded? Were relevant resources excluded?
5. **Check cascades**: After the primary tool call, look for `discover_required_updates`, `build_perspective_update_operation`, `build_relation_update`, and `discover_backlog` calls — these produce side effects that may have corrupted the workspace.

## LLM Output Patterns Reference

When diagnosing output issues, know which pattern each template uses:

| Pattern | How to Identify | Templates Using It |
|---|---|---|
| **Structured Output** (Pydantic) | Caller uses `with_structured_output(Model)` — no response format in template | `translate_request_to_goal`, `assess_pursuit_progress`, `context_discovery`, `context_relevance`, `refinement_analysis`, `discover_required_updates`, `sort_edit_proposals`, `build_operation`, `build_perspective_update_operation`, `build_relation_update`, `discover_backlog`, `parse_relation_update_uris`, `initialize_aspect_metadata`, `generate_content_perspectives`, `generate_content_from_perspective`, `exploration_context_discovery`, `concept_gap_analysis`, `bootstrap_from_beliefs`, `goal_from_backlog`, `goal_from_exploration` |
| **JSON Object** | `response_format: json_object` in trace log; template has `## Response Format` section | `action_decision` (via `strategic_context_orchestrator.jinja2`), `get_updated_relations` |
| **Tool Binding** | Caller uses `bind_tools()`; response has `tool_calls` | `action_decision.jinja2` (when used with tool binding mode) |

## Diagnosis Output Format

When presenting a diagnosis to the user, structure it as:

```
## Diagnosis

**Issue**: <one-line summary of what went wrong>
**Pipeline Stage**: <stage name from the pipeline map>
**Template**: <template filename>
**Root Cause Category**: <category name>

### Evidence
- <specific evidence from trace log, workspace state, or user report>
- <quote from LLM response showing the problem>
- <comparison of expected vs actual behavior>

### Affected Resources
- <list of URIs that were incorrectly created/modified>

### Prescribed Fix
<description of what should change in the template>

### Recommended Action
1. <first action — usually a template edit via prompt-template-author>
2. <optional: backlog cleanup>
3. <optional: rollback — only if user explicitly requested>
```

## Key Principles

1. **Investigate before acting.** Always gather enough evidence to confirm the root cause before prescribing a fix.
2. **Rollback is the last resort.** Never undo operations without explicit user consent. Even after diagnosis, prefer fixing the template so future executions are correct.
3. **Delegate template edits.** This skill diagnoses and prescribes; the `prompt-template-author` skill performs the actual template modification with proper Jinja2 validation.
4. **Cross-reference trace logs with templates.** The trace log contains the rendered prompt — compare it against the `.jinja2` source to see if the rendering was correct (variable values, conditional blocks, etc.).
5. **Check the caller code.** When output format issues are suspected, read the Python caller to verify the output pattern (structured output, json_object, or tool binding) and the expected response schema.
6. **Consider cascade effects.** A single bad LLM call can corrupt multiple resources through cascade updates and generate misleading backlog entries. Trace the full chain of effects.
7. **Use workspace config as the entry point.** If the workspace config path is known, derive all other file paths from it rather than guessing.
