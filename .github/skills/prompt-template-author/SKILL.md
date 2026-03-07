---
name: prompt-template-author
description: The skill is used to author, edit, manage, and verify prompt templates used in NovelRAG. Use this skill when you need to create or modify prompt templates or ensure they are correctly formatted and functional.
argument-hint: "[action] [template_name]"
user-invocable: true
---

# Jinja2 Prompt Template Authoring Skill

This skill is designed to assist in the creation, editing, management, and verification of Jinja2 prompt templates used in NovelRAG.

## Supported Actions

- **create** — Create a new template file and integrate it with its caller Python code. Workflow: create `.jinja2` file → create/update caller class with `TemplateEnvironment` and `load_template` → (optional) define a Pydantic structured output model.
- **edit** — Modify an existing template. Always read the caller's `.render()` call first to understand available variables and the output pattern (structured output vs. template-defined format).
- **review** — Check an existing template for issues: undefined variables, missing conditional guards, terminology inconsistencies, heading convention violations, cache-unfriendly ordering.
- **verify** — Validate that a template renders correctly: all referenced variables exist in the caller, conditional blocks handle None/empty, structured output model fields align with template instructions, and JSON response formats are parseable.

## File Layout and Naming

### Directory Structure

Templates are co-located with their owning module under a `templates/<lang>/` directory.

Each `en/` directory has a corresponding `zh/` directory for Chinese translations. You can find other language directories as well.

## Template Loading and Invocation

### TemplateEnvironment

Each module creates a `TemplateEnvironment` pointing to its own package:

```python
self.env = TemplateEnvironment(package_name="novelrag.resource_agent.tool", default_lang=lang)
self.template = self.env.load_template("build_operation.jinja2")
```

`load_template(name, lang)` resolves templates with a fallback chain: **requested lang → default lang → `en` → any available lang**. See `novelrag/template.py` for the full implementation.

### Rendering and LLM Invocation

Templates are rendered to a string prompt via `.render(**kwargs)` and then passed to the LLM:

```python
prompt = self.template.render(goal=goal, beliefs=beliefs, context=snapshot)
response = await llm.ainvoke([SystemMessage(content=prompt)])
```

The rendered template is almost always sent as a `SystemMessage`. Some callers add a short `HumanMessage` as an execution trigger (e.g., `"Generate the operations in JSON format."`). The template itself is the system instruction, not the user query.

## Prompt Template Structure

A prompt template typically includes these components, in this order:

1. **System Architecture / Domain Education** — Explain relevant domain concepts (aspects, resources, URIs, operations) so the LLM has the necessary context. Reference `docs/terminology.md` for canonical term definitions. Key terms that must be used consistently.
2. **Template Functionalities** — The specific functionalities that the prompt template is designed to support.
3. **Task Definition** — A clear definition of the task. Use `## Your Responsibility` or `## Task` as the heading.
4. **Guidelines / Critical Principles** — Numbered lists of constraints, priorities, and rules. Use `## Guidelines` or `## Critical Principles`.
5. **Contextual Information** — Dynamic data sections (goal, assessment, workspace state, tool schemas).
6. **Response Format** — Only needed when the template defines its own response format (see LLM Output Patterns below). Omit when using `with_structured_output()`.

### Section Heading Conventions

- `##` for major sections (e.g., `## System Architecture`, `## Your Responsibility`, `## Current Goal`, `## Available Tools`)
- `###` for subsections (e.g., `### Expanded Tools`, `### When to EXECUTE`)
- `**Bold text**` for field labels and emphasis within lists (e.g., `**Finished Tasks**:`)
- Numbered lists for ordered guidelines and rules
- Avoid `#` top-level headings (used only in a few agenturn templates)

### Cache-Friendly Ordering

Put the most stable, widely-reused content at the top of the template to maximize LLM provider prompt caching across invocations:

1. **Static content first**: System architecture explanations, domain education, terminology definitions, static guidelines/rules
2. **Semi-static content next**: Tool schemas, response format definitions
3. **Dynamic content last**: Goal, assessment, workspace segments, completed steps, interaction history

This ordering creates a shared prefix across invocations that benefits from prompt caching.

## LLM Output Patterns

Templates pair with one of three LLM output patterns. Always check the caller to determine which pattern is in use before editing a template.

### Pattern A: Structured Output (majority of templates)

The template provides instructions only. The response schema is enforced via `with_structured_output(PydanticModel)` on the LLM. The template does **NOT** need a response format section — the Pydantic model defines the schema.

```python
llm = self.chat_llm.with_structured_output(DiscoveryPlan)
response = await llm.ainvoke([SystemMessage(content=prompt)])
# response is a DiscoveryPlan instance
```

When writing templates for this pattern, ensure the template instructions naturally guide the LLM to produce content that aligns with the Pydantic model's field names and types.

### Pattern B: JSON Object Response (some tool templates)

The template includes an explicit response format section. The LLM is configured with `response_format={"type": "json_object"}` and the response is manually parsed.

```python
llm = self.chat_llm.bind(response_format={"type": "json_object"})
response = await llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content="Generate in JSON format.")])
parsed = json.loads(response.content)
```

When writing templates for this pattern, include a `## Response Format` section with the exact JSON schema the caller expects.

### Pattern C: Tool Binding (rare)

The template provides a decision framework. The LLM responds by calling bound tools, not by producing JSON or text.

```python
llm = self.chat_llm.bind_tools(tool_definitions)
response = await llm.ainvoke([SystemMessage(content=prompt)])
# response.tool_calls contains the LLM's tool invocations
```

Used by `action_decision.jinja2`. When writing templates for this pattern, describe the available tools, their purposes, and when to use each one.

## Jinja2 Syntax Conventions

### Whitespace Stripping

Always use `{%-` and `-%}` for control tags to produce clean output without extra blank lines:

```jinja2
{%- if pursuit_assessment %}
- **Finished Tasks**: {{ pursuit_assessment.finished_tasks | join("; ") }}
{%- else %}
No assessment available yet.
{%- endif %}
```

### Conditional Blocks

Wrap all optional context in conditional guards. Provide fallback text for empty sections:

```jinja2
{%- if beliefs %}
## Agent Beliefs
{%- for belief in beliefs %}
- {{ belief }}
{%- endfor %}
{%- endif %}
```

### Iteration

```jinja2
{%- for key, value in segment.included_data.items() %}
- {{ key }}: {{ value }}
{%- endfor %}

{%- for step in completed_steps %}
### Step {{ loop.index }}
{%- endfor %}
```

### Filters

Only use built-in Jinja2 filters. No custom macros or filters are defined.

| Filter | Usage | Example |
|--------|-------|---------|
| `join` | Concatenate list items | `{{ list \| join("; ") }}` |
| `replace` | Indent multi-line content | `{{ text \| replace("\n", "\n    > ") }}` |
| `upper` | Uppercase status values | `{{ status.value \| upper }}` |
| `length` | Check collection size | `{%- if items \| length > 0 %}` |

## Variable Passing Conventions

Templates receive variables via `.render(**kwargs)`. Common variable types:

- **Pydantic models**: `Goal`, `PursuitAssessment`, `ContextSnapshot`, `SegmentData` — access nested fields via `{{ object.field }}`
- **Dataclasses**: `OperationOutcome`, `OperationPlan` — same dot-access pattern
- **Dicts**: Tool schemas, operation data — access via `{{ dict.key }}` or `{% for k, v in dict.items() %}`
- **Lists**: `beliefs` (list of strings), `completed_steps` (list of outcomes) — iterate with `{% for item in list %}`
- **Strings**: `request`, `interaction_history` — direct interpolation via `{{ variable }}`

**Before editing any template**, always read the caller's `.render()` call to discover the exact variable names and types available.

## Prompt Template Best Practices

When creating or editing prompt templates, consider the following best practices:
- **Clarity and Specificity**: Ensure that the prompt is clear and specific to guide the language model effectively.
- **Use Python object instead of rebuilding JSON**: When possible, use Python objects to represent complex data structures instead of rebuilding them as JSON when calling the prompt.
- **Shared Terminology**: Define and use shared terminology consistently across prompt templates. Reference `docs/terminology.md` as the canonical source of truth for term definitions.
- **Understand the Context**: Always fully understand the code context and the task requirements before authoring or editing a prompt template to ensure it is appropriately tailored to the intended use case.
- **Multi-Lingual Support**: The english version of the prompt template should be the source of truth. Only start authoring or editing the prompt template in other languages after the english version is finalized. When authoring or editing non-english versions of the prompt template, ensure that the meaning and intent of the original english version are preserved while making necessary adjustments for cultural and linguistic differences.

## Verification Checklist

After creating or editing a template, verify the following:

1. **Variable existence**: Every `{{ variable }}` and `{%- if variable %}` referenced in the template has a corresponding keyword argument in the caller's `.render()` call.
2. **Conditional guards**: All optional context sections are wrapped in `{%- if variable %}...{%- endif %}` and handle the None/empty case gracefully.
3. **Output pattern alignment**: If the caller uses `with_structured_output(Model)`, the template instructions naturally guide the LLM to produce content matching the Pydantic model's field names and types. If the caller uses `json_object` response format, the template includes a complete JSON schema in a Response Format section.
4. **Terminology consistency**: Domain terms match the definitions in `docs/terminology.md` (e.g., "aspect" not "category", "resource" not "entity", "URI" not "path").
5. **Heading conventions**: Major sections use `##`, subsections use `###`, field labels use `**bold**`.
6. **Whitespace**: Control tags use `{%-` / `-%}` to avoid blank lines in rendered output.
7. **Cache ordering**: Static domain education and guidelines appear before dynamic context data.
