# Data Model

This document describes the data structures persisted by the NovelRAG system and their storage representations.

---

## Table of Contents

1. [Data](#data)
2. [Storage](#storage)

---

## Data

### Element

The core unit of world-building data. Each element represents a single narrative entity (a character, a location, an event, etc.).

| Field | Type | Description |
|---|---|---|
| `id` | `str` | Unique identifier within the aspect |
| `uri` | `str` | Hierarchical path (e.g. `/character/john_doe`) |
| `relationships` | `dict[str, list[str]]` | Map of target URI → list of relationship descriptions |
| `aspect` | `str` | The aspect this element belongs to |
| `children_keys` | `list[str]` | Keys under which child elements are nested |
| *(extra fields)* | any | Arbitrary properties (name, description, traits, etc.) |

Elements form a recursive tree: each `children_key` holds a list of child elements with the same structure.

When persisted, `uri`, `aspect`, and `children_keys` are excluded — they are derived from the aspect and tree position on load.

### Aspect

A named category that groups related elements. The aspect registry maps aspect names to their metadata.

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Aspect identifier (e.g. `character`, `faction`) |
| `description` | `str?` | Human-readable description |
| `children_keys` | `list[str]` | Keys defining the nested structure of elements |
| `metadata` | `dict` | Arbitrary metadata (constraints, definitions, examples, etc.) |

### Vector Embedding (Local)

A semantic embedding for each element, enabling similarity search.

| Field | Type | Description |
|---|---|---|
| `vector` | `float[3072]` | 3072-dimensional embedding vector |
| `hash` | `str` | MD5 hash of element content (for change detection) |
| `resource_uri` | `str` | URI of the source element |
| `aspect` | `str` | Aspect of the source element |

### Reversible Action

A single undoable operation recorded in the undo/redo stack.

| Field | Type | Description |
|---|---|---|
| `method` | `str` | Repository method name (`apply`, `update_relationships`, `add_aspect`, `remove_aspect`) |
| `params` | `dict` | Parameters to replay the inverse operation |
| `group` | `str?` | Optional tag for batched undo/redo |

### Backlog Entry

A follow-up work item for future autonomous processing.

| Field | Type | Description |
|---|---|---|
| `type` | `str` | Category (e.g. `dependency`, `continuity_check`, `world_building`) |
| `priority` | `int` | Numeric priority: `high=30`, `normal=20`, `low=10` |
| `description` | `str` | What needs to be done |
| `metadata` | `dict` | Extensible fields (context references, target resources, rationale, etc.) |

### Configuration

Top-level application configuration.

| Field | Type | Description |
|---|---|---|
| `chat_llm` | `ChatConfig` | Chat LLM endpoint settings (Azure OpenAI / OpenAI / DeepSeek) |
| `embedding` | `EmbeddingConfig` | Embedding model endpoint settings (Azure OpenAI / OpenAI) |
| `vector_store` | `VectorStoreConfig` | LanceDB URI, table name, overwrite/cleanup flags |
| `language` | `str?` | Content language code |
| `resource_config` | `str` | Path to the aspect registry file (default: `aspect.yml`) |
| `agent_beliefs` | `list[str]` | Background knowledge provided to the agent |
| `default_resource_dir` | `str` | Default directory for resource files |
| `undo_path` | `str?` | Path to undo/redo persistence file |
| `backlog_path` | `str?` | Path to backlog persistence file |

### Trace Span

A node in the hierarchical trace tree for observability.

| Field | Type | Description |
|---|---|---|
| `kind` | `SpanKind` | Nesting level: `session → intent → pursuit → tool_call → llm_call` |
| `name` | `str` | Span label |
| `span_id` | `str` | 12-char hex identifier |
| `start_time` | `datetime` | When the span started |
| `end_time` | `datetime?` | When the span ended |
| `duration_ms` | `float?` | Elapsed time in milliseconds |
| `status` | `str` | `"ok"` or `"error"` |
| `error` | `str?` | Error message if failed |
| `attributes` | `dict` | Arbitrary key-value data (LLM messages, token usage, etc.) |
| `children` | `list[Span]` | Nested child spans |

---

## Storage

### Local Storage

The default storage backend for CLI mode. All data is persisted to the workspace directory as flat files.

#### Resource Files

| Data | Format | Path |
|---|---|---|
| Aspect registry | YAML | `workspace/aspect.yml` |
| Element data | YAML (one file per aspect) | `workspace/<aspect>.yml` |

The aspect registry is a top-level dictionary keyed by aspect name:

```yaml
character:
  description: "Character entities in the narrative"
  children_keys: [sub_beliefs]
  constraints: "..."
  definition: "..."
```

Each aspect file is a top-level list of element records:

```yaml
- id: john_doe
  relationships:
    /character/jane_smith:
      - "childhood friend"
  name: John Doe
  description: "A retired detective"
  personality_traits: [cautious, analytical]
  sub_beliefs:
    - id: core_belief_1
      relationships: {}
      description: "..."
```

#### Vector Store

| Data | Format | Path |
|---|---|---|
| Embeddings | LanceDB on-disk | `workspace/resource/lancedb/` (table: `vectors`) |

#### Operational State

| Data | Format | Path |
|---|---|---|
| Undo/redo stacks | JSON | `workspace/undo.json` |
| Backlog queue | JSON | `workspace/backlog.json` |

Undo/redo JSON structure:

```json
{
  "undo_stack": [
    {"method": "apply", "params": {"op": {...}}, "group": "group_tag"}
  ],
  "redo_stack": [...]
}
```

Backlog JSON structure:

```json
[
  {"type": "continuity_check", "priority": 30, "description": "...", "metadata": {...}}
]
```

#### Configuration

| Data | Format | Path |
|---|---|---|
| Application config | YAML | `workspace/config.yml` |

#### Trace Logs

| Data | Format | Path |
|---|---|---|
| Session traces | YAML (one file per session) | `workspace/logs/llm_log_<timestamp>.yaml` |
