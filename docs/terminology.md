# NovelRAG Resource System Terminology

This document defines the core terminology used throughout the NovelRAG resource system to ensure consistent understanding and usage.

---

## Core Concepts

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
