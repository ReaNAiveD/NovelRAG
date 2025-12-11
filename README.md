# NovelRAG

A **context-driven intelligent agent framework** for managing narrative content through multi-phase orchestration and Retrieval-Augmented Generation (RAG). NovelRAG uses a sophisticated decision-making architecture to dynamically discover, refine, and act on contextual information.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## ✨ Key Features

- **Context-Driven Architecture**: Decisions based on dynamically discovered and refined resource contexts
- **Multi-Phase Orchestration**: Four-phase strategic decision engine (Discovery → Refinement → Decision → Analysis)
- **Hierarchical Resource System**: Organize narrative elements (characters, locations, events) in a flexible structure
- **Vector-Based Semantic Search**: Fast similarity search using LanceDB embeddings
- **Dynamic Tool Management**: Tools expand/collapse based on relevance to reduce context overhead
- **Iterative Goal Refinement**: Goals evolve through iterations to incorporate discovered requirements
- **Async-First Design**: Modern Python async/await patterns throughout
- **Multi-LLM Support**: OpenAI, Azure OpenAI, and DeepSeek integrations

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Context-Driven Agent Design](#context-driven-agent-design)
- [Resource System](#resource-system)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Requirements

- Python 3.9 or higher
- An LLM API key (OpenAI, Azure OpenAI, or DeepSeek)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/ReaNAiveD/NovelRAG.git
cd NovelRAG

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e ".[openai]"           # For OpenAI API
pip install -e ".[azure-ai]"         # For Azure AI Inference
pip install -e ".[azure-identity]"   # For Azure Identity authentication
pip install -e ".[test]"             # For testing
```

### Dependencies

Core dependencies are automatically installed:
- `jinja2` - Template engine for LLM prompts
- `lancedb` - Vector database for semantic search
- `pydantic` - Data validation and settings management
- `numpy`, `pandas`, `pyarrow` - Data processing

## Quick Start

### 1. Create a Configuration File

Create a `config.yml` file:

```yaml
# Embedding model configuration
embedding:
  type: openai
  endpoint: https://api.openai.com/v1
  model: text-embedding-3-large
  api_key: ${OPENAI_API_KEY}

# Chat model configuration
chat_llm:
  type: openai
  endpoint: https://api.openai.com/v1
  model: gpt-4
  api_key: ${OPENAI_API_KEY}

# Vector store configuration
vector_store:
  lancedb_uri: ./data/lancedb
  table_name: novel_embeddings
  overwrite: true

# Resource configuration
resource_config: aspect.yml
default_resource_dir: ./data/resources

# Define intents for interactive shell
intents:
  quit:
    cls: novelrag.intent.QuitIntent
```

### 2. Define Resource Aspects

Create an `aspect.yml` file to define your resource categories:

```yaml
character:
  path: characters.yml
  description: Story characters and their attributes
  children_keys:
    - relationships

location:
  path: locations.yml
  description: Story locations and settings
  children_keys:
    - sub_locations

scene:
  path: scenes.yml
  description: Story scenes and events
  children_keys: []
```

### 3. Run the Interactive Shell

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key"

# Run the shell
python -m novelrag --config config.yml

# With verbose logging
python -m novelrag --config config.yml -v

# Execute a single request
python -m novelrag --config config.yml "Find the protagonist"
```

### 4. Interactive Shell Commands

Once in the shell, you can:

```
> @character Find the protagonist
> @location Describe the main setting
> /search mysterious artifact
```

- `@aspect` - Switch to a specific aspect context
- `/intent` - Trigger a specific intent
- Free text - The agent will determine the appropriate action

## Architecture Overview

NovelRAG is built on two main systems that work together:

```
┌─────────────────────────────────────────────────────────────┐
│                    NovelRAG System                          │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐│
│  │               Agent System                             ││
│  │  ┌────────┐   ┌──────────────────┐   ┌──────────┐    ││
│  │  │ Agent  │──→│OrchestrationLoop │──→│  Tools   │    ││
│  │  │        │   │  (4 Phases)      │   │          │    ││
│  │  └────────┘   └────────┬─────────┘   └──────────┘    ││
│  │                        │                              ││
│  │             ┌──────────▼──────────┐                  ││
│  │             │  ResourceContext    │                  ││
│  │             │  (Context Builder)  │                  ││
│  │             └──────────┬──────────┘                  ││
│  └────────────────────────┼──────────────────────────────┘│
│                           │                                │
│  ┌────────────────────────▼──────────────────────────────┐│
│  │               Resource System                          ││
│  │  ┌─────────┐   ┌──────────────┐   ┌─────────────┐    ││
│  │  │ Aspects │──→│  Resources   │──→│  Elements   │    ││
│  │  │         │   │  (Entities)  │   │  (Data)     │    ││
│  │  └─────────┘   └──────────────┘   └─────────────┘    ││
│  └────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Context-Driven Agent Design

NovelRAG's agent system is fundamentally **context-driven**, meaning all decisions are based on dynamically discovered and refined contextual information rather than static step-by-step planning.

### Multi-Phase Orchestration

The orchestration loop implements a sophisticated four-phase architecture:

```
User Request → Initial Goal
    ↓
[Main Iteration Loop]
    │
    ├─[Context Loop]
    │  ├→ Phase 1: Context Discovery
    │  ├→ Apply discovery (search, load resources, expand tools)
    │  ├→ Phase 2: Context Refinement
    │  └→ Apply refinement (filter, sort, collapse tools)
    │
    ├→ Phase 3: Action Decision (execute/finalize)
    ├→ Phase 4: Refinement Analysis (approve/refine)
    │
    └→ Execute or return response
```

#### Phase 1: Context Discovery

**Purpose**: Aggressively explore and identify relevant context.

The system searches for resources semantically related to the goal, identifies specific resources to load, and determines which tools need detailed schemas.

**Output**: `DiscoveryPlan` with search queries, resource URIs, and tools to expand.

#### Phase 2: Context Refinement

**Purpose**: Filter and prioritize the discovered context.

After discovery, the system removes irrelevant resources, excludes unnecessary properties, collapses unneeded tools, and sorts context segments by priority.

**Output**: `RefinementPlan` with exclusions and prioritization.

#### Phase 3: Action Decision

**Purpose**: Make a decisive choice between executing a tool or finalizing.

The system always makes a clear decision—no ambiguous "need more context" states. It either selects a tool to execute with parameters or provides a final response.

**Output**: `ActionDecision` with execution details or finalization.

#### Phase 4: Refinement Analysis

**Purpose**: Strategic oversight and goal evolution.

The system validates the decision quality, identifies missing prerequisites, and refines the goal if needed. This enables the goal to evolve across iterations.

**Output**: `RefinementDecision` with approval or goal refinement.

### Goal Evolution

A key feature of the context-driven approach is that goals evolve through iterations:

```
Iteration 1: "Create protagonist named 余归"

Iteration 2: "Create protagonist named 余归 (Prerequisites: Check if character aspect exists)"

Iteration 3: "Create protagonist named 余归 (Prerequisites: 1. Verify character aspect, 2. Check for existing character...)"
```

### Dynamic Tool Management

Tools can be in two states to optimize context size:

- **Collapsed**: Only name and description visible (minimal context)
- **Expanded**: Full schema with parameters (detailed context)

The orchestration phases dynamically manage tool states based on relevance.

## Resource System

The resource system provides hierarchical data management for narrative content.

### Core Concepts

| Concept | Description |
|---------|-------------|
| **Aspect** | A category of resources (e.g., `character`, `location`, `scene`) |
| **Resource** | An individual entity within an aspect, identified by a URI |
| **Element** | The data structure representing a resource in memory |
| **URI** | Hierarchical path uniquely identifying a resource (e.g., `/character/john_doe`) |

### Resource Hierarchy

```
Root (/)
├── Aspect: character
│   ├── Resource: /character/john_doe
│   │   ├── Properties: {name, age, description}
│   │   └── Relations: {friend_of: [...], knows: [...]}
│   └── Resource: /character/sarah_chen
├── Aspect: location
│   └── Resource: /location/london
│       └── Children: {districts: [/location/london/westminster]}
└── Aspect: scene
    └── Resource: /scene/opening
```

### Resource URIs

URIs follow a hierarchical format: `/<aspect>[/<resource_id>[/<nested_id>...]]`

Examples:
- `/` - Root (all aspects)
- `/character` - All characters
- `/character/john_doe` - Specific character
- `/location/europe/london` - Nested location

## Configuration

### Complete Configuration Example

```yaml
# LLM Configuration
embedding:
  type: azure_openai
  endpoint: https://your-resource.openai.azure.com
  deployment: text-embedding-3-large
  api_version: "2024-02-01"
  model: text-embedding-3-large
  api_key: ${OPENAI_API_KEY}

chat_llm:
  type: azure_openai
  endpoint: https://your-resource.openai.azure.com
  deployment: gpt-4o
  api_version: "2024-08-01-preview"
  model: gpt-4o
  api_key: ${OPENAI_API_KEY}
  max_tokens: 4000
  temperature: 0.0

# Vector Store
vector_store:
  lancedb_uri: ./data/vectors
  table_name: resources
  overwrite: true
  cleanup_invalid_on_init: true

# Resources
resource_config: aspect.yml
default_resource_dir: ./data
template_lang: en  # or "zh" for Chinese templates

# Scopes and Intents
scopes:
  character:
    search:
      cls: novelrag.intent.SearchIntent
      kwargs:
        aspect: character

intents:
  quit:
    cls: novelrag.intent.QuitIntent
  help:
    cls: novelrag.intent.HelpIntent
```

### Environment Variables

Configuration supports environment variable substitution:

```yaml
api_key: ${OPENAI_API_KEY}
endpoint: ${AZURE_OPENAI_ENDPOINT}
```

## Documentation

For detailed documentation, see the `docs/` directory:

- [Agent System Design](docs/agent_system_design.md) - Complete technical design document
- [Multi-Phase Orchestration](docs/multi_phase_orchestration.md) - Deep dive into the orchestration architecture
- [Terminology](docs/terminology.md) - Definitions of all system concepts

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Setup

```bash
# Install with test dependencies
pip install -e ".[test]"

# Run tests
pytest

# Run with coverage
pytest --cov=novelrag --cov-report=term-missing
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Qinkai Wu
