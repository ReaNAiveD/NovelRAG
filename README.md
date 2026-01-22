# NovelRAG

A **context-driven intelligent agent framework** for managing narrative content through multi-phase orchestration and Retrieval-Augmented Generation (RAG).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

## ✨ Key Features

- **Hierarchical Resource System**: Organize narrative elements (characters, locations, events) in a flexible structure
- **Vector-Based Semantic Search**: Fast similarity search using LanceDB embeddings
- **Async-First Design**: Modern Python async/await patterns throughout
- **Multi-LLM Support**: OpenAI, Azure OpenAI, and DeepSeek integrations

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
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
```

### 3. Run the Interactive Shell

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key"

# Run the shell
python -m novelrag.cli --config config.yml

# With verbose logging
python -m novelrag.cli --config config.yml -v

# Execute a single request
python -m novelrag.cli --config config.yml "Find the protagonist"
```

## Documentation

For detailed documentation, see the `docs/` directory:

- [Agent System Design](docs/agent_system_design.md) - Complete technical design document
- [Multi-Phase Orchestration](docs/multi_phase_orchestration.md) - Deep dive into the orchestration architecture
- [Terminology](docs/terminology.md) - Definitions of all system concepts

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2024 Qinkai Wu
