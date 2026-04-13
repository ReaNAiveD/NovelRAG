---
name: lint-and-typecheck
description: Format, lint, and type check the NovelRAG codebase using ruff and ty. Use this skill after making code changes to ensure they conform to project style and pass type checking.
argument-hint: "[check | fix | format]"
user-invocable: true
---

# Lint and Type Check Skill

This skill runs the project's formatting, linting, and type checking tools (`ruff` and `ty`) and helps resolve any issues found.

## Prerequisites

Install dev dependencies before running any commands:

```shell
uv sync --extra dev
```

## Commands

All commands MUST be run with `uv run` from the project root.

### Formatting (ruff format)

```shell
uv run ruff format
```

- Check formatting without modifying files: `uv run ruff format --check`
- Format a specific file: `uv run ruff format <path>`

### Linting (ruff check)

```shell
uv run ruff check
```

- Auto-fix lint issues: `uv run ruff check --fix`
- Lint a specific file: `uv run ruff check <path>`

### Type Checking (ty)

```shell
uv run ty check
```

## Typical Workflow

After editing code, run the tools in this order:

1. `uv run ruff format` — apply formatting
2. `uv run ruff check --fix` — fix auto-fixable lint issues
3. `uv run ty check` — verify types

If any step reports errors that cannot be auto-fixed, read the diagnostic output, fix the code, and re-run.

## Configuration Reference

All tool configuration lives in `pyproject.toml`:

- **ruff** — `[tool.ruff]`, `[tool.ruff.lint]`, `[tool.ruff.format]`
  - Target: Python 3.12, line length 120
  - Enabled rule sets: E, W, F, I, UP, B, SIM
  - Quote style: double, indent style: space
- **ty** — `[tool.ty.*]`
  - Includes `novelrag/**/*.py`, excludes `.pyi` stubs
