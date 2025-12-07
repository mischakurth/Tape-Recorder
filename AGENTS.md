# Development Rules for Bandsalat

## General Coding Principles

- **Never assume any default values anywhere**
- Always be explicit about values, paths, and configurations
- If a value is not provided, handle it explicitly (raise error, use null, or prompt for input)

## Git Commit Guidelines

- **NEVER include AI attribution in commit messages**
- **NEVER add "Generated with Claude Code" or similar phrases**
- **NEVER add "Co-Authored-By: Claude" or similar attribution**
- **NEVER run `git add -A` or `git add .` - always stage files explicitly**
- Keep commit messages professional and focused on the changes made
- Commit messages should describe what changed and why, without mentioning AI assistance

## Testing

- After **every change** to the code, the tests must be executed
- Always verify the program runs correctly with `make run` after modifications

## Python Execution Rules

- Python code must be executed **only** via `uv run ...`
  - Example: `uv run src/main.py`
  - **Never** use: `python src/main.py` or `python3 src/main.py`
- The virtual environment must be created and updated **only** via `uv sync`
  - **Never** use: `pip install`, `python -m pip`, or `uv pip`
- All dependencies must be managed through `uv` and declared in `pyproject.toml`

## Makefile Rules

- All Python execution in the Makefile uses `uv run`, never `python` directly
- Run `make help` to see all available targets

## Project Structure

- All source code lives in `src/`
- Test scripts and utilities go in `scripts/`
- Prompt templates go in `prompts/`
- **Input data**: `data/input/`
- **Output data**: `data/output/`
- **Never create Python files in the project root directory**
  - Wrong: `./test.py`, `./helper.py`
  - Correct: `./src/helper.py`, `./scripts/test.py`

## Error Handling

- Scripts should continue processing other files even if one fails
- Failed/invalid outputs should be logged or moved to an `error/` subdirectory
- Scripts should track and report success/failure counts
- Exit with code 1 if any files failed, 0 if all succeeded

## Optimization

- **Skip processing if output already exists** - Don't reprocess files unnecessarily
- Check if output file exists before starting expensive operations
- Track skipped files separately in summary reports
- Allow users to force reprocessing by deleting output files
