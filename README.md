# Bandsalat

A Python project.

## Repository Structure

```
bandsalat/
├── pyproject.toml          # Project dependencies and metadata
├── Makefile                # Build and run commands
├── CLAUDE.md               # AI development rules
├── README.md               # This file
├── .gitignore              # Git ignore patterns
├── src/                    # Source code
│   └── main.py             # Main entry point
├── scripts/                # Utility scripts
├── prompts/                # LLM prompt templates
└── data/                   # Data files
    ├── input/              # Input data files
    └── output/             # Generated output files
```

## Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) installed

## Setup

Initialize the project:

```bash
make init
```

## Usage

Run the main program:

```bash
make run
```

View all available commands:

```bash
make help
```
