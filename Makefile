.PHONY: help init run jupyter test clean destroy

help:
	@clear
	@echo ""
	@echo "Available targets:"
	@echo "  help    - Show this help message"
	@echo "  init    - Initialize the project (create directories, sync dependencies)"
	@echo "  run     - Run the main program"
	@echo "  jupyter - Launch Jupyter notebook server"
	@echo "  test    - Run unit tests"
	@echo "  clean   - Remove generated files and caches"
	@echo "  destroy - Remove .venv directory"
	@echo ""

init:
	@echo ""
	@mkdir -p src scripts prompts notebooks data/input data/output tests
	@uv sync --all-extras
	@echo "Project initialized successfully."
	@echo ""

test:
	@uv run pytest tests/ -v

run:
	@echo ""
	@uv run src/main.py
	@echo ""

jupyter:
	@uv run jupyter notebook --NotebookApp.token='' --NotebookApp.password=''

clean:
	@echo ""
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cleaned up generated files."
	@echo ""

destroy:
	@echo ""
	@rm -rf .venv
	@echo "Removed .venv directory."
	@echo ""
