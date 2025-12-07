.PHONY: help init run jupyter clean

help:
	@clear
	@echo ""
	@echo "Available targets:"
	@echo "  help    - Show this help message"
	@echo "  init    - Initialize the project (create directories, sync dependencies)"
	@echo "  run     - Run the main program"
	@echo "  jupyter - Launch Jupyter notebook server"
	@echo "  clean   - Remove generated files and caches"
	@echo ""

init:
	@echo ""
	@mkdir -p src scripts prompts notebooks data/input data/output
	@uv sync
	@echo "Project initialized successfully."
	@echo ""

run:
	@echo ""
	@uv run src/main.py
	@echo ""

jupyter:
	@echo ""
	@uv run jupyter notebook --notebook-dir=notebooks --NotebookApp.token='' --NotebookApp.password=''
	@echo ""

clean:
	@echo ""
	@rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@echo "Cleaned up generated files."
	@echo ""
