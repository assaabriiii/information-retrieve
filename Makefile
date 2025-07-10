# Information Retrieval Project Makefile

.PHONY: setup test run clean jupyter help

help:
	@echo "Information Retrieval Project"
	@echo ""
	@echo "Usage:"
	@echo "  make setup      Create virtual environment and install dependencies"
	@echo "  make test       Run tests to verify project components"
	@echo "  make run        Run the information retrieval pipeline"
	@echo "  make jupyter    Start Jupyter notebook"
	@echo "  make clean      Remove generated files and directories"
	@echo "  make help       Show this help message"

setup:
	@echo "Setting up the project..."
	@chmod +x setup.sh
	@./setup.sh

test:
	@echo "Running tests..."
	@chmod +x test_project.py
	@./test_project.py

run:
	@echo "Running the information retrieval pipeline..."
	@chmod +x cli.py
	@./cli.py $(ARGS)

jupyter:
	@echo "Starting Jupyter notebook..."
	@jupyter notebook information_retrieval_demo.ipynb

clean:
	@echo "Cleaning up..."
	@rm -rf data/books/html/*
	@rm -rf data/quotes/html/*
	@rm -rf results/figures/*
	@rm -rf results/models/*
	@rm -rf __pycache__
	@rm -rf src/__pycache__
	@rm -rf src/*/__pycache__
	@echo "Cleanup complete!"