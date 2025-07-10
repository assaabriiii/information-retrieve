#!/bin/bash

# Information Retrieval Project Setup Script

# Create virtual environment


# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/books/html
mkdir -p data/quotes/html
mkdir -p results/figures
mkdir -p results/models

echo "Setup complete! You can now run the project using:"
echo "python main.py"
echo "or"
echo "jupyter notebook information_retrieval_demo.ipynb"