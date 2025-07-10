# Contributing to the Information Retrieval Project

Thank you for considering contributing to this project! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Code Contributions](#code-contributions)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

This project is committed to providing a welcoming and inclusive experience for everyone. Please be respectful and considerate of others when participating in this project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/info-ret.git`
3. Set up the development environment: `./setup.sh`
4. Run tests to ensure everything is working: `./test_project.py`

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue with the following information:

- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Any relevant logs or screenshots

### Suggesting Enhancements

If you have an idea for an enhancement, please create an issue with the following information:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples or mockups

### Code Contributions

1. Create a new branch for your feature or bugfix: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests to ensure your changes don't break existing functionality: `./test_project.py`
4. Commit your changes with a descriptive commit message
5. Push your branch to your fork: `git push origin feature/your-feature-name`
6. Create a pull request

## Development Setup

1. Clone the repository
2. Run `./setup.sh` to set up the virtual environment and install dependencies
3. Activate the virtual environment: `source venv/bin/activate`

## Pull Request Process

1. Ensure your code follows the style guidelines
2. Update the documentation if necessary
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit your pull request with a clear description of the changes

## Style Guidelines

### Python Code Style

- Follow PEP 8 style guidelines
- Use 4 spaces for indentation (no tabs)
- Use docstrings for all classes and functions
- Use type hints where appropriate

### Documentation Style

- Use Markdown for documentation
- Keep documentation up-to-date with code changes
- Be clear and concise

### Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests after the first line

## Project Structure

When adding new features, please follow the existing project structure:

- `src/scraper/`: Web scraping modules
- `src/preprocessing/`: Text preprocessing modules
- `src/analysis/`: Data analysis modules
- `src/visualization/`: Visualization modules

## Adding New Components

### Adding a New Scraper

To add a new scraper:

1. Create a new file in `src/scraper/` (e.g., `new_scraper.py`)
2. Implement a class that inherits from `BaseScraper`
3. Implement the required methods: `scrape_pages` and `extract_content`
4. Add tests for your scraper in `test_project.py`

### Adding a New Analysis Method

To add a new analysis method:

1. Add your method to the appropriate class in `src/analysis/`
2. Add tests for your method in `test_project.py`
3. Update the documentation to reflect the new method

### Adding a New Visualization

To add a new visualization:

1. Add your method to the `Visualizer` class in `src/visualization/visualizer.py`
2. Add tests for your visualization in `test_project.py`
3. Update the documentation to reflect the new visualization

## Thank You

Thank you for contributing to this project! Your time and effort are greatly appreciated.