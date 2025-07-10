# Information Retrieval Project Documentation

## Project Overview

This project implements an information retrieval system that crawls web pages, processes text data, and performs various analyses including clustering and classification. The system is designed to be modular and extensible, allowing for easy addition of new data sources and analysis methods.

## Project Structure

```
.
├── cli.py                         # Command-line interface for the project
├── data/                          # Directory for storing scraped data
│   ├── books/                     # Books data from books.toscrape.com
│   │   └── html/                  # Raw HTML files for books
│   └── quotes/                    # Quotes data from quotes.toscrape.com
│       └── html/                  # Raw HTML files for quotes
├── information_retrieval_demo.ipynb  # Jupyter notebook demonstrating the project
├── main.py                        # Main script for running the pipeline
├── README.md                      # Project overview and instructions
├── requirements.txt               # Python dependencies
├── results/                       # Directory for storing analysis results
│   ├── figures/                   # Visualizations and plots
│   └── models/                    # Trained models
├── setup.sh                       # Setup script for the project
├── src/                           # Source code directory
│   ├── __init__.py               # Package initialization
│   ├── analysis/                  # Data analysis modules
│   │   ├── __init__.py           # Package initialization
│   │   ├── classification.py     # Classification algorithms
│   │   └── clustering.py         # Clustering algorithms
│   ├── preprocessing/             # Text preprocessing modules
│   │   ├── __init__.py           # Package initialization
│   │   └── text_processor.py     # Text processing utilities
│   ├── scraper/                   # Web scraping modules
│   │   ├── __init__.py           # Package initialization
│   │   ├── base_scraper.py       # Base scraper class
│   │   ├── books_scraper.py      # Books scraper implementation
│   │   └── quotes_scraper.py     # Quotes scraper implementation
│   └── visualization/             # Visualization modules
│       ├── __init__.py           # Package initialization
│       └── visualizer.py         # Visualization utilities
└── test_project.py               # Test script for the project
```

## Module Descriptions

### Scraper Module

- **base_scraper.py**: Defines the `BaseScraper` abstract class with common scraping functionality.
- **books_scraper.py**: Implements the `BooksScraper` class for scraping book data from books.toscrape.com.
- **quotes_scraper.py**: Implements the `QuotesScraper` class for scraping quote data from quotes.toscrape.com.

### Preprocessing Module

- **text_processor.py**: Implements the `TextProcessor` class for text cleaning, normalization, and vectorization.

### Analysis Module

- **clustering.py**: Implements the `ClusteringAnalyzer` class for performing clustering analysis.
- **classification.py**: Implements the `ClassificationAnalyzer` class for performing classification analysis.

### Visualization Module

- **visualizer.py**: Implements the `Visualizer` class for creating various visualizations.

## Usage

### Setup

To set up the project, run the setup script:

```bash
./setup.sh
```

This will create a virtual environment, install dependencies, and create necessary directories.

### Command-Line Interface

The project provides a command-line interface for running the information retrieval pipeline:

```bash
./cli.py --pages 2 --clusters 3
```

Options:
- `--pages`: Number of pages to scrape (default: 2)
- `--clusters`: Number of clusters for clustering (default: 3)
- `--skip-scraping`: Skip scraping if data already exists
- `--skip-visualization`: Skip visualization generation
- `--skip-saving`: Skip saving results to CSV

### Main Script

Alternatively, you can run the main script directly:

```bash
python main.py --pages 2 --skip-scraping
```

### Jupyter Notebook

The project includes a Jupyter notebook that demonstrates the information retrieval pipeline:

```bash
jupyter notebook information_retrieval_demo.ipynb
```

### Testing

To test the project components, run the test script:

```bash
./test_project.py
```

## Data Flow

1. **Web Scraping**: The system crawls web pages from books.toscrape.com and quotes.toscrape.com to collect data.
2. **Text Preprocessing**: The collected text data is cleaned, normalized, and vectorized.
3. **Data Analysis**: The preprocessed data is analyzed using clustering and classification algorithms.
4. **Visualization**: The analysis results are visualized using various plots and charts.
5. **Results Storage**: The analysis results are saved to CSV files and visualizations are saved as image files.

## Extending the Project

### Adding a New Data Source

To add a new data source, create a new scraper class that inherits from `BaseScraper` and implements the required methods:

```python
from src.scraper.base_scraper import BaseScraper

class NewScraper(BaseScraper):
    def __init__(self):
        super().__init__(base_url="https://example.com")
    
    def scrape_pages(self, num_pages):
        # Implementation
        pass
    
    def extract_content(self, html):
        # Implementation
        pass
```

### Adding a New Analysis Method

To add a new analysis method, extend the appropriate analyzer class or create a new one:

```python
from src.analysis.clustering import ClusteringAnalyzer

class EnhancedClusteringAnalyzer(ClusteringAnalyzer):
    def cluster_new_method(self, X):
        # Implementation
        pass
```

### Adding a New Visualization

To add a new visualization, extend the `Visualizer` class:

```python
from src.visualization.visualizer import Visualizer

class EnhancedVisualizer(Visualizer):
    def plot_new_visualization(self, data):
        # Implementation
        pass
```

## Dependencies

The project depends on the following Python libraries:

- beautifulsoup4: For parsing HTML
- requests: For making HTTP requests
- numpy: For numerical operations
- pandas: For data manipulation
- scikit-learn: For machine learning algorithms
- matplotlib: For creating visualizations
- wordcloud: For generating word clouds
- seaborn: For enhanced visualizations
- tqdm: For progress bars

See `requirements.txt` for specific versions.