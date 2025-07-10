#!/usr/bin/env python3

"""
Command-line interface for the information retrieval project.
"""

import argparse
import os
import sys
import time
from typing import List, Dict, Any, Optional

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from src.scraper.books_scraper import BooksScraper
from src.scraper.quotes_scraper import QuotesScraper
from src.scraper.selenium_books_scraper import SeleniumBooksScraper
from src.scraper.selenium_quotes_scraper import SeleniumQuotesScraper
from src.preprocessing.text_processor import TextProcessor
from src.analysis.clustering import ClusteringAnalyzer
from src.analysis.classification import ClassificationAnalyzer
from src.visualization.visualizer import Visualizer


def setup_directories() -> None:
    """Create necessary directories for the project."""
    directories = [
        'data',
        'data/books',
        'data/books/html',
        'data/quotes',
        'data/quotes/html',
        'results',
        'results/figures',
        'results/models'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")


def scrape_data(pages: int, skip_scraping: bool, use_selenium: bool = False, headless: bool = True, chromedriver_path: str = None, offline_mode: bool = False, html_dir: str = None) -> Dict[str, List[Dict[str, Any]]]:
    """Scrape data from websites.
    
    Args:
        pages: Number of pages to scrape
        skip_scraping: Whether to skip scraping if data already exists
        use_selenium: Whether to use Selenium for JavaScript-enabled scraping
        headless: Whether to run the browser in headless mode (only applies when use_selenium is True)
        chromedriver_path: Optional path to ChromeDriver executable (useful when offline)
        offline_mode: Whether to use local HTML files instead of fetching from the web
        html_dir: Directory containing local HTML files for offline mode
        
    Returns:
        Dictionary containing scraped books and quotes data
    """
    data = {}
    
    # Check if data already exists
    books_file = 'data/books/books_data.json'
    quotes_file = 'data/quotes/quotes_data.json'
    
    if skip_scraping and os.path.exists(books_file) and os.path.exists(quotes_file):
        print("Using existing scraped data...")
        return {
            'books': BooksScraper().load_data(books_file),
            'quotes': QuotesScraper().load_data(quotes_file)
        }
    
    # Create HTML directory if in offline mode and it doesn't exist
    if offline_mode and html_dir:
        os.makedirs(html_dir, exist_ok=True)
    
    chromedriver_path = os.getenv("CHROME_DRIVER")    
    
    # Scrape books
    print("\nScraping books data...")
    if use_selenium:
        print("Using Selenium with JavaScript support for books scraping")
        if chromedriver_path:
            print(f"Using ChromeDriver from: {chromedriver_path}")
        books_scraper = SeleniumBooksScraper(headless=headless, chromedriver_path=chromedriver_path, 
                                           offline_mode=offline_mode, html_dir=html_dir)
    else:
        books_scraper = BooksScraper(offline_mode=offline_mode, html_dir=html_dir)
    
    books_data = books_scraper.scrape_pages(pages)
    books_scraper.save_data(books_data, books_file)
    data['books'] = books_data
    print(f"Scraped {len(books_data)} books")
    
    # Scrape quotes
    print("\nScraping quotes data...")
    if use_selenium:
        print("Using Selenium with JavaScript support for quotes scraping")
        quotes_scraper = SeleniumQuotesScraper(headless=headless, chromedriver_path=chromedriver_path,
                                             offline_mode=offline_mode, html_dir=html_dir)
    else:
        quotes_scraper = QuotesScraper(offline_mode=offline_mode, html_dir=html_dir)
    
    quotes_data = quotes_scraper.scrape_pages(pages)
    quotes_scraper.save_data(quotes_data, quotes_file)
    data['quotes'] = quotes_data
    print(f"Scraped {len(quotes_data)} quotes")
    
    return data


def preprocess_data(data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Preprocess the scraped data.
    
    Args:
        data: Dictionary containing scraped books and quotes data
        
    Returns:
        Dictionary containing preprocessed data
    """
    print("\nPreprocessing data...")
    processor = TextProcessor()
    
    # Preprocess books data
    books_texts = [book['description'] for book in data['books'] if 'description' in book]
    books_categories = [book['category'] for book in data['books'] if 'category' in book]
    
    # Preprocess quotes data
    quotes_texts = [quote['text'] for quote in data['quotes'] if 'text' in quote]
    quotes_authors = [quote['author'] for quote in data['quotes'] if 'author' in quote]
    
    # Preprocess texts
    print("Preprocessing book descriptions...")
    books_processed = processor.preprocess_documents(books_texts)
    print("Preprocessing quotes...")
    quotes_processed = processor.preprocess_documents(quotes_texts)
    
    # Vectorize texts
    print("Vectorizing book descriptions...")
    books_vectors, books_feature_names = processor.vectorize_documents(books_processed)
    print("Vectorizing quotes...")
    quotes_vectors, quotes_feature_names = processor.vectorize_documents(quotes_processed)
    
    return {
        'books_texts': books_texts,
        'books_processed': books_processed,
        'books_vectors': books_vectors,
        'books_feature_names': books_feature_names,
        'books_categories': books_categories,
        'quotes_texts': quotes_texts,
        'quotes_processed': quotes_processed,
        'quotes_vectors': quotes_vectors,
        'quotes_feature_names': quotes_feature_names,
        'quotes_authors': quotes_authors
    }


def analyze_data(preprocessed_data: Dict[str, Any], n_clusters: int) -> Dict[str, Any]:
    """Analyze the preprocessed data using clustering and classification.
    
    Args:
        preprocessed_data: Dictionary containing preprocessed data
        n_clusters: Number of clusters for clustering
        
    Returns:
        Dictionary containing analysis results
    """
    results = {}
    
    # Clustering
    print("\nPerforming clustering analysis...")
    clustering = ClusteringAnalyzer(n_clusters=n_clusters)
    
    # Cluster books
    print("Clustering book descriptions...")
    books_clusters = clustering.cluster_kmeans(preprocessed_data['books_vectors'])
    results['books_clusters'] = books_clusters
    
    # Get silhouette score for books clustering
    books_silhouette = clustering.evaluate_silhouette(preprocessed_data['books_vectors'])
    results['books_clustering_metrics'] = {
        'silhouette_score': books_silhouette,
        'cluster_distribution': clustering.get_cluster_distribution()
    }
    print(f"Books clustering silhouette score: {books_silhouette:.4f}")
    print(f"Books cluster distribution: {clustering.get_cluster_distribution()}")
    
    # Cluster quotes
    print("Clustering quotes...")
    quotes_clusters = clustering.cluster_kmeans(preprocessed_data['quotes_vectors'])
    results['quotes_clusters'] = quotes_clusters
    
    # Get silhouette score for quotes clustering
    quotes_silhouette = clustering.evaluate_silhouette(preprocessed_data['quotes_vectors'])
    results['quotes_clustering_metrics'] = {
        'silhouette_score': quotes_silhouette,
        'cluster_distribution': clustering.get_cluster_distribution()
    }
    print(f"Quotes clustering silhouette score: {quotes_silhouette:.4f}")
    print(f"Quotes cluster distribution: {clustering.get_cluster_distribution()}")
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters for books...")
    optimal_books_clusters, books_inertia = clustering.get_optimal_clusters(preprocessed_data['books_vectors'])
    results['optimal_books_clusters'] = optimal_books_clusters
    print(f"Optimal number of clusters for books: {optimal_books_clusters}")
    
    print("Finding optimal number of clusters for quotes...")
    optimal_quotes_clusters, quotes_inertia = clustering.get_optimal_clusters(preprocessed_data['quotes_vectors'])
    results['optimal_quotes_clusters'] = optimal_quotes_clusters
    print(f"Optimal number of clusters for quotes: {optimal_quotes_clusters}")
    
    
    # Classification
    print("\nPerforming classification analysis...")
    classification = ClassificationAnalyzer()
    
    # Classify books by category
    if len(set(preprocessed_data['books_categories'])) > 1:
        print("Classifying books by category...")
        books_X = preprocessed_data['books_vectors']
        books_y = preprocessed_data['books_categories']
        
        # Check if there are enough samples per class for classification
        class_counts = {}
        for category in books_y:
            class_counts[category] = class_counts.get(category, 0) + 1
        
        min_samples_per_class = min(class_counts.values())
        
        if min_samples_per_class < 2:
            print(f"Warning: Some categories have only {min_samples_per_class} sample, which is too few for reliable classification.")
            print("Classification will proceed with adjusted settings, but results may not be statistically significant.")
        
        # Compare all classification models
        try:
            print("Comparing classification models for books...")
            books_model_comparison = classification.compare_models(books_X, books_y)
            results['books_model_comparison'] = books_model_comparison
            
            # Print model comparison results
            print("\nBooks classification model comparison:")
            for model_name, metrics in books_model_comparison.items():
                print(f"  {model_name.replace('_', ' ').title()}:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
            
            # Train and evaluate models with detailed metrics
            classification.train(books_X, books_y, model_name='naive_bayes')
            books_nb_metrics = classification.evaluate(books_X, books_y)
            
            classification.train(books_X, books_y, model_name='svm')
            books_svm_metrics = classification.evaluate(books_X, books_y)
            
            # Cross-validation
            books_nb_cv = classification.cross_validate(books_X, books_y)
            books_svm_cv = classification.cross_validate(books_X, books_y)
            
            results['books_classification'] = {
                'naive_bayes': {
                    'metrics': books_nb_metrics,
                    'cross_validation': books_nb_cv
                },
                'svm': {
                    'metrics': books_svm_metrics,
                    'cross_validation': books_svm_cv
                }
            }
            
            # Print detailed metrics
            print("\nBooks classification detailed metrics:")
            print(f"  Naive Bayes: accuracy={books_nb_metrics['accuracy']:.4f}, "
                  f"precision={books_nb_metrics['precision']:.4f}, "
                  f"recall={books_nb_metrics['recall']:.4f}, "
                  f"f1={books_nb_metrics['f1']:.4f}")
            print(f"  SVM: accuracy={books_svm_metrics['accuracy']:.4f}, "
                  f"precision={books_svm_metrics['precision']:.4f}, "
                  f"recall={books_svm_metrics['recall']:.4f}, "
                  f"f1={books_svm_metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error during books classification: {str(e)}")
            print("Skipping books classification due to error.")
            results['books_classification_error'] = str(e)
    
    # Classify quotes by author
    if len(set(preprocessed_data['quotes_authors'])) > 1:
        print("\nClassifying quotes by author...")
        quotes_X = preprocessed_data['quotes_vectors']
        quotes_y = preprocessed_data['quotes_authors']
        
        # Check if there are enough samples per class for classification
        class_counts = {}
        for author in quotes_y:
            class_counts[author] = class_counts.get(author, 0) + 1
        
        min_samples_per_class = min(class_counts.values())
        
        if min_samples_per_class < 2:
            print(f"Warning: Some authors have only {min_samples_per_class} quote, which is too few for reliable classification.")
            print("Classification will proceed with adjusted settings, but results may not be statistically significant.")
        
        # Compare all classification models
        try:
            print("Comparing classification models for quotes...")
            quotes_model_comparison = classification.compare_models(quotes_X, quotes_y)
            results['quotes_model_comparison'] = quotes_model_comparison
            
            # Print model comparison results
            print("\nQuotes classification model comparison:")
            for model_name, metrics in quotes_model_comparison.items():
                print(f"  {model_name.replace('_', ' ').title()}:")
                for metric_name, value in metrics.items():
                    print(f"    {metric_name}: {value:.4f}")
            
            # Train and evaluate models with detailed metrics
            classification.train(quotes_X, quotes_y, model_name='naive_bayes')
            quotes_nb_metrics = classification.evaluate(quotes_X, quotes_y)
            
            classification.train(quotes_X, quotes_y, model_name='svm')
            quotes_svm_metrics = classification.evaluate(quotes_X, quotes_y)
            
            # Cross-validation
            quotes_nb_cv = classification.cross_validate(quotes_X, quotes_y)
            quotes_svm_cv = classification.cross_validate(quotes_X, quotes_y)
            
            results['quotes_classification'] = {
                'naive_bayes': {
                    'metrics': quotes_nb_metrics,
                    'cross_validation': quotes_nb_cv
                },
                'svm': {
                    'metrics': quotes_svm_metrics,
                    'cross_validation': quotes_svm_cv
                }
            }
            
            # Print detailed metrics
            print("\nQuotes classification detailed metrics:")
            print(f"  Naive Bayes: accuracy={quotes_nb_metrics['accuracy']:.4f}, "
                  f"precision={quotes_nb_metrics['precision']:.4f}, "
                  f"recall={quotes_nb_metrics['recall']:.4f}, "
                  f"f1={quotes_nb_metrics['f1']:.4f}")
            print(f"  SVM: accuracy={quotes_svm_metrics['accuracy']:.4f}, "
                  f"precision={quotes_svm_metrics['precision']:.4f}, "
                  f"recall={quotes_svm_metrics['recall']:.4f}, "
                  f"f1={quotes_svm_metrics['f1']:.4f}")
        except Exception as e:
            print(f"Error during quotes classification: {str(e)}")
            print("Skipping quotes classification due to error.")
            results['quotes_classification_error'] = str(e)

    
    return results


def visualize_results(data: Dict[str, List[Dict[str, Any]]], 
                     preprocessed_data: Dict[str, Any], 
                     analysis_results: Dict[str, Any]) -> None:
    """Create visualizations for the analysis results.
    
    Args:
        data: Dictionary containing scraped data
        preprocessed_data: Dictionary containing preprocessed data
        analysis_results: Dictionary containing analysis results
    """
    print("\nCreating visualizations...")
    visualizer = Visualizer()
    
    # Create word clouds for book clusters
    if 'books_clusters' in analysis_results:
        print("Creating word clouds for book clusters...")
        books_texts = preprocessed_data['books_processed']
        books_clusters = analysis_results['books_clusters']
        
        for cluster_id in set(books_clusters):
            cluster_texts = [text for i, text in enumerate(books_texts) if books_clusters[i] == cluster_id]
            if cluster_texts:
                cluster_text = ' '.join(cluster_texts)
                fig = visualizer.create_wordcloud(cluster_text, title=f"Book Cluster {cluster_id}")
                fig.savefig(f"results/figures/books_cluster_{cluster_id}_wordcloud.png")
    
    # Create word clouds for quote clusters
    if 'quotes_clusters' in analysis_results:
        print("Creating word clouds for quote clusters...")
        quotes_texts = preprocessed_data['quotes_processed']
        quotes_clusters = analysis_results['quotes_clusters']
        
        for cluster_id in set(quotes_clusters):
            cluster_texts = [text for i, text in enumerate(quotes_texts) if quotes_clusters[i] == cluster_id]
            if cluster_texts:
                cluster_text = ' '.join(cluster_texts)
                fig = visualizer.create_wordcloud(cluster_text, title=f"Quote Cluster {cluster_id}")
                fig.savefig(f"results/figures/quotes_cluster_{cluster_id}_wordcloud.png")
    
    # Create PCA visualization for books
    if 'books_vectors' in preprocessed_data and 'books_clusters' in analysis_results:
        print("Creating PCA visualization for books...")
        books_vectors = preprocessed_data['books_vectors']
        books_clusters = analysis_results['books_clusters']
        
        fig = visualizer.plot_document_similarity(books_vectors, books_clusters, method='pca', 
                                               title="Book Clusters (PCA)")
        fig.savefig("results/figures/books_clusters_pca.png")
    
    # Create PCA visualization for quotes
    if 'quotes_vectors' in preprocessed_data and 'quotes_clusters' in analysis_results:
        print("Creating PCA visualization for quotes...")
        quotes_vectors = preprocessed_data['quotes_vectors']
        quotes_clusters = analysis_results['quotes_clusters']
        
        fig = visualizer.plot_document_similarity(quotes_vectors, quotes_clusters, method='pca', 
                                               title="Quote Clusters (PCA)")
        fig.savefig("results/figures/quotes_clusters_pca.png")
    
    # Create classification evaluation plots
    if 'books_classification' in analysis_results:
        print("Creating classification evaluation plot for books...")
        models = ['Naive Bayes', 'SVM']
        metrics = {}
        
        # Extract metrics from the analysis results
        for model_name in ['naive_bayes', 'svm']:
            if model_name in analysis_results['books_classification']:
                model_data = analysis_results['books_classification'][model_name]
                display_name = model_name.replace('_', ' ').title()
                metrics[display_name] = {
                    'Accuracy': model_data['metrics']['accuracy'],
                    'Precision': model_data['metrics']['precision'],
                    'Recall': model_data['metrics']['recall'],
                    'F1 Score': model_data['metrics']['f1']
                }
        
        fig = visualizer.plot_evaluation_metrics(metrics, title="Book Classification Performance")
        fig.savefig("results/figures/books_classification_performance.png")
    elif 'books_classification_error' in analysis_results:
        print(f"Skipping books classification visualization due to error: {analysis_results['books_classification_error']}")
    
    if 'quotes_classification' in analysis_results:
        print("Creating classification evaluation plot for quotes...")
        models = ['Naive Bayes', 'SVM']
        metrics = {}
        
        # Extract metrics from the analysis results
        for model_name in ['naive_bayes', 'svm']:
            if model_name in analysis_results['quotes_classification']:
                model_data = analysis_results['quotes_classification'][model_name]
                display_name = model_name.replace('_', ' ').title()
                metrics[display_name] = {
                    'Accuracy': model_data['metrics']['accuracy'],
                    'Precision': model_data['metrics']['precision'],
                    'Recall': model_data['metrics']['recall'],
                    'F1 Score': model_data['metrics']['f1']
                }
        
        fig = visualizer.plot_evaluation_metrics(metrics, title="Quote Classification Performance")
        fig.savefig("results/figures/quotes_classification_performance.png")
    elif 'quotes_classification_error' in analysis_results:
        print(f"Skipping quotes classification visualization due to error: {analysis_results['quotes_classification_error']}")
    
    print("Visualizations saved to 'results/figures/'")


def save_results(data: Dict[str, List[Dict[str, Any]]], 
                analysis_results: Dict[str, Any]) -> None:
    """Save analysis results to CSV files.
    
    Args:
        data: Dictionary containing scraped data
        analysis_results: Dictionary containing analysis results
    """
    import pandas as pd
    import json
    import numpy as np
    
    print("\nSaving results to CSV files...")
    
    # Save books with cluster labels
    if 'books_clusters' in analysis_results:
        books_df = pd.DataFrame(data['books'])
        books_df['cluster'] = analysis_results['books_clusters']
        books_df.to_csv('results/books_with_clusters.csv', index=False)
        print("Saved books with cluster labels to 'results/books_with_clusters.csv'")
    
    # Save quotes with cluster labels
    if 'quotes_clusters' in analysis_results:
        quotes_df = pd.DataFrame(data['quotes'])
        quotes_df['cluster'] = analysis_results['quotes_clusters']
        quotes_df.to_csv('results/quotes_with_clusters.csv', index=False)
        print("Saved quotes with cluster labels to 'results/quotes_with_clusters.csv'")
    
    # Save clustering metrics to JSON
    clustering_metrics = {}
    if 'books_clustering_metrics' in analysis_results:
        books_metrics = analysis_results['books_clustering_metrics'].copy()
        # Convert numpy int32 keys to regular Python integers for JSON serialization
        if 'cluster_distribution' in books_metrics:
            books_metrics['cluster_distribution'] = {int(k): int(v) for k, v in books_metrics['cluster_distribution'].items()}
        clustering_metrics['books'] = books_metrics
    
    if 'quotes_clustering_metrics' in analysis_results:
        quotes_metrics = analysis_results['quotes_clustering_metrics'].copy()
        # Convert numpy int32 keys to regular Python integers for JSON serialization
        if 'cluster_distribution' in quotes_metrics:
            quotes_metrics['cluster_distribution'] = {int(k): int(v) for k, v in quotes_metrics['cluster_distribution'].items()}
        clustering_metrics['quotes'] = quotes_metrics
    
    if clustering_metrics:
        with open('results/clustering_metrics.json', 'w') as f:
            json.dump(clustering_metrics, f, indent=2)
        print("Saved clustering metrics to 'results/clustering_metrics.json'")
    
    # Save classification metrics
    classification_metrics = {}
    
    # Books classification metrics
    if 'books_classification' in analysis_results:
        classification_metrics['books'] = analysis_results['books_classification']
        
        # Create a DataFrame for books classification metrics
        books_metrics = []
        for model, data in analysis_results['books_classification'].items():
            model_metrics = data['metrics']
            model_metrics['model'] = model
            books_metrics.append(model_metrics)
        
        books_metrics_df = pd.DataFrame(books_metrics)
        books_metrics_df.to_csv('results/books_classification_metrics.csv', index=False)
        print("Saved books classification metrics to 'results/books_classification_metrics.csv'")
    elif 'books_classification_error' in analysis_results:
        # Save error information if classification failed
        with open('results/books_classification_error.txt', 'w') as f:
            f.write(analysis_results['books_classification_error'])
        print("Saved books classification error to 'results/books_classification_error.txt'")
    
    # Quotes classification metrics
    if 'quotes_classification' in analysis_results:
        classification_metrics['quotes'] = analysis_results['quotes_classification']
        
        # Create a DataFrame for quotes classification metrics
        quotes_metrics = []
        for model, data in analysis_results['quotes_classification'].items():
            model_metrics = data['metrics']
            model_metrics['model'] = model
            quotes_metrics.append(model_metrics)
        
        quotes_metrics_df = pd.DataFrame(quotes_metrics)
        quotes_metrics_df.to_csv('results/quotes_classification_metrics.csv', index=False)
        print("Saved quotes classification metrics to 'results/quotes_classification_metrics.csv'")
    elif 'quotes_classification_error' in analysis_results:
        # Save error information if classification failed
        with open('results/quotes_classification_error.txt', 'w') as f:
            f.write(analysis_results['quotes_classification_error'])
        print("Saved quotes classification error to 'results/quotes_classification_error.txt'")
    
    # Save model comparison results
    if 'books_model_comparison' in analysis_results:
        books_comparison = []
        for model, metrics in analysis_results['books_model_comparison'].items():
            model_data = metrics.copy()
            model_data['model'] = model
            books_comparison.append(model_data)
        
        books_comparison_df = pd.DataFrame(books_comparison)
        books_comparison_df.to_csv('results/books_model_comparison.csv', index=False)
        print("Saved books model comparison to 'results/books_model_comparison.csv'")
    
    if 'quotes_model_comparison' in analysis_results:
        quotes_comparison = []
        for model, metrics in analysis_results['quotes_model_comparison'].items():
            model_data = metrics.copy()
            model_data['model'] = model
            quotes_comparison.append(model_data)
        
        quotes_comparison_df = pd.DataFrame(quotes_comparison)
        quotes_comparison_df.to_csv('results/quotes_model_comparison.csv', index=False)
        print("Saved quotes model comparison to 'results/quotes_model_comparison.csv'")
    
    # Save all analysis results in one JSON file
    # with open('results/analysis_results.json', 'w') as f:
    #     # Convert numpy arrays to lists for JSON serialization
    #     serializable_results = {}
    #     for key, value in analysis_results.items():
    #         if key.endswith('_clusters') and hasattr(value, 'tolist'):
    #             serializable_results[key] = value.tolist()
    #         else:
    #             serializable_results[key] = value
        
        # Handle nested numpy arrays and other non-serializable objects
        def json_serializer(obj):
            """Custom JSON serializer for objects not serializable by default json code"""
            if isinstance(obj, dict):
                # Convert dictionary with numpy keys to regular Python types
                return {(int(k) if isinstance(k, np.integer) else k): v for k, v in obj.items()}
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # For numpy scalar types
                return obj.item()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            return str(obj)
        
        # json.dump(serializable_results, f, indent=2, default=json_serializer)
    
    print("Saved complete analysis results to 'results/analysis_results.json'")


def main() -> None:
    """Main function to run the information retrieval pipeline."""
    parser = argparse.ArgumentParser(description="Information Retrieval Project CLI")
    parser.add_argument('--pages', type=int, default=2, help="Number of pages to scrape (default: 2)")
    parser.add_argument('--clusters', type=int, default=3, help="Number of clusters for clustering (default: 3)")
    parser.add_argument('--skip-scraping', action='store_true', help="Skip scraping if data already exists")
    parser.add_argument('--skip-visualization', action='store_true', help="Skip visualization generation")
    parser.add_argument('--skip-saving', action='store_true', help="Skip saving results to CSV")
    parser.add_argument('--use-selenium', action='store_true', help="Use Selenium for JavaScript-enabled scraping")
    parser.add_argument('--no-headless', action='store_true', help="Run Selenium in non-headless mode (shows browser UI)")
    parser.add_argument('--chromedriver-path', type=str, help="Path to ChromeDriver executable (useful when offline)")
    parser.add_argument('--offline-mode', action='store_true', help="Use local HTML files instead of fetching from the web")
    parser.add_argument('--html-dir', type=str, help="Directory containing local HTML files for offline mode")
    
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Setup directories
    setup_directories()
    
    # Scrape data
    data = scrape_data(
        pages=args.pages,
        skip_scraping=args.skip_scraping,
        use_selenium=args.use_selenium,
        headless=not args.no_headless,
        chromedriver_path=args.chromedriver_path,
        offline_mode=args.offline_mode,
        html_dir=args.html_dir
    )
    
    # Preprocess data
    preprocessed_data = preprocess_data(data)
    
    # Analyze data
    analysis_results = analyze_data(preprocessed_data, args.clusters)
    
    # Visualize results
    if not args.skip_visualization:
        visualize_results(data, preprocessed_data, analysis_results)
    
    # Save results
    if not args.skip_saving:
        save_results(data, analysis_results)
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")
    print("\nInformation retrieval pipeline completed successfully!")


if __name__ == "__main__":
    main()