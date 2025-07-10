#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
from sklearn.model_selection import train_test_split

# Import project modules
from src.scraper.books_scraper import BooksScraper
from src.scraper.quotes_scraper import QuotesScraper
from src.preprocessing.text_processor import TextProcessor
from src.analysis.clustering import ClusteringAnalyzer
from src.analysis.classification import ClassificationAnalyzer
from src.visualization.visualizer import Visualizer


def setup_directories():
    """
    Create necessary directories for the project.
    """
    directories = [
        "data",
        "data/books",
        "data/books/html",
        "data/quotes",
        "data/quotes/html",
        "results",
        "results/figures",
        "results/models"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def scrape_data(num_pages: int = 1) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Scrape data from books and quotes websites.
    
    Args:
        num_pages: Number of pages to scrape from each website
        
    Returns:
        Tuple of (books data, quotes data)
    """
    print("\n=== Scraping Books ===\n")
    books_scraper = BooksScraper()
    books_data = books_scraper.scrape_pages(num_pages)
    print(f"Scraped {len(books_data)} books.\n")
    
    print("\n=== Scraping Quotes ===\n")
    quotes_scraper = QuotesScraper()
    quotes_data = quotes_scraper.scrape_pages(num_pages)
    print(f"Scraped {len(quotes_data)} quotes.\n")
    
    return books_data, quotes_data


def preprocess_data(books_data: List[Dict[str, Any]], quotes_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Preprocess the scraped data.
    
    Args:
        books_data: List of dictionaries containing book data
        quotes_data: List of dictionaries containing quote data
        
    Returns:
        Tuple of (preprocessed books data, preprocessed quotes data)
    """
    print("\n=== Preprocessing Books Data ===\n")
    text_processor = TextProcessor()
    processed_books = text_processor.preprocess_data(books_data, 'description')
    print(f"Preprocessed {len(processed_books)} books.\n")
    
    print("\n=== Preprocessing Quotes Data ===\n")
    processed_quotes = text_processor.preprocess_data(quotes_data, 'text')
    print(f"Preprocessed {len(processed_quotes)} quotes.\n")
    
    return processed_books, processed_quotes


def cluster_books(books_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, ClusteringAnalyzer]:
    """
    Cluster books based on their descriptions.
    
    Args:
        books_data: List of dictionaries containing book data
        
    Returns:
        Tuple of (cluster labels, clustering analyzer)
    """
    print("\n=== Clustering Books ===\n")
    
    # Extract preprocessed descriptions
    descriptions = [book['processed_description'] for book in books_data]
    
    # Vectorize descriptions
    text_processor = TextProcessor()
    X, vectorizer = text_processor.vectorize_tfidf(descriptions)
    
    # Find optimal number of clusters
    clustering = ClusteringAnalyzer()
    optimal_clusters, _ = clustering.get_optimal_clusters(X, max_clusters=10)
    print(f"Optimal number of clusters for books: {optimal_clusters}\n")
    
    # Create clustering analyzer with optimal number of clusters
    clustering = ClusteringAnalyzer(n_clusters=optimal_clusters)
    
    # Perform clustering
    labels = clustering.cluster_kmeans(X)
    print(f"Clustered {len(labels)} books into {len(set(labels))} clusters.\n")
    
    # Get cluster distribution
    distribution = clustering.get_cluster_distribution()
    print("Cluster distribution:")
    for cluster, count in distribution.items():
        print(f"Cluster {cluster}: {count} books")
    
    # Visualize clusters
    fig = clustering.visualize_clusters(X, title="Book Clusters")
    fig.savefig("results/figures/book_clusters.png")
    plt.close(fig)  # Close figure to prevent too many open figures
    
    return labels, clustering


def cluster_quotes(quotes_data: List[Dict[str, Any]]) -> Tuple[np.ndarray, ClusteringAnalyzer]:
    """
    Cluster quotes based on their text.
    
    Args:
        quotes_data: List of dictionaries containing quote data
        
    Returns:
        Tuple of (cluster labels, clustering analyzer)
    """
    print("\n=== Clustering Quotes ===\n")
    
    # Extract preprocessed quotes
    quotes_text = [quote['processed_text'] for quote in quotes_data]
    
    # Vectorize quotes
    text_processor = TextProcessor()
    X, vectorizer = text_processor.vectorize_tfidf(quotes_text)
    
    # Find optimal number of clusters
    clustering = ClusteringAnalyzer()
    optimal_clusters, _ = clustering.get_optimal_clusters(X, max_clusters=10)
    print(f"Optimal number of clusters for quotes: {optimal_clusters}\n")
    
    # Create clustering analyzer with optimal number of clusters
    clustering = ClusteringAnalyzer(n_clusters=optimal_clusters)
    
    # Perform clustering
    labels = clustering.cluster_kmeans(X)
    print(f"Clustered {len(labels)} quotes into {len(set(labels))} clusters.\n")
    
    # Get cluster distribution
    distribution = clustering.get_cluster_distribution()
    print("Cluster distribution:")
    for cluster, count in distribution.items():
        print(f"Cluster {cluster}: {count} quotes")
    
    # Visualize clusters
    fig = clustering.visualize_clusters(X, title="Quote Clusters")
    fig.savefig("results/figures/quote_clusters.png")
    plt.close(fig)  # Close figure to prevent too many open figures
    
    return labels, clustering


def classify_books(books_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], ClassificationAnalyzer]:
    """
    Classify books based on their descriptions.
    
    Args:
        books_data: List of dictionaries containing book data
        
    Returns:
        Tuple of (evaluation metrics, classification analyzer)
    """
    print("\n=== Classifying Books ===\n")
    
    # Extract preprocessed descriptions and categories
    descriptions = [book['processed_description'] for book in books_data]
    categories = [book['category'] for book in books_data]
    
    # Check if we have enough data for classification
    category_counts = {}
    for category in categories:
        if category in category_counts:
            category_counts[category] += 1
        else:
            category_counts[category] = 1
    
    # Filter out categories with only one instance
    valid_indices = [i for i, category in enumerate(categories) if category_counts[category] >= 2]
    
    # If we don't have enough data for classification, return empty results
    if len(valid_indices) < 2:
        print("Not enough data for classification. Each category needs at least 2 instances.")
        return {}, ClassificationAnalyzer()
    
    # Filter data to only include categories with multiple instances
    filtered_descriptions = [descriptions[i] for i in valid_indices]
    filtered_categories = [categories[i] for i in valid_indices]
    
    # Vectorize descriptions
    text_processor = TextProcessor()
    X, vectorizer = text_processor.vectorize_tfidf(filtered_descriptions)
    
    # Split data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, filtered_categories, test_size=0.2, random_state=42, stratify=filtered_categories
        )
    except ValueError as e:
        print(f"Error in train_test_split: {e}")
        print("Proceeding without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, filtered_categories, test_size=0.2, random_state=42, stratify=None
        )
    
    # Create classification analyzer
    classifier = ClassificationAnalyzer()
    
    # Compare different models
    metrics = classifier.compare_models(X, filtered_categories)
    print("Model comparison:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}:")
        for metric_name, metric_value in model_metrics.items():
            print(f"  {metric_name}: {metric_value:.3f}")
    
    # Train the best model (based on F1 score)
    best_model = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest model: {best_model}\n")
    classifier.train(X_train, y_train, model_name=best_model)
    
    # Evaluate on test set
    test_metrics = classifier.evaluate(X_test, y_test)
    print("Test set evaluation:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.3f}")
    
    # Plot confusion matrix
    fig = classifier.plot_confusion_matrix(X_test, y_test, title="Book Classification Confusion Matrix")
    fig.savefig("results/figures/book_confusion_matrix.png")
    plt.close(fig)  # Close figure to prevent too many open figures
    
    return metrics, classifier


def classify_quotes(quotes_data: List[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, float]], ClassificationAnalyzer]:
    """
    Classify quotes based on their tags.
    
    Args:
        quotes_data: List of dictionaries containing quote data
        
    Returns:
        Tuple of (evaluation metrics, classification analyzer)
    """
    print("\n=== Classifying Quotes ===\n")
    
    # Extract preprocessed quotes and primary tags
    quotes_text = [quote['processed_text'] for quote in quotes_data]
    
    # Use the first tag as the target class
    primary_tags = [quote['tags'][0] if quote['tags'] else 'unknown' for quote in quotes_data]
    
    # Check if we have enough data for classification
    tag_counts = {}
    for tag in primary_tags:
        if tag in tag_counts:
            tag_counts[tag] += 1
        else:
            tag_counts[tag] = 1
    
    # Filter out tags with only one instance
    valid_indices = [i for i, tag in enumerate(primary_tags) if tag_counts[tag] >= 2]
    
    # If we don't have enough data for classification, return empty results
    if len(valid_indices) < 2:
        print("Not enough data for classification. Each tag needs at least 2 instances.")
        return {}, ClassificationAnalyzer()
    
    # Filter data to only include tags with multiple instances
    filtered_quotes_text = [quotes_text[i] for i in valid_indices]
    filtered_primary_tags = [primary_tags[i] for i in valid_indices]
    
    # Vectorize quotes
    text_processor = TextProcessor()
    X, vectorizer = text_processor.vectorize_tfidf(filtered_quotes_text)
    
    # Split data into training and testing sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, filtered_primary_tags, test_size=0.2, random_state=42, stratify=filtered_primary_tags
        )
    except ValueError as e:
        print(f"Error in train_test_split: {e}")
        print("Proceeding without stratification.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, filtered_primary_tags, test_size=0.2, random_state=42, stratify=None
        )
    
    # Create classification analyzer
    classifier = ClassificationAnalyzer()
    
    # Compare different models
    metrics = classifier.compare_models(X, filtered_primary_tags)
    print("Model comparison:")
    for model_name, model_metrics in metrics.items():
        print(f"{model_name}:")
        for metric_name, metric_value in model_metrics.items():
            print(f"  {metric_name}: {metric_value:.3f}")
    
    # Train the best model (based on F1 score)
    best_model = max(metrics.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest model: {best_model}\n")
    classifier.train(X_train, y_train, model_name=best_model)
    
    # Evaluate on test set
    test_metrics = classifier.evaluate(X_test, y_test)
    print("Test set evaluation:")
    for metric_name, metric_value in test_metrics.items():
        print(f"  {metric_name}: {metric_value:.3f}")
    
    # Plot confusion matrix
    fig = classifier.plot_confusion_matrix(X_test, y_test, title="Quote Classification Confusion Matrix")
    fig.savefig("results/figures/quote_confusion_matrix.png")
    plt.close(fig)  # Close figure to prevent too many open figures
    
    return metrics, classifier


def create_visualizations(books_data: List[Dict[str, Any]], quotes_data: List[Dict[str, Any]],
                         book_clusters: np.ndarray, quote_clusters: np.ndarray):
    """
    Create visualizations for the data.
    
    Args:
        books_data: List of dictionaries containing book data
        quotes_data: List of dictionaries containing quote data
        book_clusters: Cluster labels for books
        quote_clusters: Cluster labels for quotes
    """
    print("\n=== Creating Visualizations ===\n")
    
    visualizer = Visualizer()
    
    # Create word clouds for book clusters
    print("Creating word clouds for book clusters...")
    book_descriptions = [book['description'] for book in books_data]
    book_wordclouds = visualizer.create_class_wordclouds(
        book_descriptions, book_clusters, title_prefix="Book Cluster"
    )
    
    for cluster, fig in book_wordclouds.items():
        fig.savefig(f"results/figures/book_cluster_{cluster}_wordcloud.png")
        plt.close(fig)  # Close figure to prevent too many open figures
    
    # Create word clouds for quote clusters
    print("Creating word clouds for quote clusters...")
    quote_texts = [quote['text'] for quote in quotes_data]
    quote_wordclouds = visualizer.create_class_wordclouds(
        quote_texts, quote_clusters, title_prefix="Quote Cluster"
    )
    
    for cluster, fig in quote_wordclouds.items():
        fig.savefig(f"results/figures/quote_cluster_{cluster}_wordcloud.png")
        plt.close(fig)  # Close figure to prevent too many open figures
    
    # Create word clouds for book categories
    print("Creating word clouds for book categories...")
    book_categories = [book['category'] for book in books_data]
    category_wordclouds = visualizer.create_class_wordclouds(
        book_descriptions, book_categories, title_prefix="Book Category"
    )
    
    for category, fig in category_wordclouds.items():
        fig.savefig(f"results/figures/book_category_{category.replace(' ', '_')}_wordcloud.png")
        plt.close(fig)  # Close figure to prevent too many open figures
    
    # Create word clouds for quote tags
    print("Creating word clouds for quote tags...")
    primary_tags = [quote['tags'][0] if quote['tags'] else 'unknown' for quote in quotes_data]
    tag_wordclouds = visualizer.create_class_wordclouds(
        quote_texts, primary_tags, title_prefix="Quote Tag"
    )
    
    for tag, fig in tag_wordclouds.items():
        fig.savefig(f"results/figures/quote_tag_{tag.replace(' ', '_')}_wordcloud.png")
        plt.close(fig)  # Close figure to prevent too many open figures
    
    print("Visualizations created successfully.")


def save_results(books_data: List[Dict[str, Any]], quotes_data: List[Dict[str, Any]],
                book_clusters: np.ndarray, quote_clusters: np.ndarray):
    """
    Save the results to CSV files.
    
    Args:
        books_data: List of dictionaries containing book data
        quotes_data: List of dictionaries containing quote data
        book_clusters: Cluster labels for books
        quote_clusters: Cluster labels for quotes
    """
    print("\n=== Saving Results ===\n")
    
    # Add cluster labels to books data
    for i, book in enumerate(books_data):
        book['cluster'] = int(book_clusters[i])
    
    # Add cluster labels to quotes data
    for i, quote in enumerate(quotes_data):
        quote['cluster'] = int(quote_clusters[i])
    
    # Convert to DataFrames
    books_df = pd.DataFrame(books_data)
    quotes_df = pd.DataFrame(quotes_data)
    
    # Save to CSV
    books_df.to_csv("results/books_results.csv", index=False)
    quotes_df.to_csv("results/quotes_results.csv", index=False)
    
    print("Results saved successfully.")


def main():
    """
    Main function to run the information retrieval pipeline.
    """
    parser = argparse.ArgumentParser(description="Information Retrieval Project")
    parser.add_argument("--pages", type=int, default=10, help="Number of pages to scrape from each website")
    parser.add_argument("--skip-scraping", action="store_true", help="Skip the scraping step and use existing data")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip the clustering step")
    parser.add_argument("--skip-classification", action="store_true", help="Skip the classification step")
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    # Scrape data or load existing data
    if args.skip_scraping:
        print("Skipping scraping and loading existing data...")
        try:
            books_df = pd.read_csv("results/books_results.csv")
            quotes_df = pd.read_csv("results/quotes_results.csv")
            books_data = books_df.to_dict('records')
            quotes_data = quotes_df.to_dict('records')
        except FileNotFoundError:
            print("No existing data found. Scraping new data...")
            books_data, quotes_data = scrape_data(args.pages)
    else:
        books_data, quotes_data = scrape_data(args.pages)
    
    # Preprocess data
    processed_books, processed_quotes = preprocess_data(books_data, quotes_data)
    
    # Initialize default values
    book_clusters = np.zeros(len(processed_books), dtype=int)
    quote_clusters = np.zeros(len(processed_quotes), dtype=int)
    book_clustering = None
    quote_clustering = None
    book_classifier = None
    quote_classifier = None
    book_metrics = {}
    quote_metrics = {}
    
    # Clustering
    if not args.skip_clustering:
        try:
            book_clusters, book_clustering = cluster_books(processed_books)
        except Exception as e:
            print(f"Error during book clustering: {e}")
            print("Using default cluster assignments for books.")
        
        try:
            quote_clusters, quote_clustering = cluster_quotes(processed_quotes)
        except Exception as e:
            print(f"Error during quote clustering: {e}")
            print("Using default cluster assignments for quotes.")
    else:
        print("Skipping clustering step...")
    
    # Classification
    if not args.skip_classification:
        try:
            book_metrics, book_classifier = classify_books(processed_books)
        except Exception as e:
            print(f"Error during book classification: {e}")
        
        try:
            quote_metrics, quote_classifier = classify_quotes(processed_quotes)
        except Exception as e:
            print(f"Error during quote classification: {e}")
    else:
        print("Skipping classification step...")
    
    # Create visualizations
    try:
        create_visualizations(processed_books, processed_quotes, book_clusters, quote_clusters)
    except Exception as e:
        print(f"Error creating visualizations: {e}")
    
    # Save results
    try:
        save_results(processed_books, processed_quotes, book_clusters, quote_clusters)
    except Exception as e:
        print(f"Error saving results: {e}")
    
    print("\n=== Information Retrieval Project Completed ===\n")


if __name__ == "__main__":
    main()