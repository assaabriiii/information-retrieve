#!/usr/bin/env python3

"""
Test script to verify that all components of the information retrieval project are working correctly.
"""

import os
import sys
import importlib
import unittest


class ProjectTest(unittest.TestCase):
    """Test case for the information retrieval project."""
    
    def test_imports(self):
        """Test that all project modules can be imported."""
        modules = [
            'src.scraper.base_scraper',
            'src.scraper.books_scraper',
            'src.scraper.quotes_scraper',
            'src.preprocessing.text_processor',
            'src.analysis.clustering',
            'src.analysis.classification',
            'src.visualization.visualizer'
        ]
        
        for module in modules:
            try:
                importlib.import_module(module)
                print(f"✓ Successfully imported {module}")
            except ImportError as e:
                self.fail(f"Failed to import {module}: {e}")
    
    def test_directory_structure(self):
        """Test that the project directory structure is correct."""
        directories = [
            'data',
            'data/books',
            'data/books/html',
            'data/quotes',
            'data/quotes/html',
            'results',
            'results/figures',
            'results/models',
            'src',
            'src/scraper',
            'src/preprocessing',
            'src/analysis',
            'src/visualization'
        ]
        
        for directory in directories:
            self.assertTrue(os.path.isdir(directory), f"Directory {directory} does not exist")
            print(f"✓ Directory {directory} exists")
    
    def test_scraper_initialization(self):
        """Test that the scrapers can be initialized."""
        from src.scraper.books_scraper import BooksScraper
        from src.scraper.quotes_scraper import QuotesScraper
        
        try:
            books_scraper = BooksScraper()
            print("✓ BooksScraper initialized successfully")
            self.assertEqual(books_scraper.base_url, "http://books.toscrape.com/")
            
            quotes_scraper = QuotesScraper()
            print("✓ QuotesScraper initialized successfully")
            self.assertEqual(quotes_scraper.base_url, "http://quotes.toscrape.com/")
        except Exception as e:
            self.fail(f"Failed to initialize scrapers: {e}")
    
    def test_text_processor(self):
        """Test that the text processor works correctly."""
        from src.preprocessing.text_processor import TextProcessor
        
        try:
            processor = TextProcessor()
            print("✓ TextProcessor initialized successfully")
            
            # Test preprocessing
            text = "<p>This is a test! With some punctuation, and numbers 123.</p>"
            processed = processor.preprocess_text(text)
            print(f"✓ Preprocessed text: {processed}")
            
            # Check that HTML tags are removed
            self.assertNotIn('<', processed)
            self.assertNotIn('>', processed)
            
            # Check that punctuation is removed
            self.assertNotIn('!', processed)
            self.assertNotIn(',', processed)
            self.assertNotIn('.', processed)
            
            # Check that numbers are removed
            self.assertNotIn('123', processed)
        except Exception as e:
            self.fail(f"Failed to test text processor: {e}")
    
    def test_clustering(self):
        """Test that the clustering analyzer works correctly."""
        from src.analysis.clustering import ClusteringAnalyzer
        import numpy as np
        
        try:
            clustering = ClusteringAnalyzer(n_clusters=2)
            print("✓ ClusteringAnalyzer initialized successfully")
            
            # Create a simple dataset
            X = np.array([[1, 1], [1, 2], [2, 2], [8, 8], [8, 9], [9, 9]])
            
            # Test clustering
            labels = clustering.cluster_kmeans(X)
            print(f"✓ Clustering performed successfully: {labels}")
            
            # Check that there are 2 clusters
            self.assertEqual(len(set(labels)), 2)
        except Exception as e:
            self.fail(f"Failed to test clustering: {e}")
    
    def test_classification(self):
        """Test that the classification analyzer works correctly."""
        from src.analysis.classification import ClassificationAnalyzer
        import numpy as np
        
        try:
            classifier = ClassificationAnalyzer()
            print("✓ ClassificationAnalyzer initialized successfully")
            
            # Create a simple dataset
            X = np.array([[1, 1], [1, 2], [2, 2], [8, 8], [8, 9], [9, 9]])
            y = np.array([0, 0, 0, 1, 1, 1])
            
            # Test training
            classifier.train(X, y, model_name='naive_bayes')
            print("✓ Classification model trained successfully")
            
            # Test prediction
            y_pred = classifier.predict(X)
            print(f"✓ Classification predictions: {y_pred}")
            
            # Check that predictions match labels
            self.assertTrue(np.array_equal(y, y_pred))
        except Exception as e:
            self.fail(f"Failed to test classification: {e}")
    
    def test_visualizer(self):
        """Test that the visualizer works correctly."""
        from src.visualization.visualizer import Visualizer
        
        try:
            visualizer = Visualizer()
            print("✓ Visualizer initialized successfully")
            
            # Test creating a word cloud
            text = "This is a test text for word cloud visualization. "
            text += "The more frequent words should appear larger in the word cloud. "
            text += "Test test test visualization visualization word cloud."
            
            fig = visualizer.create_wordcloud(text, title="Test Word Cloud")
            print("✓ Word cloud created successfully")
            
            # Check that the figure was created
            self.assertIsNotNone(fig)
        except Exception as e:
            self.fail(f"Failed to test visualizer: {e}")


if __name__ == "__main__":
    # Create necessary directories
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
    
    # Run tests
    unittest.main()