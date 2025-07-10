import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE


class Visualizer:
    """
    Class for creating visualizations of text data and analysis results.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the visualizer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.pca = PCA(n_components=2, random_state=random_state)
        self.tsne = TSNE(n_components=2, random_state=random_state)
        self.svd = TruncatedSVD(n_components=2, random_state=random_state)
    
    def create_wordcloud(self, text: str, title: str = "Word Cloud", 
                         width: int = 800, height: int = 400, 
                         background_color: str = 'white') -> plt.Figure:
        """
        Create a word cloud from text.
        
        Args:
            text: Text to create word cloud from
            title: Plot title
            width: Width of the word cloud
            height: Height of the word cloud
            background_color: Background color of the word cloud
            
        Returns:
            Matplotlib figure
        """
        # Create word cloud
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            max_words=200,
            collocations=False,
            random_state=self.random_state
        ).generate(text)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Display word cloud
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(title, fontsize=14)
        ax.axis('off')
        
        return fig
    
    def create_class_wordclouds(self, texts: List[str], labels: List[Union[str, int]], 
                               title_prefix: str = "Word Cloud for Class") -> Dict[Union[str, int], plt.Figure]:
        """
        Create word clouds for each class.
        
        Args:
            texts: List of texts
            labels: List of class labels
            title_prefix: Prefix for plot titles
            
        Returns:
            Dictionary mapping class labels to word cloud figures
        """
        # Group texts by class
        class_texts = {}
        for text, label in zip(texts, labels):
            if label not in class_texts:
                class_texts[label] = []
            class_texts[label].append(text)
        
        # Create word cloud for each class
        wordclouds = {}
        for label, texts in class_texts.items():
            combined_text = ' '.join(texts)
            title = f"{title_prefix} {label}"
            wordclouds[label] = self.create_wordcloud(combined_text, title=title)
        
        return wordclouds
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray, 
                               title: str = "Feature Importance", top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance.
        
        Args:
            feature_names: List of feature names
            importances: Array of feature importances
            title: Plot title
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        # Sort features by importance
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot feature importance
        sns.barplot(x=top_importances, y=top_features, ax=ax)
        
        # Add labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        
        return fig
    
    def plot_document_similarity(self, X: np.ndarray, labels: Optional[np.ndarray] = None, 
                               method: str = 'pca', title: str = "Document Similarity") -> plt.Figure:
        """
        Plot document similarity using dimensionality reduction.
        
        Args:
            X: Feature matrix
            labels: Document labels (optional)
            method: Dimensionality reduction method ('pca', 'tsne', or 'svd')
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Apply dimensionality reduction
        if method.lower() == 'pca':
            X_reduced = self.pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        elif method.lower() == 'tsne':
            X_reduced = self.tsne.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        elif method.lower() == 'svd':
            X_reduced = self.svd.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        else:
            raise ValueError(f"Unknown method: {method}. Available methods: 'pca', 'tsne', 'svd'")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot points
        if labels is not None:
            # Plot points with labels
            unique_labels = np.unique(labels)
            for label in unique_labels:
                mask = labels == label
                ax.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=f'Class {label}', s=100, alpha=0.7)
            ax.legend()
        else:
            # Plot points without labels
            ax.scatter(X_reduced[:, 0], X_reduced[:, 1], s=100, alpha=0.7)
        
        # Add labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(f'Component 1', fontsize=12)
        ax.set_ylabel(f'Component 2', fontsize=12)
        
        return fig
    
    def plot_term_frequency(self, texts: List[str], top_n: int = 20, 
                          title: str = "Top Terms by Frequency") -> plt.Figure:
        """
        Plot term frequency.
        
        Args:
            texts: List of texts
            top_n: Number of top terms to show
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Combine texts
        combined_text = ' '.join(texts)
        
        # Count terms
        vectorizer = CountVectorizer(stop_words='english')
        X = vectorizer.fit_transform([combined_text])
        terms = vectorizer.get_feature_names_out()
        term_counts = X.toarray()[0]
        
        # Sort terms by frequency
        indices = np.argsort(term_counts)[::-1][:top_n]
        top_terms = [terms[i] for i in indices]
        top_counts = term_counts[indices]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot term frequency
        sns.barplot(x=top_counts, y=top_terms, ax=ax)
        
        # Add labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Frequency', fontsize=12)
        ax.set_ylabel('Term', fontsize=12)
        
        return fig
    
    def plot_evaluation_metrics(self, metrics: Dict[str, Dict[str, float]], 
                              title: str = "Model Evaluation Metrics") -> plt.Figure:
        """
        Plot evaluation metrics for multiple models.
        
        Args:
            metrics: Dictionary mapping model names to dictionaries of metrics
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        # Extract model names and metric names
        model_names = list(metrics.keys())
        metric_names = list(metrics[model_names[0]].keys())
        
        # Create figure
        fig, axes = plt.subplots(len(metric_names), 1, figsize=(12, 4 * len(metric_names)))
        
        # Plot each metric
        for i, metric_name in enumerate(metric_names):
            ax = axes[i] if len(metric_names) > 1 else axes
            
            # Extract metric values for each model
            metric_values = [metrics[model][metric_name] for model in model_names]
            
            # Plot metric values
            sns.barplot(x=model_names, y=metric_values, ax=ax)
            
            # Add labels
            ax.set_title(f"{metric_name.capitalize()}", fontsize=12)
            ax.set_xlabel('Model', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            
            # Add value labels
            for j, value in enumerate(metric_values):
                ax.text(j, value, f"{value:.3f}", ha='center', va='bottom', fontsize=10)
        
        # Add overall title
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        
        return fig