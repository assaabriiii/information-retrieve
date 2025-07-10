import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from collections import Counter


class ClusteringAnalyzer:
    """
    Class for clustering text data and evaluating clustering performance.
    """
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize the clustering analyzer.
        
        Args:
            n_clusters: Number of clusters for algorithms that require this parameter
            random_state: Random state for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
        self.pca = PCA(n_components=2, random_state=random_state)
        self.cluster_labels = None
        self.model = None
    
    def cluster_kmeans(self, X: np.ndarray) -> np.ndarray:
        """
        Perform K-means clustering.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        # Check if we have enough samples for the requested number of clusters
        n_samples = X.shape[0]
        actual_n_clusters = min(self.n_clusters, n_samples - 1) if n_samples > 1 else 1
        
        if actual_n_clusters != self.n_clusters:
            print(f"Warning: Reducing number of clusters from {self.n_clusters} to {actual_n_clusters} because n_samples={n_samples} is too small.")
            # Create a new KMeans instance with the adjusted number of clusters
            kmeans = KMeans(n_clusters=actual_n_clusters, random_state=self.random_state, n_init=10)
            self.model = kmeans
        else:
            self.model = self.kmeans
            
        self.cluster_labels = self.model.fit_predict(X)
        return self.cluster_labels
    
    def cluster_dbscan(self, X: np.ndarray) -> np.ndarray:
        """
        Perform DBSCAN clustering.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        self.model = self.dbscan
        self.cluster_labels = self.dbscan.fit_predict(X)
        return self.cluster_labels
    
    def cluster_agglomerative(self, X: np.ndarray) -> np.ndarray:
        """
        Perform Agglomerative clustering.
        
        Args:
            X: Feature matrix
            
        Returns:
            Cluster labels
        """
        self.model = self.agglomerative
        self.cluster_labels = self.agglomerative.fit_predict(X)
        return self.cluster_labels
    
    def evaluate_silhouette(self, X: np.ndarray) -> float:
        """
        Evaluate clustering using silhouette score.
        
        Args:
            X: Feature matrix
            
        Returns:
            Silhouette score
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed before evaluation.")
        
        # Filter out noise points (label -1) for DBSCAN
        if hasattr(self.model, "core_sample_indices_"):
            mask = self.cluster_labels != -1
            if sum(mask) <= 1:  # Need at least 2 samples for silhouette score
                return 0.0
            return silhouette_score(X[mask], self.cluster_labels[mask])
        
        return silhouette_score(X, self.cluster_labels)
    
    def calculate_purity(self, true_labels: np.ndarray) -> float:
        """
        Calculate purity score.
        
        Args:
            true_labels: True class labels
            
        Returns:
            Purity score
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed before evaluation.")
        
        # Create contingency matrix
        contingency_matrix = {}
        for i in range(len(true_labels)):
            cluster = self.cluster_labels[i]
            label = true_labels[i]
            
            if cluster not in contingency_matrix:
                contingency_matrix[cluster] = {}
            
            if label not in contingency_matrix[cluster]:
                contingency_matrix[cluster][label] = 0
            
            contingency_matrix[cluster][label] += 1
        
        # Calculate purity
        total = len(true_labels)
        purity = 0
        
        for cluster in contingency_matrix:
            max_count = max(contingency_matrix[cluster].values())
            purity += max_count
        
        return purity / total
    
    def calculate_rand_index(self, true_labels: np.ndarray) -> float:
        """
        Calculate Rand Index.
        
        Args:
            true_labels: True class labels
            
        Returns:
            Rand Index
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed before evaluation.")
        
        n = len(true_labels)
        a, b = 0, 0
        
        # Count pairs
        for i in range(n):
            for j in range(i + 1, n):
                if (true_labels[i] == true_labels[j]) and (self.cluster_labels[i] == self.cluster_labels[j]):
                    a += 1  # Same class, same cluster
                elif (true_labels[i] != true_labels[j]) and (self.cluster_labels[i] != self.cluster_labels[j]):
                    b += 1  # Different class, different cluster
        
        # Calculate Rand Index
        total_pairs = n * (n - 1) // 2
        return (a + b) / total_pairs
    
    def visualize_clusters(self, X: np.ndarray, title: str = "Cluster Visualization") -> plt.Figure:
        """
        Visualize clusters using PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed before visualization.")
        
        # Reduce dimensionality for visualization
        X_reduced = self.pca.fit_transform(X.toarray() if hasattr(X, 'toarray') else X)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get unique cluster labels
        unique_labels = np.unique(self.cluster_labels)
        
        # Plot each cluster
        for label in unique_labels:
            if label == -1:
                # Plot noise points in black
                ax.scatter(X_reduced[self.cluster_labels == label, 0],
                           X_reduced[self.cluster_labels == label, 1],
                           c='black', marker='x', s=100, label='Noise')
            else:
                # Plot cluster points
                ax.scatter(X_reduced[self.cluster_labels == label, 0],
                           X_reduced[self.cluster_labels == label, 1],
                           s=100, label=f'Cluster {label}')
        
        # Add labels and legend
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Principal Component 1', fontsize=12)
        ax.set_ylabel('Principal Component 2', fontsize=12)
        ax.legend()
        
        return fig
    
    def get_cluster_distribution(self) -> Dict[int, int]:
        """
        Get the distribution of data points across clusters.
        
        Returns:
            Dictionary mapping cluster labels to counts
        """
        if self.cluster_labels is None:
            raise ValueError("Clustering must be performed before getting distribution.")
        
        return dict(Counter(self.cluster_labels))
    
    def get_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> Tuple[int, List[float]]:
        """
        Find the optimal number of clusters using the elbow method and silhouette score.
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to try
            
        Returns:
            Tuple of (optimal number of clusters, list of inertia values)
        """
        inertia_values = []
        silhouette_values = []
        
        # Ensure max_clusters doesn't exceed the number of samples
        n_samples = X.shape[0]
        max_possible_clusters = min(max_clusters, n_samples - 1)
        
        # If we have very few samples, just return 2 clusters or 1 if only one sample
        if max_possible_clusters < 2:
            return 1, [0]
        
        # Adjust the range to ensure we have at least 2 samples per cluster
        for n in range(2, max_possible_clusters + 1):
            kmeans = KMeans(n_clusters=n, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            inertia_values.append(kmeans.inertia_)
            
            # Calculate silhouette score
            try:
                silhouette_values.append(silhouette_score(X, labels))
            except:
                silhouette_values.append(0)
        
        # If we couldn't calculate any values, return 2 as default
        if not inertia_values:
            return min(2, n_samples), [0]
            
        # Find the optimal number of clusters using the elbow method
        # This is a simple heuristic - in practice, you might want to use more sophisticated methods
        diffs = np.diff(inertia_values)
        if len(diffs) > 0:
            optimal_clusters = np.argmax(diffs) + 2  # +2 because we start from 2 clusters
        else:
            optimal_clusters = 2
            
        # Ensure optimal_clusters doesn't exceed the number of samples
        optimal_clusters = min(optimal_clusters, n_samples)
        
        return optimal_clusters, inertia_values