import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class ClassificationAnalyzer:
    """
    Class for text classification and evaluation.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the classification analyzer.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.models = {
            'naive_bayes': MultinomialNB(),
            'svm': SVC(kernel='linear', random_state=random_state),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000)
        }
        self.model = None
        self.model_name = None
        self.label_encoder = LabelEncoder()
        self.classes = None
    
    def train(self, X: np.ndarray, y: np.ndarray, model_name: str = 'naive_bayes') -> None:
        """
        Train a classification model.
        
        Args:
            X: Feature matrix
            y: Target labels
            model_name: Name of the model to train
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {list(self.models.keys())}")
        
        # Encode labels if they are not numeric
        if not isinstance(y[0], (int, np.integer)):
            y = self.label_encoder.fit_transform(y)
            self.classes = self.label_encoder.classes_
        else:
            self.classes = np.unique(y)
        
        self.model_name = model_name
        self.model = self.models[model_name]
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions.")
        
        return self.model.predict(X)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Feature matrix
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")
        
        # Encode labels if they are not numeric
        if not isinstance(y[0], (int, np.integer)):
            y = self.label_encoder.transform(y)
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y, y_pred, average='weighted', zero_division=0)
        }
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        if self.model is None:
            raise ValueError("Model must be trained before cross-validation.")
        
        # Encode labels if they are not numeric
        if not isinstance(y[0], (int, np.integer)):
            y = self.label_encoder.transform(y)
        
        # Check if any class has only one member
        class_counts = np.bincount(y) if np.issubdtype(y.dtype, np.integer) else np.bincount(y.astype(int))
        has_single_member_class = np.any(class_counts[class_counts > 0] < 2)
        
        # Adjust CV strategy if any class has only one member
        if has_single_member_class:
            # Use simple KFold instead of StratifiedKFold
            cv_strategy = KFold(n_splits=min(cv, len(y)), shuffle=True, random_state=self.random_state)
        else:
            cv_strategy = cv
        
        # Perform cross-validation
        try:
            cv_scores = cross_val_score(self.model, X, y, cv=cv_strategy)
        except ValueError as e:
            # If cross-validation fails, return a warning and default scores
            print(f"Warning: Cross-validation failed with error: {str(e)}")
            print("Using default scores instead.")
            return {
                'mean_cv_score': 0.0,
                'std_cv_score': 0.0,
                'cv_scores': [0.0],
                'error': str(e)
            }
        
        return {
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
    
    def plot_confusion_matrix(self, X: np.ndarray, y: np.ndarray, title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            X: Feature matrix
            y: True labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        if self.model is None:
            raise ValueError("Model must be trained before plotting confusion matrix.")
        
        # Encode labels if they are not numeric
        if not isinstance(y[0], (int, np.integer)):
            y = self.label_encoder.transform(y)
            class_names = self.label_encoder.classes_
        else:
            class_names = [str(c) for c in np.unique(y)]
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        
        # Add labels
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Predicted Labels', fontsize=12)
        ax.set_ylabel('True Labels', fontsize=12)
        
        return fig
    
    def compare_models(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
        """
        Compare different classification models.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary of model names mapped to their evaluation metrics
        """
        # Encode labels if they are not numeric
        if not isinstance(y[0], (int, np.integer)):
            y = self.label_encoder.fit_transform(y)
            self.classes = self.label_encoder.classes_
        else:
            self.classes = np.unique(y)
        
        # Check if any class has only one member
        class_counts = np.bincount(y) if np.issubdtype(y.dtype, np.integer) else np.bincount(y.astype(int))
        has_single_member_class = np.any(class_counts[class_counts > 0] < 2)
        
        # Split data into training and testing sets
        # Disable stratification if any class has only one member
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=None if has_single_member_class else y
        )
        
        results = {}
        
        # Train and evaluate each model
        for name, model in self.models.items():
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
            }
        
        return results