import re
import string
from typing import List, Dict, Any, Union, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import nltk

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


class TextProcessor:
    """
    Class for preprocessing text data.
    """
    
    def __init__(self, 
                 remove_stopwords: bool = True, 
                 stemming: bool = False, 
                 lemmatization: bool = True,
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = True):
        """
        Initialize the text processor with specified options.
        
        Args:
            remove_stopwords: Whether to remove stopwords
            stemming: Whether to apply stemming
            lemmatization: Whether to apply lemmatization
            lowercase: Whether to convert text to lowercase
            remove_punctuation: Whether to remove punctuation
            remove_numbers: Whether to remove numbers
        """
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        
        # Initialize tools
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, text: str) -> str:
        """
        Apply all preprocessing steps to a text string.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove numbers if specified
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if specified
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize the text
        tokens = word_tokenize(text)
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stop_words]
        
        # Apply stemming if specified
        if self.stemming:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # Apply lemmatization if specified
        if self.lemmatization:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Join tokens back into a string
        return ' '.join(tokens)
    
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Apply preprocessing to a list of documents.
        
        Args:
            documents: List of text documents to preprocess
            
        Returns:
            List of preprocessed documents
        """
        return [self.preprocess_text(doc) for doc in documents]
    
    def preprocess_data(self, data: List[Dict[str, Any]], text_field: str) -> List[Dict[str, Any]]:
        """
        Preprocess text field in a list of dictionaries.
        
        Args:
            data: List of dictionaries containing text data
            text_field: Key for the text field to preprocess
            
        Returns:
            List of dictionaries with preprocessed text
        """
        processed_data = []
        
        for item in data:
            # Create a copy of the item
            processed_item = item.copy()
            
            # Preprocess the text field
            if text_field in processed_item and processed_item[text_field]:
                processed_item[f'processed_{text_field}'] = self.preprocess_text(processed_item[text_field])
            else:
                processed_item[f'processed_{text_field}'] = ""
            
            processed_data.append(processed_item)
        
        return processed_data
    
    def vectorize_tfidf(self, documents: List[str], max_features: Optional[int] = None) -> np.ndarray:
        """
        Convert documents to TF-IDF vectors.
        
        Args:
            documents: List of preprocessed documents
            max_features: Maximum number of features to extract
            
        Returns:
            TF-IDF matrix
        """
        vectorizer = TfidfVectorizer(max_features=max_features)
        return vectorizer.fit_transform(documents), vectorizer
    
    def vectorize_bow(self, documents: List[str], max_features: Optional[int] = None) -> np.ndarray:
        """
        Convert documents to Bag-of-Words vectors.
        
        Args:
            documents: List of preprocessed documents
            max_features: Maximum number of features to extract
            
        Returns:
            Bag-of-Words matrix
        """
        vectorizer = CountVectorizer(max_features=max_features)
        return vectorizer.fit_transform(documents), vectorizer
    
    def vectorize_documents(self, documents: List[str], max_features: Optional[int] = None, method: str = 'tfidf') -> tuple:
        """
        Convert documents to vectors using the specified method.
        
        Args:
            documents: List of preprocessed documents
            max_features: Maximum number of features to extract
            method: Vectorization method ('tfidf' or 'bow')
            
        Returns:
            Tuple of (document vectors, feature names)
        """
        if method == 'bow':
            vectorizer = CountVectorizer(max_features=max_features)
        else:  # default to tfidf
            vectorizer = TfidfVectorizer(max_features=max_features)
            
        vectors = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        return vectors, feature_names