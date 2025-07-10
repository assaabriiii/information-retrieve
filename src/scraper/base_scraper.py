import os
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseScraper(ABC):
    """
    Abstract base class for web scrapers.
    Provides common functionality for scraping websites.
    """
    
    def __init__(self, base_url: str, output_dir: str, delay: float = 1.0, offline_mode: bool = False, html_dir: str = None):
        """
        Initialize the scraper with base URL and output directory.
        
        Args:
            base_url: The base URL of the website to scrape
            output_dir: Directory to save scraped data
            delay: Time delay between requests in seconds to avoid overloading the server
            offline_mode: Whether to use local HTML files instead of fetching from the web
            html_dir: Directory containing local HTML files for offline mode
        """
        self.base_url = base_url
        self.output_dir = output_dir
        self.delay = delay
        self.offline_mode = offline_mode
        self.html_dir = html_dir or os.path.join(output_dir, "html")
        
        # Only create session if not in offline mode
        if not offline_mode:
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                            '(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create HTML directory if in offline mode and it doesn't exist
        if offline_mode and html_dir:
            os.makedirs(html_dir, exist_ok=True)
    
    def get_page(self, url: str, local_file: str = None) -> Optional[BeautifulSoup]:
        """
        Fetch a page and return its BeautifulSoup object.
        In offline mode, reads from local HTML files instead.
        
        Args:
            url: URL to fetch or identifier for local file
            local_file: Optional path to local HTML file (for offline mode)
            
        Returns:
            BeautifulSoup object or None if request failed
        """
        # If in offline mode, try to load from local file
        if self.offline_mode:
            try:
                # If local_file is provided, use it directly
                if local_file:
                    file_path = local_file
                else:
                    # Otherwise, try to derive filename from URL
                    # Extract the last part of the URL path
                    import re
                    from urllib.parse import urlparse
                    parsed_url = urlparse(url)
                    path_parts = parsed_url.path.strip('/').split('/')
                    if path_parts and path_parts[-1]:
                        filename = path_parts[-1]
                        # If it doesn't have an extension, add .html
                        if not re.search(r'\.(html|htm)$', filename):
                            filename += '.html'
                    else:
                        # For root URLs, use index.html
                        filename = 'index.html'
                    
                    file_path = os.path.join(self.html_dir, filename)
                
                print(f"Loading from local file: {file_path}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                return BeautifulSoup(html_content, 'html.parser')
            except Exception as e:
                print(f"Error loading local file for {url}: {e}")
                return None
        
        # Online mode - fetch from web
        try:
            response = self.session.get(url)
            response.raise_for_status()
            time.sleep(self.delay)  # Be nice to the server
            return BeautifulSoup(response.text, 'html.parser')
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url}: {e}")
            print("Try using --offline-mode with --html-dir to use local HTML files.")
            return None
    
    def save_html(self, html_content: str, filename: str, for_offline: bool = False) -> str:
        """
        Save HTML content to a file.
        
        Args:
            html_content: HTML content to save
            filename: Name of the file to save to
            for_offline: Whether to save in the HTML directory for offline use
            
        Returns:
            Path to the saved file
        """
        if for_offline:
            # Save in HTML directory for offline use
            os.makedirs(self.html_dir, exist_ok=True)
            filepath = os.path.join(self.html_dir, filename)
        else:
            # Save in regular output directory
            filepath = os.path.join(self.output_dir, filename)
            
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return filepath
    
    @abstractmethod
    def scrape_pages(self, num_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape a specified number of pages from the website.
        
        Args:
            num_pages: Number of pages to scrape
            
        Returns:
            List of dictionaries containing scraped data
        """
        pass
    
    @abstractmethod
    def extract_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract content from a BeautifulSoup object.
        
        Args:
            soup: BeautifulSoup object to extract content from
            
        Returns:
            Dictionary containing extracted content
        """
        pass