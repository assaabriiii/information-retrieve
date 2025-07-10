import os
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .selenium_scraper import SeleniumScraper


class SeleniumQuotesScraper(SeleniumScraper):
    """
    Selenium-based scraper for quotes.toscrape.com website.
    Handles JavaScript-rendered content.
    """
    
    def __init__(self, output_dir: str = "data/quotes", delay: float = 1.0, headless: bool = True, chromedriver_path: str = None, offline_mode: bool = False, html_dir: str = None):
        """
        Initialize the Selenium quotes scraper.
        
        Args:
            output_dir: Directory to save scraped data
            delay: Time delay between requests in seconds
            headless: Whether to run the browser in headless mode
            chromedriver_path: Optional path to ChromeDriver executable (useful when offline)
            offline_mode: Whether to use local HTML files instead of fetching from the web
            html_dir: Directory containing local HTML files for offline mode
        """
        super().__init__("http://quotes.toscrape.com/", output_dir, delay, headless, chromedriver_path, offline_mode, html_dir)
        if not html_dir:
            self.html_dir = os.path.join(output_dir, "html")
            os.makedirs(self.html_dir, exist_ok=True)
    
    def scrape_pages(self, num_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape quote pages from the website using Selenium or from local HTML files.
        
        Args:
            num_pages: Number of pages to scrape
            
        Returns:
            List of dictionaries containing quote data
        """
        all_quotes = []
        
        if self.offline_mode:
            # In offline mode, use local HTML files
            print(f"Using offline mode with HTML directory: {self.html_dir}")
            try:
                # List HTML files in the directory
                html_files = [f for f in os.listdir(self.html_dir) 
                             if f.startswith("quotes_page_") and f.endswith(".html")]
                
                # Sort files by page number to maintain order
                html_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
                
                # Limit to requested number of pages
                html_files = html_files[:num_pages]
                
                for filename in html_files:
                    # Extract page number from filename
                    page_num = int(filename.split('_')[2].split('.')[0])
                    print(f"Loading quotes page {page_num} from local file")
                    
                    file_path = os.path.join(self.html_dir, filename)
                    
                    # Create a placeholder URL based on page number
                    if page_num == 1:
                        page_url = self.base_url
                    else:
                        page_url = urljoin(self.base_url, f"page/{page_num}/")
                    
                    # Load the HTML file
                    soup = self.get_page(page_url, local_file=file_path)
                    if soup:
                        # Extract quotes from the page
                        quotes_on_page = self._extract_quotes_from_page(soup)
                        for quote in quotes_on_page:
                            quote['page_num'] = page_num
                            quote['html_file'] = file_path
                            all_quotes.append(quote)
                
            except Exception as e:
                print(f"Error loading offline quotes data: {e}")
        else:
            # Online mode - fetch from web
            page_num = 1
            
            while page_num <= num_pages:
                # Construct the page URL
                if page_num == 1:
                    page_url = self.base_url
                else:
                    page_url = urljoin(self.base_url, f"page/{page_num}/")
                
                print(f"Scraping quotes page {page_num}/{num_pages}")
                soup = self.get_page(page_url)
                if not soup:
                    break
                
                # Save the HTML content (also for future offline use)
                filename = f"quotes_page_{page_num}.html"
                filepath = self.save_html(str(soup), filename, for_offline=True)
                
                # Extract quotes from the page
                quotes_on_page = self._extract_quotes_from_page(soup)
                for quote in quotes_on_page:
                    quote['page_num'] = page_num
                    quote['html_file'] = filepath
                    all_quotes.append(quote)
                
                # Check if there's a next page
                next_button = soup.select_one('li.next a')
                if not next_button:
                    break
                
                page_num += 1
        
        return all_quotes
    
    def _extract_quotes_from_page(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract all quotes from a page.
        
        Args:
            soup: BeautifulSoup object of the quotes page
            
        Returns:
            List of dictionaries containing quote information
        """
        quotes = []
        quote_elements = soup.select('div.quote')
        
        for i, quote_element in enumerate(quote_elements):
            quote_data = self.extract_content(quote_element)
            quote_data['index_on_page'] = i + 1
            quotes.append(quote_data)
        
        return quotes
    
    def extract_content(self, quote_element: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract quote information from a quote element.
        
        Args:
            quote_element: BeautifulSoup object of the quote element
            
        Returns:
            Dictionary containing quote information
        """
        quote_data = {}
        
        # Extract text
        text_element = quote_element.select_one('span.text')
        if text_element:
            quote_data['text'] = text_element.text.strip('"')
        
        # Extract author
        author_element = quote_element.select_one('small.author')
        if author_element:
            quote_data['author'] = author_element.text.strip()
        
        # Extract tags
        tags = []
        tag_elements = quote_element.select('div.tags a.tag')
        for tag_element in tag_elements:
            tags.append(tag_element.text.strip())
        quote_data['tags'] = tags
        
        return quote_data
    
    def load_data(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load previously scraped data from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            List of dictionaries containing quote data
        """
        import json
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading data from {filepath}: {e}")
            return []
    
    def save_data(self, data: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save scraped data to a JSON file.
        
        Args:
            data: List of dictionaries containing quote data
            filepath: Path to save the JSON file
        """
        import json
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Saved data to {filepath}")
        except Exception as e:
            print(f"Error saving data to {filepath}: {e}")