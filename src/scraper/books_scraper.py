import os
import json
from typing import List, Dict, Any
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from .base_scraper import BaseScraper


class BooksScraper(BaseScraper):
    """
    Scraper for books.toscrape.com website.
    """
    
    def __init__(self, output_dir: str = "data/books", delay: float = 1.0, offline_mode: bool = False, html_dir: str = None):
        """
        Initialize the books scraper.
        
        Args:
            output_dir: Directory to save scraped data
            delay: Time delay between requests in seconds
            offline_mode: Whether to use local HTML files instead of fetching from the web
            html_dir: Directory containing local HTML files for offline mode
        """
        super().__init__("http://books.toscrape.com/", output_dir, delay, offline_mode, html_dir)
        if not html_dir:
            self.html_dir = os.path.join(output_dir, "html")
            os.makedirs(self.html_dir, exist_ok=True)
    
    def scrape_pages(self, num_pages: int = 10) -> List[Dict[str, Any]]:
        """
        Scrape book pages from the website or local HTML files.
        
        Args:
            num_pages: Number of book pages to scrape
            
        Returns:
            List of dictionaries containing book data
        """
        books_data = []
        
        if self.offline_mode:
            # In offline mode, use local HTML files
            print(f"Using offline mode with HTML directory: {self.html_dir}")
            try:
                # List HTML files in the directory
                html_files = [f for f in os.listdir(self.html_dir) 
                             if f.startswith("book_") and f.endswith(".html")]
                
                # Sort files by number to maintain order
                html_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
                
                # Limit to requested number of pages
                html_files = html_files[:num_pages]
                
                for i, filename in enumerate(html_files):
                    print(f"Loading book {i+1}/{len(html_files)} from local file")
                    file_path = os.path.join(self.html_dir, filename)
                    
                    # Create a placeholder URL based on filename
                    url = f"http://books.toscrape.com/catalogue/book_{i+1}.html"
                    
                    # Load the HTML file
                    soup = self.get_page(url, local_file=file_path)
                    if soup:
                        # Extract and store book data
                        book_data = self.extract_content(soup)
                        book_data['url'] = url
                        book_data['html_file'] = file_path
                        books_data.append(book_data)
                
            except Exception as e:
                print(f"Error loading offline book data: {e}")
        else:
            # Online mode - fetch from web
            book_urls = self._get_book_urls(num_pages)
            
            for i, url in enumerate(book_urls[:num_pages]):
                print(f"Scraping book {i+1}/{min(num_pages, len(book_urls))}")
                soup = self.get_page(url)
                if soup:
                    # Save the HTML content (also for future offline use)
                    filename = f"book_{i+1}.html"
                    filepath = self.save_html(str(soup), filename, for_offline=True)
                    
                    # Extract and store book data
                    book_data = self.extract_content(soup)
                    book_data['url'] = url
                    book_data['html_file'] = filepath
                    books_data.append(book_data)
        
        return books_data
    
    def _get_book_urls(self, num_books: int) -> List[str]:
        """
        Get URLs for individual book pages.
        
        Args:
            num_books: Number of book URLs to collect
            
        Returns:
            List of book URLs
        """
        book_urls = []
        page_num = 1
        
        while len(book_urls) < num_books:
            # Construct the catalog page URL
            if page_num == 1:
                page_url = self.base_url + "catalogue/category/books_1/index.html"
            else:
                page_url = self.base_url + f"catalogue/category/books_1/page-{page_num}.html"
            
            # Get the catalog page
            soup = self.get_page(page_url)
            if not soup:
                break
            
            # Extract book links from the catalog page
            book_elements = soup.select('article.product_pod h3 a')
            if not book_elements:
                break
            
            # Add book URLs to the list
            for element in book_elements:
                if 'href' in element.attrs:
                    relative_url = element['href']
                    # Convert relative URL to absolute URL
                    absolute_url = urljoin(page_url, relative_url)
                    book_urls.append(absolute_url)
                    
                    if len(book_urls) >= num_books:
                        break
            
            page_num += 1
        
        return book_urls
    
    def extract_content(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract book information from a book page.
        
        Args:
            soup: BeautifulSoup object of the book page
            
        Returns:
            Dictionary containing book information
        """
        book_data = {}
        
        # Extract title
        title_element = soup.select_one('div.product_main h1')
        if title_element:
            book_data['title'] = title_element.text.strip()
        else:
            book_data['title'] = "Unknown Title"
        
        # Extract description
        description_element = soup.select_one('article.product_page div#product_description + p')
        if description_element:
            book_data['description'] = description_element.text.strip()
        else:
            book_data['description'] = ""
        
        # Extract price
        price_element = soup.select_one('div.product_main p.price_color')
        if price_element:
            book_data['price'] = price_element.text.strip()
        
        # Extract availability
        availability_element = soup.select_one('div.product_main p.availability')
        if availability_element:
            book_data['availability'] = availability_element.text.strip()
        
        # Extract category
        category_element = soup.select('ul.breadcrumb li a')
        if len(category_element) >= 3:
            book_data['category'] = category_element[2].text.strip()
        else:
            book_data['category'] = "Unknown Category"
        
        # Extract rating
        rating_element = soup.select_one('div.product_main p.star-rating')
        if rating_element and 'class' in rating_element.attrs:
            classes = rating_element['class']
            # The rating is stored as a class, e.g., 'star-rating Three'
            rating_classes = [c for c in classes if c != 'star-rating']
            if rating_classes:
                book_data['rating'] = rating_classes[0]
        
        return book_data
    
    def save_data(self, data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save scraped data to a JSON file.
        
        Args:
            data: List of dictionaries containing book data
            file_path: Path to save the JSON file
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save data to JSON file
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        
        print(f"Saved {len(data)} books to {file_path}")
    
    def load_data(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load scraped data from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            List of dictionaries containing book data
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return []
        
        # Load data from JSON file
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} books from {file_path}")
        return data