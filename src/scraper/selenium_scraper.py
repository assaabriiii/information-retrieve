import os
import time
from typing import Optional
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from .base_scraper import BaseScraper


class SeleniumScraper(BaseScraper):
    """
    Extension of BaseScraper that uses Selenium with JavaScript support.
    Useful for scraping dynamic websites that require JavaScript execution.
    """
    
    def __init__(self, base_url: str, output_dir: str, delay: float = 1.0, headless: bool = True, chromedriver_path: str = None, offline_mode: bool = False, html_dir: str = None):
        """
        Initialize the Selenium scraper.
        
        Args:
            base_url: The base URL of the website to scrape
            output_dir: Directory to save scraped data
            delay: Time delay between requests in seconds
            headless: Whether to run the browser in headless mode
            chromedriver_path: Optional path to ChromeDriver executable (useful when offline)
            offline_mode: Whether to use local HTML files instead of fetching from the web
            html_dir: Directory containing local HTML files for offline mode
        """
        super().__init__(base_url, output_dir, delay)
        self.headless = headless
        self.chromedriver_path = chromedriver_path
        self.offline_mode = offline_mode
        self.html_dir = html_dir or os.path.join(output_dir, "html")
        self.driver = None
        if not offline_mode:
            self._setup_driver()
    
    def _setup_driver(self):
        """
        Set up the Selenium WebDriver.
        """
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
        
        # If a specific ChromeDriver path is provided, use it
        if self.chromedriver_path:
            try:
                print(f"Using provided ChromeDriver at: {self.chromedriver_path}")
                service = Service(executable_path=self.chromedriver_path)
                self.driver = webdriver.Chrome(service=service, options=options)
                return
            except Exception as e:
                print(f"Error using provided ChromeDriver path: {e}")
                print("Falling back to automatic detection...")
        
        try:
            # Try to use ChromeDriverManager to automatically download the driver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=options)
        except Exception as e:
            print(f"Error setting up ChromeDriver with ChromeDriverManager: {e}")
            print("Trying alternative setup method...")
            try:
                # Try to use Chrome directly without ChromeDriverManager
                self.driver = webdriver.Chrome(options=options)
            except Exception as e2:
                print(f"Error setting up Chrome directly: {e2}")
                print("\nConnection Error: Could not set up Chrome WebDriver.")
                print("Please check your internet connection and try again.")
                print("If you're offline, make sure ChromeDriver is already installed or provide the path using --chromedriver-path.")
                raise Exception("Failed to initialize Chrome WebDriver. Check your internet connection or provide a valid ChromeDriver path.")
    
    def get_page(self, url: str, local_file: str = None) -> Optional[BeautifulSoup]:
        """
        Fetch a page using Selenium and return its BeautifulSoup object.
        This method overrides the parent class method to use Selenium instead of requests.
        In offline mode, it reads from local HTML files instead.
        
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
        
        # Online mode - first check if the URL is reachable
        try:
            import requests
            # Just check the head to see if the server is reachable
            response = requests.head(url, timeout=5)
            # We don't care about the status code, just that the server responded
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not reach {url}: {e}")
            print("The website may be down or you may be offline.")
            print("Try using --offline-mode with --html-dir to use local HTML files.")
            return None
            
        try:
            self.driver.get(url)
            time.sleep(self.delay)  # Wait for JavaScript to execute
            
            # Get the page source after JavaScript execution
            html_content = self.driver.page_source
            return BeautifulSoup(html_content, 'html.parser')
        except Exception as e:
            print(f"Error fetching {url} with Selenium: {e}")
            return None
    
    def execute_js(self, script: str):
        """
        Execute JavaScript on the current page.
        
        Args:
            script: JavaScript code to execute
            
        Returns:
            Result of the JavaScript execution
        """
        try:
            return self.driver.execute_script(script)
        except Exception as e:
            print(f"Error executing JavaScript: {e}")
            return None
    
    def scroll_to_bottom(self):
        """
        Scroll to the bottom of the page to load lazy-loaded content.
        """
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(1)  # Wait for content to load
    
    def scroll_by(self, pixels: int):
        """
        Scroll down by a specific number of pixels.
        
        Args:
            pixels: Number of pixels to scroll down
        """
        self.driver.execute_script(f"window.scrollBy(0, {pixels});")
        time.sleep(0.5)  # Wait for content to load
    
    def wait_for_element(self, selector: str, timeout: int = 10):
        """
        Wait for an element to be present on the page.
        
        Args:
            selector: CSS selector for the element
            timeout: Maximum time to wait in seconds
            
        Returns:
            The element if found, None otherwise
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        try:
            element = WebDriverWait(self.driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            return element
        except Exception as e:
            print(f"Error waiting for element {selector}: {e}")
            return None
    
    def __del__(self):
        """
        Clean up resources when the object is destroyed.
        """
        if self.driver:
            try:
                self.driver.quit()
            except:
                pass  # Ignore errors during cleanup