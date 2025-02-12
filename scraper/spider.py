import hashlib
import json
import requests
import time
from bs4 import BeautifulSoup
from collections import deque
from urllib.parse import urlparse, urlunparse, urljoin


class WebCrawler:
    def __init__(self, seed_urls, max_depth=2, time_limit=1800, visited_file="../visited_urls.json", output_file="marvel_sankalp3.json"):
        """
        Initializes the WebCrawler with multiple seed URLs.
        - max_depth: BFS depth limit (default = 2)
        - time_limit: Max crawling time in seconds (default = 10 minutes)
        """
        # Load existing visited URLs but maintain a separate set for current session
        self.existing_visited = self.load_json(visited_file)
        self.current_session_visited = set()  # Track URLs visited in current session
        self.scraped_data = self.load_json(output_file)
        self.url_queue = deque([(url, 0) for url in seed_urls])  # Store (URL, depth)
        self.visited_file = visited_file
        self.output_file = output_file
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = time.time()

    @staticmethod
    def load_json(file):
        """Loads JSON data from a file."""
        try:
            with open(file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_json(self, data, file):
        """Saves JSON data to a file.""" 
        with open(file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def generate_url_hash(url):
        """Generates a hash for a URL."""
        return hashlib.sha256(url.encode('utf-8')).hexdigest()

    @staticmethod
    def normalize_url(url):
        """Normalizes a URL by removing fragments and query parameters."""
        parsed_url = urlparse(url)
        return urlunparse(parsed_url._replace(fragment='', query=''))

    @staticmethod
    def extract_text_content(soup):
        """Extracts main text content from a webpage."""
        paragraphs = soup.find_all('p')
        return "\n".join([para.get_text(strip=True) for para in paragraphs])

    @staticmethod
    def extract_links(soup, base_url):
        """Extracts all fandom-related links from a webpage."""
        links = set()
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)
            if full_url.startswith('https://') and 'fandom.com' in full_url:
                links.add(full_url)
        return links

    def crawl(self, url, depth):
        """
        Crawls a given URL: extracts text content, finds new links, and adds them to the queue.
        - Stops if max_depth is reached.
        """
        normalized_url = self.normalize_url(url)
        url_hash = self.generate_url_hash(normalized_url)

        # Check both existing and current session visited URLs
        if url_hash in self.existing_visited or url_hash in self.current_session_visited:
            return

        print(f"Scraping: {normalized_url} (Depth: {depth})")

        try:
            response = requests.get(normalized_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Failed to fetch {normalized_url}: {e}")
            return

        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract and store text content
        text_content = self.extract_text_content(soup)
        self.current_session_visited.add(url_hash)  # Add to current session set
        self.scraped_data[url_hash] = {
            "url": normalized_url,
            "content": text_content
        }

        # Stop adding new links if max depth is reached
        if depth >= self.max_depth:
            return

        # Extract and queue new links
        new_links = self.extract_links(soup, normalized_url)
        for new_url in new_links:
            normalized_new_url = self.normalize_url(new_url)
            new_url_hash = self.generate_url_hash(normalized_new_url)
            if new_url_hash not in self.existing_visited and new_url_hash not in self.current_session_visited:
                self.url_queue.append((normalized_new_url, depth + 1))

    def start_crawl(self):
        """Starts the crawling process with stopping conditions."""
        try:
            while self.url_queue:
                # Stop if time limit is reached
                if time.time() - self.start_time > self.time_limit:
                    print("Stopping: Time limit reached (10 minutes)")
                    break

                current_url, current_depth = self.url_queue.popleft()
                self.crawl(current_url, current_depth)

                # Save only scraped data periodically, not visited URLs
                if len(self.current_session_visited) % 10 == 0:
                    self.save_json(self.scraped_data, self.output_file)

        finally:
            # Update existing_visited with new URLs and save at the end
            new_visited = {}
            for url_hash in self.current_session_visited:
                if url_hash not in self.existing_visited:  # Only add new URLs
                    new_visited[url_hash] = self.scraped_data[url_hash]["url"]
            
            # Merge with existing visited URLs and save
            self.existing_visited.update(new_visited)
            self.save_json(self.existing_visited, self.visited_file)
            self.save_json(self.scraped_data, self.output_file)
            
            print(f"Crawl complete. {len(self.current_session_visited)} new URLs crawled.")
            print(f"Data saved in {self.output_file} and visited URLs in {self.visited_file}")


# Initialize with multiple seed URLs
seed_urls = [
    "https://marvel.fandom.com/wiki/Howard_the_Duck",
    "https://marvel.fandom.com/wiki/Jeffrey_(Land_Shark)_(Earth-616)",
    "https://marvel.fandom.com/wiki/Avengers:_Roll_Call_Vol_1_1"
]

# Run the crawler
crawler = WebCrawler(seed_urls, max_depth=2, time_limit=1800)
crawler.start_crawl()
