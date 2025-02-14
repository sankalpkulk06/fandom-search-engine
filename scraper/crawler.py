import sys
import os
from spider import WebCrawler  # WebCrawler in spider.py

if len(sys.argv) != 5:
    print("Usage: python crawler.py <seed-urls> <max-depth> <time-limit> <output-file.json>")
    sys.exit(1)

# Extract command-line arguments
seed_urls = sys.argv[1].split()  
max_depth = int(sys.argv[2])     
time_limit = int(sys.argv[3])     
output_file = sys.argv[4]

# Initialize and run crawler
crawler = WebCrawler(seed_urls, max_depth=2, time_limit=1800, output_file="marvel_sankalp3.json")
crawler.start_crawl()
