import argparse
import logging
from pathlib import Path
from index import LuceneIndexer  # 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('lucene_indexer.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        # Argument parser setup
        parser = argparse.ArgumentParser(description="Lucene Indexer and Searcher")
        parser.add_argument("--max_workers", type=int, default=4, help="Number of threads for indexing")
        parser.add_argument("--search_query", type=str, required=True, help="Search query for the index")

        args = parser.parse_args()
        max_workers = args.max_workers
        search_query = args.search_query

        # Initialize indexer with user-specified max_workers
        indexer = LuceneIndexer(
            max_workers=max_workers,
            commit_batch=1000,
            ram_buffer_size_mb=256
        )

        json_files = [f"marvel_aarav{i}.json" for i in range(1, 17)]
        json_files.append("marvel_aarav.json")

        # Validate files before processing
        valid_files = [json_file for json_file in json_files if Path(json_file).is_file()]
        
        if not valid_files:
            logger.error("No valid JSON files found")
            exit(1)

        # Process files
        indexer.index_documents(valid_files)

        # Execute search query
        indexer.search(search_query)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
