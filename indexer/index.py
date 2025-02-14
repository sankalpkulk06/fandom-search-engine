import json
import lucene
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Any, Generator
from dataclasses import dataclass
from pathlib import Path
from contextlib import contextmanager
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, TextField, StringField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser
from org.apache.lucene.search import IndexSearcher, BooleanQuery, BooleanClause
from org.apache.lucene.store import SimpleFSDirectory
from java.io import File

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('lucene_indexer.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Initialize JVM globally
# lucene.initVM(vmargs=['-Djava.awt.headless=true'])

@dataclass
class IndexingStats:
    """Statistics for the indexing process"""
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def elapsed_time(self) -> float:
        """Calculate elapsed time in seconds"""
        if self.end_time == 0.0:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def start(self):
        """Start timing"""
        self.start_time = time.time()

    def stop(self):
        """Stop timing"""
        self.end_time = time.time()

class LuceneIndexer:
    def __init__(self, 
                 index_dir: str = "marvel_index",
                 max_workers: int = 4,
                 commit_batch: int = 1000,
                 ram_buffer_size_mb: int = 256):
        """
        Initialize the Lucene Indexer with configurable parameters.
        
        Args:
            index_dir: Directory to store the Lucene index
            max_workers: Maximum number of worker threads for parallel processing
            commit_batch: Number of documents to process before committing
            ram_buffer_size_mb: Size of RAM buffer for indexing in MB
        """
        self.index_dir = Path(index_dir)
        self.max_workers = max_workers
        self.commit_batch = commit_batch
        self.ram_buffer_size_mb = ram_buffer_size_mb
        self.stats = IndexingStats()
        
        # Create index directory if it doesn't exist
        self.index_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initializing index in directory: {self.index_dir}")
        
        # Initialize Lucene components
        self._init_lucene()

    def _init_lucene(self):
        """Initialize Lucene components with optimized settings"""
        self.directory = SimpleFSDirectory(Paths.get(str(self.index_dir.resolve())))
        self.analyzer = StandardAnalyzer()
        
        # Configure IndexWriter with optimized settings
        config = IndexWriterConfig(self.analyzer)
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)
        config.setRAMBufferSizeMB(float(self.ram_buffer_size_mb))
        config.setUseCompoundFile(False)  # Disable compound file format for better performance
        
        self.writer = IndexWriter(self.directory, config)

    @contextmanager
    def _get_searcher(self):
        """Context manager for safely handling IndexSearcher resources"""
        reader = None
        try:
            reader = DirectoryReader.open(self.directory)
            searcher = IndexSearcher(reader)
            yield searcher
        finally:
            if reader:
                reader.close()

    def extract_character_info(self, content: str) -> Dict[str, Dict[str, Any]]:
        """Extract structured information from the content field with improved parsing"""
        info = {
            "basic_info": {},
            "appearance": {},
            "origin": {},
            "powers": {},
            "affiliations": {},
            "story": {}
        }
        
        if not content:
            return info
            
        lines = content.split('\n')
        current_section = None
        current_field = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Improved field detection with error handling
            try:
                # Map of field identifiers to their corresponding sections and fields
                field_mappings = {
                    "Name": ("basic_info", "name"),
                    "Current Alias": ("basic_info", "current_alias"),
                    "Aliases": ("basic_info", "aliases"),
                    "Gender": ("appearance", "gender"),
                    "Eyes": ("appearance", "eyes"),
                    "Skin": ("appearance", "skin"),
                    "Unusual Features": ("appearance", "features"),
                    "Origin": ("origin", "origin"),
                    "Living Status": ("origin", "status"),
                    "Reality": ("origin", "reality"),
                    "Powers:": ("powers", "powers", True)  # True indicates list field
                }
                
                for identifier, (section, field, *is_list) in field_mappings.items():
                    if identifier in line:
                        if is_list:
                            info[section][field] = []
                            current_section = section
                            current_field = field
                        else:
                            value = line.split(identifier, 1)[1].strip()
                            if value:
                                info[section][field] = value
                        break
                else:
                    if current_section == "powers" and current_field == "powers":
                        info[current_section][current_field].append(line)
                        
            except Exception as e:
                logger.warning(f"Error processing line '{line}': {str(e)}")
                continue
                
        return info

    def process_document(self, args: tuple) -> Optional[Document]:
        """Process a single document with improved error handling"""
        if not lucene.getVMEnv().isCurrentThreadAttached():
            lucene.getVMEnv().attachCurrentThread()
            
        key, value = args
        try:
            doc = Document()
            doc.add(StringField("id", key, Field.Store.YES))

            # Process URL field
            if url := value.get("url"):
                doc.add(TextField("url", url, Field.Store.YES))

            # Process content and extract character info
            if content := value.get("content"):
                doc.add(TextField("content", content, Field.Store.YES))
                char_info = self.extract_character_info(content)
                
                # Add extracted fields to document
                for section, section_data in char_info.items():
                    for field, field_value in section_data.items():
                        if isinstance(field_value, list):
                            field_value = "; ".join(field_value)
                        if field_value:
                            doc.add(TextField(f"{section}.{field}", str(field_value), Field.Store.YES))

            return doc
            
        except Exception as e:
            logger.error(f"Error processing document {key}: {str(e)}")
            self.stats.failed_documents += 1
            return None

    def _process_batch(self, batch: List[tuple]) -> Generator[Document, None, None]:
        """Process a batch of documents in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.process_document, item) for item in batch]
            for future in as_completed(futures):
                try:
                    if doc := future.result():
                        yield doc
                except Exception as e:
                    logger.error(f"Error in document processing: {str(e)}")
                    self.stats.failed_documents += 1

    def index_documents(self, json_files: List[str]):
        """Index documents from multiple JSON files with improved error handling and progress tracking"""
        self.stats = IndexingStats()
        self.stats.start()
        
        try:
            for json_file in json_files:
                logger.info(f"Processing file: {json_file}")
                
                try:
                    with open(json_file, 'r', encoding='utf-8') as file:
                        data = json.load(file)
                except Exception as e:
                    logger.error(f"Error reading file {json_file}: {str(e)}")
                    continue
                
                self.stats.total_documents += len(data)
                logger.info(f"Found {len(data)} documents to index in {json_file}")
                
                # Process documents in batches
                batch = []
                for key, value in data.items():
                    batch.append((key, value))
                    
                    if len(batch) >= self.commit_batch:
                        self._process_and_commit_batch(batch)
                        batch = []
                
                # Process remaining documents
                if batch:
                    self._process_and_commit_batch(batch)
                
                logger.info(f"Completed processing {json_file}")
                
        except Exception as e:
            logger.error(f"Fatal error during indexing: {str(e)}")
            raise
        finally:
            self._cleanup()
            self._log_statistics()

    def _process_and_commit_batch(self, batch: List[tuple]):
        """Process and commit a batch of documents"""
        try:
            for doc in self._process_batch(batch):
                self.writer.addDocument(doc)
                self.stats.processed_documents += 1
                
            self.writer.commit()
            logger.info(f"Processed {self.stats.processed_documents}/{self.stats.total_documents} documents")
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            self.writer.rollback()

    def _cleanup(self):
        """Cleanup resources"""
        try:
            self.writer.close()
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self.stats.stop()

    def _log_statistics(self):
        """Log indexing statistics"""
        logger.info("\nIndexing Statistics:")
        logger.info(f"Total documents: {self.stats.total_documents}")
        logger.info(f"Successfully processed: {self.stats.processed_documents}")
        logger.info(f"Failed documents: {self.stats.failed_documents}")
        logger.info(f"Total time: {self.stats.elapsed_time:.2f} seconds")
        logger.info(f"Average processing rate: {self.stats.processed_documents / self.stats.elapsed_time:.2f} docs/second")

    def search(self, query_str: str, num_results: int = 10, fields: Optional[List[str]] = None):
        """Enhanced search with multi-field support and improved error handling"""
        try:
            with self._get_searcher() as searcher:
                logger.info(f"\nExecuting search for: '{query_str}'")
                logger.info(f"Index contains {searcher.getIndexReader().numDocs()} documents")
                
                # Create multi-field query if fields are specified
                if fields:
                    parser = MultiFieldQueryParser(fields, self.analyzer)
                else:
                    parser = QueryParser("content", self.analyzer)
                
                query = parser.parse(query_str)
                results = searcher.search(query, num_results)
                
                if len(results.scoreDocs) == 0:
                    logger.info("No results found.")
                    return
                
                self._print_search_results(searcher, results.scoreDocs)
                
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            raise

    def _print_search_results(self, searcher: IndexSearcher, hits: List):
        """Format and print search results"""
        print("\nSearch Results:")
        print("=" * 50)
        
        for hit in hits:
            doc = searcher.doc(hit.doc)
            formatted_results = self.format_results(doc)
            
            print(f"\nMatch Score: {hit.score:.2f}")
            print("-" * 50)
            
            for section, details in formatted_results.items():
                non_empty_fields = {k: v for k, v in details.items() if v and v != "Unknown"}
                if non_empty_fields:
                    print(f"\n{section}:")
                    for key, value in non_empty_fields.items():
                        print(f"  • {key}: {value}")
            
            print("\n" + "=" * 50)

    def format_results(self, doc):
        """Format search results in the requested structure"""
        sections = {
            "Basic Character Information": {
                "Name": doc.get("basic_info.name") or "Unknown",
                "Current Alias": doc.get("basic_info.current_alias") or "Unknown",
                "Aliases": doc.get("basic_info.aliases") or "Unknown",
                "Affiliations": doc.get("affiliations") or "Unknown"
            },
            "Appearance and Physical Traits": {
                "Gender": doc.get("appearance.gender") or "Unknown",
                "Eye Color": doc.get("appearance.eyes") or "Unknown",
                "Skin Colors": doc.get("appearance.skin") or "Unknown",
                "Notable Features": doc.get("appearance.features") or "Unknown"
            },
            "Origin and Status": {
                "Origin": doc.get("origin.origin") or "Unknown",
                "Living Status": doc.get("origin.status") or "Unknown",
                "Reality": doc.get("origin.reality") or "Unknown"
            },
            "Powers & Abilities": {
                "Powers": doc.get("powers.powers") or "Unknown"
            }
        }
        return sections

# if __name__ == "__main__":
#     try:
#         # Initialize indexer with optimized settings
#         indexer = LuceneIndexer(
#             max_workers=4,
#             commit_batch=1000,
#             ram_buffer_size_mb=256
#         )
        
#         json_files = [f"marvel_aarav{i}.json" for i in range(1, 17)]
#         json_files.append("marvel_aarav.json")
        
#         # Validate files before processing
#         valid_files = []
#         for json_file in json_files:
#             if Path(json_file).is_file():
#                 valid_files.append(json_file)
#             else:
#                 logger.warning(f"File not found: {json_file}")
        
#         if not valid_files:
#             logger.error("No valid JSON files found")
#             exit(1)
            
#         # Process files
#         indexer.index_documents(valid_files)
        
#         # Example searches
#         indexer.search("Shang Chi")
#         indexer.search("Spider-Man") #, fields=["basic_info.name", "basic_info.aliases"])
        
#     except Exception as e:
#         logger.error(f"Fatal error: {str(e)}")
#         raise
