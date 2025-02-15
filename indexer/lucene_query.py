import lucene
import os
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import SimpleFSDirectory
from java.io import File

class LuceneSearcher:
    def __init__(self, index_dir="marvel_index"):
        # Initialize JVM
        lucene.initVM()
        
        if not os.path.exists(index_dir):
            raise Exception(f"Index directory '{index_dir}' does not exist. Please index data first.")
        
        print(f"Loading index from directory: {index_dir}")
        self.directory = SimpleFSDirectory(File(index_dir).toPath())
        self.reader = DirectoryReader.open(self.directory)
        self.searcher = IndexSearcher(self.reader)
        self.analyzer = StandardAnalyzer()

    def search(self, query_str, num_results=10):
        try:
            print(f"\nExecuting search for: '{query_str}'")
            parser = QueryParser("content", self.analyzer)
            query = parser.parse(query_str)

            results = self.searcher.search(query, num_results)

            if len(results.scoreDocs) == 0:
                print("No results found.")
                return
            
            print("\nSearch Results:")
            print("=" * 50)

            for hit in results.scoreDocs:
                doc = self.searcher.doc(hit.doc)
                print(f"\nMatch Score: {hit.score:.2f}")
                print("-" * 50)
                print(f"ID: {doc.get('id')}")
                print(f"URL: {doc.get('url') or 'N/A'}")
                print(f"Content Snippet: {doc.get('content')[:300]}")  # Show first 300 characters
                print("\n" + "=" * 50)
            
        except Exception as e:
            print(f"Error during search: {str(e)}")
            raise

if __name__ == "__main__":
    try:
        searcher = LuceneSearcher()
        query = input("Enter search query: ")  # User input for query
        searcher.search(query)
    except Exception as e:
        print(f"Fatal error: {str(e)}")
