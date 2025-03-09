import tkinter as tk
from tkinter import scrolledtext, messagebox
# from lucene_search import LuceneSearcher  # Assuming the LuceneSearcher code is in 'lucene_search.py'

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



def perform_search():
    query = entry.get()
    if not query.strip():
        messagebox.showwarning("Warning", "Please enter a search query!")
        return
    
    try:
        results_box.config(state=tk.NORMAL)  # Enable editing
        results_box.delete(1.0, tk.END)  # Clear previous results
        
        searcher = LuceneSearcher()
        results = searcher.search(query, num_results=10)
        
        if not results:
            results_box.insert(tk.END, "No results found.\n")
        else:
            for result in results:
                results_box.insert(tk.END, f"ID: {result['id']}\n")
                results_box.insert(tk.END, f"URL: {result['url']}\n")
                results_box.insert(tk.END, f"Content: {result['content'][:300]}...\n")
                results_box.insert(tk.END, "-" * 50 + "\n")
        
        results_box.config(state=tk.DISABLED)  # Disable editing after inserting results
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# GUI setup
root = tk.Tk()
root.title("Lucene Search App")
root.geometry("600x400")

tk.Label(root, text="Enter search query:").pack(pady=5)
entry = tk.Entry(root, width=50)
entry.pack(pady=5)

tk.Button(root, text="Search", command=perform_search).pack(pady=5)

results_box = scrolledtext.ScrolledText(root, width=70, height=15, state=tk.DISABLED)
results_box.pack(pady=10)

root.mainloop()
