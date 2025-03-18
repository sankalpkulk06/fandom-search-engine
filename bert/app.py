import json
import faiss
import numpy as np
import time
from sentence_transformers import SentenceTransformer
import streamlit as st
import lucene
from lucene import initVM
import os
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.store import SimpleFSDirectory
from java.io import File

class BERTIndexer:
    def __init__(self, index_file='bert_fandom_index.faiss', mapping_file='doc_id_mapping.json', data_dir='../scraper/data'):
        """Initialize with index and mapping files."""
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.data_dir = data_dir
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # Same model as used for indexing
        self.index = faiss.read_index(self.index_file)
        
        # Load doc ID mapping
        with open(self.mapping_file, 'r') as f:
            mapping = json.load(f)
            self.doc_ids = mapping['doc_ids']
            self.quality_scores = mapping['quality_scores']
            self.urls = mapping['urls']
            
            # Debug the structure of doc_ids and urls
            # print(f"doc_ids: {self.doc_ids}")
            # print(f"urls: {self.urls}")
            # print(f"doc_ids type: {type(self.doc_ids)}")
            # print(f"urls type: {type(self.urls)}")

    def get_bert_embedding(self, text):
        """Generate a BERT embedding (768-dim) for a given text."""
        embedding = self.model.encode(text)
        return np.array(embedding).astype('float32')

    def get_snippet(self, doc_id):
        """Retrieve the snippet (first 50 words) for a document."""
        # Load the document from the data directory
        doc_path = self.get_doc_path(doc_id)
        if not doc_path:
            return f"Snippet for doc_id {doc_id} not found."
        
        with open(doc_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Retrieve the content associated with the doc_id
        content = data.get(doc_id, {}).get('content', '')
        if not content:
            return f"No content found for doc_id {doc_id}."

        # Extract the first 50 words from the content
        words = content.split()
        snippet = ' '.join(words[:30])  # Get the first 50 words
        return snippet + ('...' if len(words) > 50 else '')

    def get_doc_path(self, doc_id):
        """Return the file path for the document corresponding to the doc_id."""
        # Check each JSON file in the data directory for the doc_id
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if doc_id in data:
                        return file_path
        return None
    
    def query_index(self, query, top_k=5):
        """Query the FAISS index and return the top-k closest passages."""
        query_embedding = self.get_bert_embedding(query)
        distances, indices = self.index.search(np.array([query_embedding]), top_k)

        results = []
        for i in range(top_k):
            try:
                # Ensure the index is being used correctly
                if isinstance(self.doc_ids, list):
                    # Access directly if doc_ids is a list
                    doc_id = self.doc_ids[indices[0][i]]
                    url = self.urls[indices[0][i]]  # Similarly, use indices directly for urls
                else:
                    # Use str() for dictionary keys if doc_ids is a dictionary
                    doc_id = self.doc_ids[str(indices[0][i])]
                    url = self.urls[str(indices[0][i])]

                quality_score = self.quality_scores.get(doc_id, 0)
                snippet = self.get_snippet(doc_id)
                distance = distances[0][i]
                results.append({
                    'doc_id': doc_id,
                    'url': url,
                    'snippet': snippet,
                    'quality_score': quality_score,
                    'distance': distance
                })
            except Exception as e:
                print(f"Error processing index {i}: {e}")
                continue
        
        return results

class LuceneSearcher:
    def __init__(self, index_dir="../indexer/marvel_index"):
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
                return []

            search_results = []
            for hit in results.scoreDocs:
                doc = self.searcher.doc(hit.doc)
                search_results.append({
                    'score': hit.score,
                    'id': doc.get('id'),
                    'url': doc.get('url') or 'N/A',
                    'content_snippet': doc.get('content')[:300]  # Show first 300 characters
                })
            return search_results

        except Exception as e:
            print(f"Error during search: {str(e)}")
            return []

import asyncio
import concurrent.futures

async def run_bert_search(indexer, query, top_k=4):
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, indexer.query_index, query, top_k)
    return results

# async def run_lucene_search(searcher, query, num_results):
#     loop = asyncio.get_running_loop()
#     with concurrent.futures.ThreadPoolExecutor() as pool:
#         results = await loop.run_in_executor(pool, searcher.search, query, num_results)
#     return results


st.title("🦸 Superhero Search 🦹")
st.subheader("🔍 Find details about your favorite Marvel and DC superheroes! 💥")
indexer = BERTIndexer()


search_option = st.radio("Select Search Method", ('BERT', 'Lucene'))

#col1, col2 = st.columns([5, 1])
#with col1:
query = st.text_input("Enter your query", "", placeholder="Search any of your favourite Marvel or DC characters...")

#with col2:
search_button = st.button("Search")


if search_button and query:
    start_time = time.time()
    if search_option == 'BERT':
        
        
        with st.spinner(f"Searching for **{query}**... Please wait."):
            # results = indexer.query_index(query, top_k=4)
            results = asyncio.run(run_bert_search(indexer, query, 4))

        search_time = time.time() - start_time
    
        st.write(f"**Search Time:** {search_time:.2f} seconds")
        st.write(f"Searching for: **{query}**...")
        # Display the results in a clean format
        st.subheader("Search Results:")
        for rank, result in enumerate(results, 1):
            st.write(f"**Rank {rank}:**")
            st.write(f"{result['url']}")
            st.write(f"[Doc ID: {result['doc_id']}]")
            st.write(f"   Distance: {result['distance']:.4f}, Quality Score: {result['quality_score']:.2f}")
            st.write(f"   Snippet: ... {result['snippet']} ...")
            st.markdown("-" * 50)
    else:
        searcher = LuceneSearcher()

        with st.spinner(f"Searching for **{query}**... Please wait."):
            search_results = searcher.search(query, num_results=7)
            # search_results = asyncio.run(run_lucene_search(searcher, query, 4))

        if not search_results:
            st.write("No results found.")
        else:
            search_time = time.time() - start_time
    
            st.write(f"**Search Time:** {search_time:.2f} seconds")
            st.subheader("Search Results:")

            # Display the results
            for rank, result in enumerate(search_results, 1):
                st.write(f"**Rank {rank}:**")
                st.write(f"**ID:** {result['id']}")
                st.write(f"**URL:** {result['url']}")
                st.write(f"**Match Score:** {result['score']:.2f}")
                st.write(f"**Snippet:** {result['content_snippet']}...")

                st.markdown("-" * 50)
