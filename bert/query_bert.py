import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse
import os
import streamlit as st

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

def main():
    """Interactive loop for querying the FAISS index."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, default='bert_fandom_index.faiss', help='Path to the FAISS index file')
    parser.add_argument('--mapping_file', type=str, default='doc_id_mapping.json', help='Path to the doc ID mapping file')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory where scraped data files is stored')
    
    args = parser.parse_args()

    indexer = BERTIndexer(index_file=args.index_file, mapping_file=args.mapping_file, data_dir=args.data_dir)

    while True:
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            print("Exiting the search...")
            break

        # Query the index for the provided query
        results = indexer.query_index(query, top_k=5)

        print("\nSearch Results:")
        for rank, result in enumerate(results, 1):
            print(f"Rank {rank}: [Doc ID: {result['doc_id']}] {result['url']}")
            print(f"   Distance: {result['distance']:.4f}, Quality Score: {result['quality_score']:.2f}")
            print(f"   Snippet: ... {result['snippet']} ...")
            print("-" * 50)


if __name__ == "__main__":
    main()
