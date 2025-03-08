import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse

class BERTQueryEngine:
    def __init__(self, index_file='bert_fandom_index.faiss', mapping_file='doc_id_mapping.json'):
        """Load FAISS index and document mappings."""
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # Same model used for indexing
        self.index = faiss.read_index(self.index_file)
        self.doc_mapping = self.load_mapping()
        self.synonyms = {
            'avengers': ['marvel superheroes', 'marvel heroes', 'superheroes', 'avenger team'],
            'iron man': ['tony stark', 'ironman', 'stark industries'],
            'batman': ['dark knight', 'bruce wayne', 'gotham'],
            'superman': ['clark kent', 'man of steel', 'kryptonian'],
            'wonder woman': ['diana prince', 'amazonian', 'themyscira']
        }

    def load_mapping(self):
        """Load document-to-ID mapping including URLs and quality scores."""
        with open(self.mapping_file, 'r') as f:
            return json.load(f)

    def get_bert_embedding(self, text):
        """Generate an embedding using SentenceTransformer."""
        return self.model.encode(text).astype('float32')

    def categorize_query_length(self, query):
        """Categorize query length into Short, Medium, or Long."""
        tokens = query.split()
        length = len(tokens)
        if length <= 2:
            return "Short"
        elif length <= 5:
            return "Medium"
        else:
            return "Long"

    def expand_query(self, query):
        """Expand query using synonyms to improve search."""
        query_tokens = query.lower().split()
        expanded_query = []
        for token in query_tokens:
            expanded_query.append(token)
            if token in self.synonyms:
                expanded_query.extend(self.synonyms[token])
        return " ".join(expanded_query)

    def search(self, query, top_k=5, filter_quality=True):
        """Search the FAISS index for the top K relevant results."""
        print(f"Searching for query: {query}")
    
        query_embedding = self.embed_query(query)
        print(f"Query embedding: {query_embedding}")  # Print query embedding for debugging

        results = self.index.search(query_embedding, top_k)
        print(f"Search results: {results}")
        expanded_query = self.expand_query(query)
        query_embedding = self.get_bert_embedding(expanded_query).reshape(1, -1)

        distances, indices = self.index.search(query_embedding, top_k * 3)  # Retrieve more for filtering

        query_category = self.categorize_query_length(query)
        print(f"\n[Query Length: {len(query.split())} tokens] [Category: {query_category}]")

        results = []
        seen_docs = set()

        for idx, dist in zip(indices[0], distances[0]):
            if idx >= len(self.doc_mapping['doc_ids']):
                continue  # Skip invalid indices

            doc_id = self.doc_mapping['doc_ids'][idx]

            if doc_id in seen_docs:
                continue  # Skip duplicates
            seen_docs.add(doc_id)

            passage_text, quality_score, url = self.get_passage_details(doc_id)

            if passage_text is None:
                continue  # Skip if we can't find content

            results.append((doc_id, passage_text[:500], dist, quality_score, url))  # Show first 500 chars of text

        if filter_quality:
            results = sorted(results, key=lambda x: (-x[3], x[2]))  # Sort by Quality desc, Distance asc
        else:
            results = sorted(results, key=lambda x: x[2])  # Sort by Distance asc

        return results[:top_k]

    def get_passage_details(self, doc_id):
        """Retrieve passage text, quality score, and URL by document ID."""
        if doc_id not in self.doc_mapping:
            return None, 0, "No URL Available"

        doc_info = self.doc_mapping[doc_id]  # Directly access by doc_id
        passage_text = doc_info.get('content', 'No content available')
        quality_score = doc_info.get('quality_score', 0)  # If quality score is missing, default to 0
        url = doc_info.get('url', 'No URL Available')

        return passage_text, quality_score, url


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, default='bert_fandom_index.faiss', help='Path to FAISS index file')
    parser.add_argument('--mapping_file', type=str, default='doc_id_mapping.json', help='Path to doc mapping file')
    args = parser.parse_args()

    engine = BERTQueryEngine(index_file=args.index_file, mapping_file=args.mapping_file)

    while True:
        query = input("\nEnter search query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        filter_quality = input("Filter by quality? (y/n): ").strip().lower() == 'y'
        results = engine.search(query, top_k=10, filter_quality=filter_quality)

        print("\nTop results:")
        for doc_id, snippet, dist, quality, url in results:
            print(f"Doc: {doc_id}\nSnippet: {snippet}\nDistance: {dist:.4f} | Quality Score: {quality:.2f}\nURL: {url}\n")

if __name__ == "__main__":
    main()
