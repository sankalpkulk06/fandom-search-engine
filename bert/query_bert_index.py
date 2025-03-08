import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse

class BERTSearcher:
    def __init__(self, index_file='bert_fandom_index.faiss', mapping_file='doc_id_mapping.json'):
        """Initialize the searcher by loading the FAISS index and metadata."""
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # Same model used in indexing

        # Load FAISS index
        self.index = faiss.read_index(self.index_file)

        # Load metadata (doc IDs, quality scores, URLs, passages)
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
            self.doc_ids = mapping_data['doc_ids']
            self.quality_scores = mapping_data['quality_scores']
            self.urls = mapping_data['urls']

        # Load passages separately for snippet extraction
        self.passages = self.load_passages()

        print("Loaded index and mappings.")

    def load_passages(self):
        """Load passages from original JSON files using doc_ids."""
        passages = {}
        for filename in self.doc_ids:  # Assuming doc_ids match the filenames in input_dir
            json_file = f"data/{filename}.json"  # Adjust path as necessary
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for doc_id, doc in data.items():
                        passages[doc_id] = doc['content']
            except FileNotFoundError:
                continue  # Skip missing files
        return passages

    def search(self, query, top_k=5):
        """Search FAISS index and return top-k results with snippets."""
        query_embedding = self.model.encode(query).astype('float32').reshape(1, -1)
        distances, indices = self.index.search(query_embedding, top_k)

        print("\nSearch Results:")
        for rank, (idx, distance) in enumerate(zip(indices[0], distances[0]), 1):
            if idx == -1:
                continue  # Ignore invalid indices

            doc_id = self.doc_ids[idx]
            url = self.urls[idx]
            quality_score = self.quality_scores.get(doc_id, 0)
            passage_text = self.passages.get(doc_id, "")

            # Extract a snippet containing the most relevant part
            snippet = self.extract_snippet(query, passage_text)

            print(f"Rank {rank}: [Doc ID: {doc_id}] {url}")
            print(f"   Distance: {distance:.4f}, Quality Score: {quality_score:.2f}")
            print(f"   Snippet: {snippet}")
            print("-")

    def extract_snippet(self, query, passage_text, window=25):
        """Extracts a snippet around the most relevant query term in the passage."""
        if not passage_text:
            return "Snippet not available."

        words = passage_text.split()
        query_words = query.lower().split()
        
        # Find the first occurrence of any query word
        for i, word in enumerate(words):
            if any(qw in word.lower() for qw in query_words):
                start = max(0, i - window // 2)
                end = min(len(words), i + window // 2)
                return "... " + " ".join(words[start:end]) + " ..."
        
        # Fallback: return first 'window' words
        return "... " + " ".join(words[:window]) + " ..."


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, default='bert_fandom_index.faiss', help='Path to FAISS index file')
    parser.add_argument('--mapping_file', type=str, default='doc_id_mapping.json', help='Path to document ID mapping file')
    args = parser.parse_args()

    searcher = BERTSearcher(index_file=args.index_file, mapping_file=args.mapping_file)

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() in ['exit', 'quit']:
            print("Exiting search.")
            break
        searcher.search(query)

if __name__ == "__main__":
    main()
