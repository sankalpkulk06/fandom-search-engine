import json
import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel

class BERTQueryEngine:
    def __init__(self, index_file, mapping_file):
        self.index_file = index_file
        self.mapping_file = mapping_file

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

        self.index = faiss.read_index(self.index_file)

        with open(self.mapping_file, 'r') as f:
            self.doc_ids = json.load(f)

        self.passages = self._load_passages()

    def _load_passages(self):
        """
        Optional: Load full passage data if you want to show snippets with results.
        You could load from the same JSON file you indexed, or just store them separately.
        """
        # Example (replace with actual loading if needed):
        passages = {}
        # Load the JSON where passages were originally stored.
        # If you stored passages separately, you can load them here.
        with open("../scraper/data/marvel_aarav1.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
            for doc_id, doc in data.items():
                passages[doc_id] = doc['content']
        return passages

    def get_bert_embedding(self, text):
        """Get BERT embedding for the query text."""
        tokens = self.tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            output = self.model(**tokens)
            cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token
        return cls_embedding.squeeze().numpy().astype('float32')

    def search(self, query, top_k=5, filter_quality=True):
        """Search FAISS index and rank results (optionally using quality scores)."""
        expanded_query = self.expand_query(query)
        query_embedding = self.get_bert_embedding(expanded_query).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k * 3)

        query_category = self.categorize_query_length(query)
        print(f"\n[Query Length: {len(query.split())} tokens] [Category: {query_category}]")

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc_id = self.doc_ids[idx]
            passage_text, quality_score = self.get_passage_and_score(doc_id)
            url = self.doc_urls.get(doc_id, "URL not available")  # Load URL from mapping

            results.append((doc_id, passage_text[:1000], dist, quality_score, url))  # Add URL

        if filter_quality:
            results = sorted(results, key=lambda x: (-x[3], x[2]))  # Quality desc, Distance asc
        else:
            results = sorted(results, key=lambda x: x[2])  # Only Distance asc

        return results[:top_k]



if __name__ == '__main__':
    engine = BERTQueryEngine(
        index_file='marvel_dc_bert_index.faiss',
        mapping_file='marvel_dc_doc_mapping.json'
    )

    while True:
        query = input("\nEnter search query (or 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break

        results = engine.search(query, top_k=5)

        print("\nTop Search Results:")
        for doc_id, snippet, dist in results:
            print(f"Doc ID: {doc_id}\nSnippet: {snippet}\nDistance: {dist:.4f}\n")
