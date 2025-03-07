import json
import time
import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel
import argparse
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords from NLTK (run this once)
nltk.download('stopwords')

class BERTIndexer:
    def __init__(self, input_file, index_file='bert_fandom_index.faiss', mapping_file='doc_id_mapping.json'):
        """Initialize with file paths and load BERT tokenizer & model."""
        self.input_file = input_file
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.passages = []
        self.index = None
        self.doc_ids = []
        self.quality_scores = {}
        self.stop_words = set(stopwords.words('english'))  # Set of stopwords

    def clean_content(self, text):
        """
        Preprocess content to:
        - Fix missing spaces between sections (e.g., GalleryName -> Gallery Name)
        - Remove newlines, tabs, and excessive spaces
        - Remove stopwords
        """
        # Insert space between lowercase+Uppercase
        text = re.sub(r'(?<=[a-z])([A-Z][a-z])', r' \1', text)

        # Fix cases like "something:description" -> "something: description"
        text = re.sub(r'([a-zA-Z0-9])([:])', r'\1 \2 ', text)

        # Remove newlines and tabs
        text = text.replace('\n', ' ').replace('\t', ' ')

        # Collapse multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters (retain only basic punctuation)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"()-]', '', text)

        # Remove excessive punctuation like multiple "!!!" or "..."
        text = re.sub(r'([!?.])\1+', r'\1', text)

        # Remove stopwords
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        cleaned_text = ' '.join(filtered_words)

        # Trim leading/trailing spaces
        return cleaned_text.strip()

    def load_data(self):
        """Load JSON data, clean passages, compute quality, and filter low-quality ones."""
        print(f"Loading data from {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for doc_id, doc in data.items():
            content = self.clean_content(doc['content'])
            quality_score = self.compute_quality_score(content)
            if quality_score > 0:  # Filter out low-quality passages
                self.passages.append((doc_id, content, quality_score))
                self.quality_scores[doc_id] = quality_score

        print(f"Loaded {len(self.passages)} quality passages (filtered).")

    def compute_quality_score(self, text):
        """Compute content quality score (sentence count + scaled length)."""
        sentences = text.split('.')
        sentence_count = sum(1 for s in sentences if len(s.strip()) > 10)
        length_score = len(text)

        if sentence_count == 0:
            return 0

        return sentence_count + (length_score / 1000)

    def get_bert_embedding(self, text):
        """Generate a BERT embedding (768-dim) for a given text."""
        tokens = self.tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            output = self.model(**tokens)
            cls_embedding = output.last_hidden_state[:, 0, :]  # CLS token embedding
        return cls_embedding.squeeze().numpy()

    def create_faiss_index(self):
        """Build FAISS index from all passage embeddings."""
        dimension = 768  # BERT embedding size
        self.index = faiss.IndexFlatL2(dimension)

        embeddings = []
        self.doc_ids = []

        start_time = time.time()

        for idx, (doc_id, passage, _) in enumerate(self.passages):
            embedding = self.get_bert_embedding(passage)
            embeddings.append(embedding)
            self.doc_ids.append(doc_id)

            if idx % 100 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {idx}/{len(self.passages)} passages in {elapsed:.2f}s")

        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)

    def save_index_and_mapping(self):
        """Save FAISS index and doc-to-ID mapping (with quality scores)."""
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, 'w') as f:
            json.dump({
                'doc_ids': self.doc_ids,
                'quality_scores': self.quality_scores
            }, f)

        print(f"Saved FAISS index to {self.index_file}")
        print(f"Saved document ID mapping and quality scores to {self.mapping_file}")

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

    def search(self, query, top_k=5, filter_quality=True):
        """Search FAISS index and rank results (optionally using quality scores)."""
        query_embedding = self.get_bert_embedding(query).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k * 3)

        query_category = self.categorize_query_length(query)
        print(f"\n[Query Length: {len(query.split())} tokens] [Category: {query_category}]")

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc_id = self.doc_ids[idx]
            passage_text, quality_score = self.get_passage_and_score(doc_id)
            results.append((doc_id, passage_text[:500], dist, quality_score))

        if filter_quality:
            results = sorted(results, key=lambda x: (-x[3], x[2]))  # Quality desc, Distance asc
        else:
            results = sorted(results, key=lambda x: x[2])  # Only Distance asc

        return results[:top_k]

    def get_passage_and_score(self, doc_id):
        """Retrieve passage text and quality score by doc_id."""
        for d_id, passage, quality in self.passages:
            if d_id == doc_id:
                return passage, self.quality_scores[doc_id]
        return "", 0

    def index_and_search(self):
        """Full pipeline: load, clean, index, save, then enable interactive search."""
        self.load_data()

        print("Starting BERT indexing...")
        start_time = time.time()
        self.create_faiss_index()
        print(f"Indexing complete in {time.time() - start_time:.2f} seconds")

        self.save_index_and_mapping()

        print("\nYou can now search your data!")
        while True:
            query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                break

            filter_quality = input("Filter by quality? (y/n): ").strip().lower() == 'y'

            results = self.search(query, top_k=5, filter_quality=filter_quality)
            print("\nTop results:")
            for doc_id, snippet, dist, quality in results:
                print(f"Doc: {doc_id}\nSnippet: {snippet}\nDistance: {dist:.4f} | Quality Score: {quality:.2f}\n")


def main(args):
    """Entry point to initialize BERTIndexer and run indexing + search."""
    indexer = BERTIndexer(input_file=args.input_file, 
                          index_file=args.index_file, 
                          mapping_file=args.mapping_file)

    indexer.index_and_search()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help='Path to input JSON file')
    parser.add_argument('--index_file', type=str, default='bert_fandom_index.faiss', help='Path to save FAISS index')
    parser.add_argument('--mapping_file', type=str, default='doc_id_mapping.json', help='Path to save document ID mapping')

    args = parser.parse_args()
    main(args)
