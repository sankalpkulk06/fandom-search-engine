import json
import time
import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel
import argparse
import re
import os
from sentence_transformers import SentenceTransformer

class BERTIndexer:
    def __init__(self, input_dir, index_file='bert_fandom_index.faiss', mapping_file='doc_id_mapping.json'):
        """Initialize with directory path and load SentenceTransformer model."""
        self.input_dir = input_dir
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = SentenceTransformer('paraphrase-distilroberta-base-v1')  # Fine-tuned model for semantic search
        self.passages = []
        self.index = None
        self.doc_ids = []
        self.quality_scores = {}
        self.urls = []
        self.synonyms = {
            'avengers': ['marvel superheroes', 'marvel heroes', 'superheroes', 'avenger team'],
            'iron man': ['tony stark', 'ironman', 'stark industries'],
            'batman': ['dark knight', 'bruce wayne', 'gotham'],
            'superman': ['clark kent', 'man of steel', 'kryptonian'],
            'wonder woman': ['diana prince', 'amazonian', 'themyscira']
        }

    def clean_content(self, text):
        """
        Preprocess content to:
        - Fix missing spaces between sections (e.g., GalleryName -> Gallery Name)
        - Remove newlines, tabs, and excessive spaces
        """
        # Insert space between lowercase+Uppercase
        text = re.sub(r'(?<=[a-z])([A-Z][a-z])', r' \1', text)

        # Fix cases like "something:description" -> "something: description"
        text = re.sub(r'([a-zA-Z0-9])([:])', r'\1 \2 ', text)

        # Remove newlines and tabs
        text = text.replace('\n', ' ').replace('\t', ' ')

        # Collapse multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text)

        # Trim leading/trailing spaces
        return text.strip()

    def load_data(self):
        """Load data from all JSON files in the directory, clean passages, compute quality, and filter low-quality ones."""
        print(f"Loading data from {self.input_dir}")
        
        for filename in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, filename)
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                for doc_id, doc in data.items():
                    content = self.clean_content(doc['content'])

                    # Exclude content with less than 50 words
                    if len(content.split()) < 50:
                        continue

                    quality_score = self.compute_quality_score(content)

                    if quality_score > 0.5:  # Reduced quality score threshold
                        self.passages.append((doc_id, content, quality_score))
                        self.quality_scores[doc_id] = quality_score
                        self.urls.append(doc['url'])  # Storing the URL as well

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
        embedding = self.model.encode(text)
        return embedding

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
                'quality_scores': self.quality_scores,
                'urls': self.urls  # Storing the URLs
            }, f)

        print(f"Saved FAISS index to {self.index_file}")
        print(f"Saved document ID mapping, quality scores, and URLs to {self.mapping_file}")

    def index_and_save(self):
        """Full pipeline: load, clean, index, save."""
        self.load_data()

        print("Starting BERT indexing...")
        start_time = time.time()
        self.create_faiss_index()
        print(f"Indexing complete in {time.time() - start_time:.2f} seconds")

        self.save_index_and_mapping()


def main(args):
    """Entry point to initialize BERTIndexer and run indexing."""
    indexer = BERTIndexer(input_dir=args.input_dir, 
                          index_file=args.index_file, 
                          mapping_file=args.mapping_file)

    indexer.index_and_save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to directory containing JSON files')
    parser.add_argument('--index_file', type=str, default='bert_fandom_index.faiss', help='Path to save FAISS index')
    parser.add_argument('--mapping_file', type=str, default='doc_id_mapping.json', help='Path to save document ID mapping')

    args = parser.parse_args()
    main(args)
