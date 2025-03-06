import json
import time
import numpy as np
import faiss
import torch
from transformers import BertTokenizer, BertModel
import argparse
from tqdm import tqdm  

class BERTIndexer:
    def __init__(self, input_file, index_file='bert_fandom_index.faiss', mapping_file='doc_id_mapping.json', batch_size=16):
        self.input_file = input_file
        self.index_file = index_file
        self.mapping_file = mapping_file
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.passages = []
        self.index = None
        self.doc_ids = []
        self.batch_size = batch_size  

    def load_data(self):
        """ Load the input data from the JSON file. """
        print(f"Loading data from {self.input_file}")
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.passages = [(doc_id, doc['content']) for doc_id, doc in data.items()]
        print(f"Loaded {len(self.passages)} passages.")

    def get_bert_embedding(self, text):
        """ Get the BERT embedding for a given passage. """
        tokens = self.tokenizer(text, max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        with torch.no_grad():
            output = self.model(**tokens)
            cls_embedding = output.last_hidden_state[:, 0, :]  # Use [CLS] token
        return cls_embedding.squeeze().numpy()  # shape (768,)

    def create_faiss_index(self):
        """ Create and populate a FAISS index with BERT embeddings. """
        dimension = 768  # BERT base output dimension
        self.index = faiss.IndexFlatL2(dimension)
        
        embeddings = []
        self.doc_ids = []

        # Process data in batches for faster indexing
        for i in tqdm(range(0, len(self.passages), self.batch_size), desc="Indexing Passages"):
            batch = self.passages[i:i + self.batch_size]
            batch_embeddings = []

            for doc_id, passage in batch:
                embedding = self.get_bert_embedding(passage)
                batch_embeddings.append(embedding)
                self.doc_ids.append(doc_id)

            batch_embeddings = np.array(batch_embeddings).astype('float32')
            embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings).astype('float32')
        self.index.add(embeddings)

    def save_index_and_mapping(self):
        """ Save the FAISS index and the doc_id mapping. """
        faiss.write_index(self.index, self.index_file)
        with open(self.mapping_file, 'w') as f:
            json.dump(self.doc_ids, f)

        print(f"Saved FAISS index to {self.index_file}")
        print(f"Saved document ID mapping to {self.mapping_file}")

    def search(self, query, top_k=5):
        """ Search the FAISS index using a query and return the top-k results. """
        query_embedding = self.get_bert_embedding(query).reshape(1, -1).astype('float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, dist in zip(indices[0], distances[0]):
            doc_id = self.doc_ids[idx]
            passage_text = next(p for doc, p in self.passages if doc == doc_id)
            results.append((doc_id, passage_text[:500], dist))  # show snippet
        return results

    def index_and_search(self):
        """ Perform indexing and allow searching. """
        # Step 1: Load data
        self.load_data()

        # Step 2: Create FAISS index with BERT embeddings
        print("Starting BERT indexing...")
        start_time = time.time()
        self.create_faiss_index()
        end_time = time.time()
        indexing_time = end_time - start_time
        print(f"Indexing complete in {indexing_time:.2f} seconds")

        # Step 3: Save the index and mapping
        self.save_index_and_mapping()

        # Step 4: Search interface
        print("\nYou can now search your data!")
        while True:
            query = input("\nEnter your search query (or type 'exit' to quit): ").strip()
            if query.lower() == 'exit':
                break

            results = self.search(query, top_k=5)
            print("\nTop results:")
            for doc_id, snippet, dist in results:
                print(f"Doc: {doc_id}\nSnippet: {snippet}\nDistance: {dist:.4f}\n")


# ============================
# Main Function with Command Line Arguments
# ============================
def main(args):
    # Create BERTIndexer instance
    indexer = BERTIndexer(input_file=args.input_file, 
                          index_file=args.index_file, 
                          mapping_file=args.mapping_file, 
                          batch_size=args.batch_size)

    # Perform indexing and search
    indexer.index_and_search()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='marvel_aarav1.json', help='Path to the input JSON file')
    parser.add_argument('--index_file', type=str, default='bert_fandom_index.faiss', help='Path to save FAISS index')
    parser.add_argument('--mapping_file', type=str, default='doc_id_mapping.json', help='Path to save document ID mapping')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for indexing')

    args = parser.parse_args()
    main(args)
