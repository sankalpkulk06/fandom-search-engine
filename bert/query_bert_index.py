import json
import numpy as np
import faiss
import argparse
from sentence_transformers import SentenceTransformer

def load_index_and_mapping(index_file, mapping_file):
    """Load the FAISS index and document mapping."""
    index = faiss.read_index(index_file)
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    return index, mapping

def get_query_embedding(query, model):
    """Generate an embedding for the query."""
    return model.encode(query).astype('float32')

def search_index(query, index, mapping, model, top_k=5):
    """Search FAISS index and return top-k results."""
    query_embedding = get_query_embedding(query, model).reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx == -1:
            continue  # Skip if FAISS returns an invalid index
        
        doc_id = mapping['doc_ids'][idx]
        url = mapping['urls'][idx]
        quality_score = mapping['quality_scores'].get(doc_id, 0)
        snippet = "..."  # Placeholder, could extract from stored content
        
        results.append({
            'rank': i + 1,
            'doc_id': doc_id,
            'url': url,
            'snippet': snippet,
            'distance': float(distances[0][i]),
            'quality_score': quality_score
        })
    
    return results

def main(args):
    """Main function to load index and perform search."""
    print("Loading index and mappings...")
    index, mapping = load_index_and_mapping(args.index_file, args.mapping_file)
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    print("Enter your search query:")
    query = input().strip()
    results = search_index(query, index, mapping, model, args.top_k)
    
    print("\nSearch Results:")
    for res in results:
        print(f"Rank {res['rank']}: [Doc ID: {res['doc_id']}] {res['url']}")
        print(f"   Distance: {res['distance']:.4f}, Quality Score: {res['quality_score']:.2f}")
        print(f"   Snippet: {res['snippet']}")
        print("-")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, default='bert_fandom_index.faiss', help='Path to FAISS index file')
    parser.add_argument('--mapping_file', type=str, default='doc_id_mapping.json', help='Path to document ID mapping')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to return')
    
    args = parser.parse_args()
    main(args)
