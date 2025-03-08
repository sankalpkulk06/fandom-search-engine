import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Preload the data (or load it dynamically)
with open('data.json', 'r') as file:
    data = json.load(file)

# Function to compute cosine similarity
def compute_cosine_similarity(query_embedding, doc_embedding):
    return cosine_similarity([query_embedding], [doc_embedding])[0][0]

# Function to handle search query
def search(query, embeddings, filter_quality=False):
    # Compute query embedding (this would be an actual model in practice)
    query_embedding = np.random.rand(768)  # Placeholder: replace with real query embedding

    # Store results
    results = []

    # Iterate through the documents and calculate similarities
    for doc_id, doc_data in embeddings.items():
        doc_embedding = doc_data['embedding']
        similarity_score = compute_cosine_similarity(query_embedding, doc_embedding)
        
        # Compute "Quality Score" based on some function, could be length or other metrics
        quality_score = doc_data['length'] * similarity_score  # Placeholder function
        
        results.append({
            'doc_id': doc_id,
            'url': doc_data['url'],
            'snippet': doc_data['snippet'],
            'distance': similarity_score,
            'quality_score': quality_score,
        })
    
    # Sort results by distance (ascending) and quality score (descending)
    results = sorted(results, key=lambda x: (x['distance'], -x['quality_score']), reverse=True)

    # Output the results
    print(f"\n[Query Length: {len(query.split())} tokens] [Category: Short]\n")
    print("Top results:")

    for result in results[:5]:  # Limit to top 5 results
        print(f"Doc: {result['doc_id']}")
        print(f"URL: {result['url']}")
        print(f"Snippet: {result['snippet']}")
        print(f"Distance: {result['distance']:.4f} | Quality Score: {result['quality_score']:.2f}")
        print()

# Main function
def main():
    while True:
        query = input("Enter your search query (or type 'exit' to quit): ").strip()
        if query.lower() == 'exit':
            break
        
        filter_quality = input("Filter by quality? (y/n): ").strip().lower() == 'y'
        
        # Call the search function
        search(query, data, filter_quality)

if __name__ == '__main__':
    main()
