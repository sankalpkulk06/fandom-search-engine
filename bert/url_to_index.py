import os
import json

# Paths
DATA_DIR = "../scraper/data"  # Directory containing original JSON files
MAPPING_FILE = "doc_id_mapping.json"  # Existing document mapping file

def load_existing_mapping():
    """Load the existing document ID mapping (without URLs)."""
    if not os.path.exists(MAPPING_FILE):
        print(f"Error: {MAPPING_FILE} not found!")
        return None

    with open(MAPPING_FILE, 'r') as f:
        mapping = json.load(f)

    if 'doc_ids' not in mapping:
        print("Error: Invalid mapping file format.")
        return None

    return mapping

def extract_urls_from_json():
    """Extract URLs from JSON data files."""
    urls = {}

    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for doc_id, doc in data.items():
                    urls[doc_id] = doc.get('url', 'URL not found')
        except Exception as e:
            print(f"Error processing {file_name}: {e}")

    return urls

def update_mapping_with_urls(mapping, urls):
    """Update document mapping with extracted URLs."""
    mapping['doc_urls'] = {}

    for doc_id in mapping['doc_ids']:
        mapping['doc_urls'][doc_id] = urls.get(doc_id, 'URL not found')

    return mapping

def save_updated_mapping(mapping):
    """Save the updated document ID mapping with URLs."""
    with open(MAPPING_FILE, 'w') as f:
        json.dump(mapping, f, indent=4)

    print(f"Updated {MAPPING_FILE} with URLs.")

def main():
    """Main function to add URLs to the existing index mapping."""
    print("Loading existing document mapping...")
    mapping = load_existing_mapping()
    if mapping is None:
        return

    print("Extracting URLs from JSON files...")
    urls = extract_urls_from_json()

    print("Updating mapping with URLs...")
    mapping = update_mapping_with_urls(mapping, urls)

    print("Saving updated mapping...")
    save_updated_mapping(mapping)

    print("✅ Successfully added URLs to indexed data.")

if __name__ == "__main__":
    main()
