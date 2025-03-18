#!/bin/bash

# Ensure script exits on failure
set -e

# Check if input directory argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_bert_indexing.sh <input_directory>"
    exit 1
fi

INPUT_DIR=$1
INDEX_FILE="bert_fandom_index.faiss"
MAPPING_FILE="doc_id_mapping.json"

echo "Creating FAISS index from data in $INPUT_DIR..."

# Run the indexing script
python3 bert_indexer.py --input_dir "$INPUT_DIR" --index_file "$INDEX_FILE" --mapping_file "$MAPPING_FILE"

echo "Indexing complete. Files saved:"
echo "- FAISS index: $INDEX_FILE"
echo "- Document mapping: $MAPPING_FILE"
