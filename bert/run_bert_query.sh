#!/bin/bash

# Ensure script exits on failure
set -e

# Check if the required argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./run_query_bert.sh <data_directory>"
    exit 1
fi

DATA_DIR=$1
INDEX_FILE="bert_fandom_index.faiss"
MAPPING_FILE="doc_id_mapping.json"

echo "Starting FAISS search on indexed data in $DATA_DIR..."

# Run the search script
python3 query_bert.py --index_file "$INDEX_FILE" --mapping_file "$MAPPING_FILE" --data_dir "$DATA_DIR"

echo "Search process exited."
