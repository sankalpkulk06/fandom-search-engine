#!/bin/bash

# Default values
MAX_WORKERS=4
SEARCH_QUERY="Spider-Man"

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --max_workers) MAX_WORKERS="$2"; shift ;;
        --search_query) SEARCH_QUERY="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the parsed arguments
python3 indexbuilder.py --max_workers "$MAX_WORKERS" --search_query "$SEARCH_QUERY" > output.txt
