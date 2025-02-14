#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <seed-file> <max-depth> <time-limit> <output-file.json>"
    exit 1
fi

SEED_FILE=$1
MAX_DEPTH=$2
TIME_LIMIT=$3
OUTPUT_FILE=$4

# Ensure the seed file exists
if [ ! -f "$SEED_FILE" ]; then
    echo "Error: Seed file '$SEED_FILE' not found!"
    exit 1
fi

# Read seed URLs from file (space-separated for Python script)
SEED_URLS=$(tr '\n' ' ' < "$SEED_FILE")

# Run the crawler
# ./crawler.sh seed.txt 2 1800 marvel_output.json

python3 crawler.py "$SEED_URLS" "$MAX_DEPTH" "$TIME_LIMIT" "$OUTPUT_FILE"
