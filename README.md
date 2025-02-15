# [CS 242] Fandom Wiki Search Engine
**Fandom Wiki Search Engine** is our final project for **CS242: Information Retrieval and Web Search**, designed to enhance search capabilities across various Fandom wikis. This specialized search engine indexes and retrieves relevant content from multiple fan-driven knowledge bases, allowing users to efficiently find information on characters, lore, game mechanics, and more. By implementing advanced search techniques such as ranking algorithms, keyword-based retrieval, and possibly NLP-driven enhancements, our goal is to provide accurate and fast search results tailored to the needs of fandom communities.
# Instructions to run the code

# First run the crawler
Code Instruction for crawler.sh
go to scraper directory

`chmod +x crawler.sh`

"Usage: <seed-file> <max-depth> <time-limit>"

`./crawler.sh seed.txt 2 1800`

After successfully running the crawler, the marvel.json file is created in the data directory

# To run the Indexer after all data is scraped
Code Instruction indexbuilder.sh
go to indexer directory

`chmod +x ./indexbuilder.sh`

"Usage: <max-workers for multithreading> <search_query>"

`sh ./indexbuilder.sh --max_workers 8 --search_query "Shang Chi"`

After the builder is done running an index "marvel_index" is created, which we can now use to query

# To try multiple search queries after Index is built
Code Instruction search.sh 

`chmod +x ./search.sh`

"Usage: <search_query>"

`sh ./search.sh "Natasha Romonaff"`
