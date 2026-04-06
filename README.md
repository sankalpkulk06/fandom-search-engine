# [CS 242] Fandom Wiki Search Engine
**Fandom Wiki Search Engine** is our final project for **CS242: Information Retrieval and Web Search**, designed to enhance search capabilities across various Fandom wikis. This specialized search engine indexes and retrieves relevant content from multiple fan-driven knowledge bases, allowing users to efficiently find information on characters, lore, game mechanics, and more. By implementing advanced search techniques such as ranking algorithms, keyword-based retrieval, and possibly NLP-driven enhancements, our goal is to provide accurate and fast search results tailored to the needs of fandom communities.

## Tech Stack
- **Scraping:** BeautifulSoup, Scrapy
- **Indexing:** Apache Lucene, FAISS
- **Search:** Keyword-based (Lucene), Semantic (BERT + FAISS)
- **GUI:** Streamlit
- **Language:** Python

# Instructions to run the code

# To run the Lucene Indexer after all data is scraped
 Code Instruction indexbuilder.sh
 go to indexer directory
 
 `chmod +x ./indexbuilder.sh`
 
 "Usage: <max-workers for multithreading> <search_query>"
 
 `sh ./indexbuilder.sh --max_workers 8 --search_query "Shang Chi"`
 
 After the builder is done running an index "marvel_index" is created, which we can now use to query

# To try multiple search queries after Lucene Index is built
 Code Instruction search.sh 
 
 `chmod +x ./search.sh`
 
 "Usage: <search_query>"

 `sh ./search.sh "Natasha Romanaff"`

# To run the Bert Indexer after all data is scraped
Code Instruction `indexbuilder.sh`
go to indexer directory
Once you have scraped the necessary data, you need to build the BERT index. Run the following script:

`chmod +x ./run_indexer.sh`

Make sure to add path to directory containing all the scraped data (JSON files)

`sh run_bert_indexing.sh <PATH TO DIR>`

This script will generate the FAISS index (bert_fandom_index.faiss) and the mapping file (doc_id_mapping.json) in the directory required for BERT-based search.
After the builder is done running an index "bert_fandom_index.faiss" and mapping file "doc_id_mapping.json" is created, which we can now use to query.

# To try multiple search queries after Bert Index is built

Once the index is built, you can test multiple search queries using BERT by running: 

`chmod +x ./run_query_bert.sh`

Make sure to add path to directory containing all the scraped data (JSON files)

`sh ./run_query_bert.sh <PATH TO DIR>`

This will allow you to perform searches using the pre-built FAISS index.

# GUI 
- Easy to use and see results
- Built using Streamlit can be hosted locally or on a server

## To run the GUI 

To launch the Streamlit-based web interface for searching superheroes:
- If you are connected to the campus WiFi
    - Run the following commands:
      
      `chmod +x ./run_app.sh`
  
      `sh ./run_app.sh`
  
- If you are off campus and are not connected to the UCR campus WiFi,
    - First connect to the campus VPN using GlobalProtect.
    - Then run the following commands:
      
      `chmod +x ./run_app.sh`
      
      `sh ./run_app.sh`

- This will start the application on port 8888. Open your browser and go to:
  
  `http://localhost:8888`
  
- Select the indexing method to be used for the search.
- Input a query
- Press the Search button.
- The search results will be displayed on the screen, along with retrieval time and ranking.
