import streamlit as st
import os
from bert.bert_index_and_search import BERTIndexer

def main():
    st.set_page_config(page_title="Superhero Search", page_icon="🦸", layout="centered")
    
    # Custom styling
    st.markdown(
        """
        <style>
            .stTextInput>div>div>input {
                text-align: center;
            }
            div.stButton > button:first-child {
                width: 100%;
                display: block;
                margin: auto;
            }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Title
    st.title("🦸 Superhero Search 🦹")
    
    # Subtitle
    st.subheader("🔍 Find details about your favorite Marvel and DC superheroes! 💥")
    
    # Paths to the BERT index and doc mapping files
    index_file = os.path.join("..", "bert", "marvel_dc_bert_index.faiss")
    mapping_file = os.path.join("..", "bert", "marvel_dc_doc_mapping.json")
    data_dir = os.path.join("..", "scraper", "data")
    
    # BERTIndexer object
    indexer = BERTIndexer(index_file=index_file, mapping_file=mapping_file, data_dir=data_dir)
    
    # Centered Search Bar
    search_query = st.text_input("", "", placeholder="Search any of your favourite Marvel or DC characters...")

    # Search Button
    search_button = st.button("Search")
    
    if search_button and search_query:
        st.write(f"Searching for: **{search_query}**...")

        # Perform query on the index
        results = indexer.query_index(search_query, top_k=5)

        st.write("### Search Results:")
        for rank, result in enumerate(results, 1):
            st.markdown(f"**[{result['doc_id']}]({result['url']})**")
            st.write(f"**Distance**: {result['distance']:.4f}, **Quality Score**: {result['quality_score']:.2f}")
            st.write(f"**Snippet**: ... {result['snippet']} ...")
            st.write("---")

if __name__ == "__main__":
    main()
