import streamlit as st

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
    
    # Display Image
    # st.image("https://upload.wikimedia.org/wikipedia/en/3/35/Superhero_collage.png", use_column_width=True)
    
    # Centered Search Bar
    search_query = st.text_input("", "", placeholder="Search any of your favourite Marvel or DC characters...")
    
    # Search Button
    search_button = st.button("Search")
    
    # Sample search results class
    class SearchResult:
        def __init__(self, name, url, description):
            self.name = name
            self.url = url
            self.description = description
    
    sample_results = [
        SearchResult("Spider-Man", "https://www.marvel.com/characters/spider-man-peter-parker", "A young hero with spider-like abilities and a strong sense of responsibility."),
        SearchResult("Batman", "https://www.dc.com/characters/batman", "A billionaire vigilante who fights crime using his intellect, gadgets, and martial arts."),
        SearchResult("Iron Man", "https://www.marvel.com/characters/iron-man-tony-stark", "Genius billionaire Tony Stark fights evil in his high-tech armored suit."),
        SearchResult("Wonder Woman", "https://www.dc.com/characters/wonder-woman", "An Amazonian warrior princess with superhuman strength and combat skills."),
    ]
    
    # Display search results if search is performed
    if search_button and search_query:
        st.write(f"Searching for: **{search_query}**...")
        st.write("### Sample Search Results:")
        for result in sample_results:
            st.markdown(f"**[{result.name}]({result.url})**")
            st.write(result.description)
            st.write("---")

if __name__ == "__main__":
    main()
