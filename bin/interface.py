import streamlit as st
import requests


BASE_URL = "http://localhost:8000"

def fetch_data(endpoint, method='get', data=None):
    """Generic function to make API calls."""
    try:
        if method == 'get':
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method == 'post':
            response = requests.post(f"{BASE_URL}{endpoint}", json=data)
        
        response.raise_for_status()
        try:
            return response.json()
        except ValueError:
            st.error("Invalid JSON response from server")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error: {e}")
        return None


def display_anime_list(anime_list):
    """Display anime list in a formatted manner."""
    for anime in anime_list:
        with st.expander(f"{anime.get('name', 'Unknown')} ({anime.get('type', 'N/A')})"):
            st.write(f"**Genre:** {anime.get('genre', 'N/A')}")
            st.write(f"**Rating:** {anime.get('rating', 'N/A')}")
            st.write(f"**Episodes:** {anime.get('episodes', 'N/A')}")

def main():
    st.set_page_config(page_title="Anime Recommender", page_icon="🎬")
    st.title("Anime Recommendation System")

    page = st.sidebar.selectbox("Choose a Feature", [
        "Genres & Types", 
        "Starter Anime", 
        "Initial Recommendations", 
        # "User Recommendations",
        # "Add User Ratings"
    ])

    if page == "Genres & Types":
        st.header("Available Genres and Types")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Genres")
            genres = fetch_data("/genres")
            if genres:
                st.write(", ".join(genres[:20]) + "...")
        
        with col2:
            st.subheader("Anime Types")
            types = fetch_data("/types")
            if types:
                st.write(", ".join(types))

    elif page == "Starter Anime":
        st.header("Starter Anime Recommendations")
        starter_anime = fetch_data("/starter-anime")
        
        if starter_anime:
            display_anime_list(starter_anime)

    elif page == "Initial Recommendations":
        st.header("Initial Recommendations")

        genres = fetch_data("/genres") or []
        types = fetch_data("/types") or []

        favorite_genres = st.multiselect("Favorite Genres", genres)
        preferred_types = st.multiselect("Preferred Types", types)
        max_episodes = st.slider("Maximum Episodes", 1, 100, 26)
        min_rating = st.slider("Minimum Rating", 0.0, 10.0, 7.5, 0.5)
        
        if st.button("Get Recommendations"):
            preferences = {
                "favorite_genres": favorite_genres,
                "preferred_types": preferred_types,
                "max_episodes": max_episodes,
                "min_rating": min_rating
            }
            
            recommendations = fetch_data("/initial-recommendations", method='post', data=preferences)
            
            if recommendations:
                display_anime_list(recommendations)


if __name__ == "__main__":
    main()
