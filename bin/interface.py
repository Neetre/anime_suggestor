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
        "User Recommendations",
        "Add User Ratings"
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

    elif page == "User Recommendations":
        st.header("Personalized Recommendations")
        
        user_id = st.number_input("Enter User ID", min_value=1, step=1)
        recommendation_method = st.selectbox("Recommendation Method", ["SVD", "User-Based"])
        
        if st.button("Get Recommendations"):
            method = "svd" if recommendation_method == "SVD" else "user_based"
            recommendations = fetch_data(f"/recommendations/{user_id}?method={method}")
            
            if recommendations:
                display_anime_list(recommendations)

    elif page == "Add User Ratings":
        st.header("Add User Ratings")
        
        user_id = st.number_input("User ID", min_value=1, step=1)

        num_ratings = st.number_input("Number of Anime to Rate", min_value=1, max_value=10, step=1)
        
        ratings = []
        for i in range(num_ratings):
            col1, col2 = st.columns(2)
            with col1:
                anime_id = st.number_input(f"Anime ID {i+1}", min_value=1, step=1, key=f"anime_id_{i}")
            with col2:
                rating = st.slider(f"Rating {i+1}", 1, 10, 5, key=f"rating_{i}")
            
            ratings.append({"anime_id": anime_id, "rating": rating})
        
        if st.button("Submit Ratings"):
            data = {"user_id": user_id, "ratings": ratings}
            result = fetch_data("/add-user-ratings", method='post', data=data)
            
            if result:
                st.success("User ratings added successfully!")

if __name__ == "__main__":
    main()
