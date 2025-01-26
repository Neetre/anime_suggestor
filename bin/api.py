# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from suggestion import AnimeRecommender
import uvicorn

recommender = AnimeRecommender('../data/anime.csv', '../data/rating.csv')
recommender.preprocess_data().train_collaborative_filtering()

app = FastAPI(title="Anime Recommender API")

class UserPreferences(BaseModel):
    favorite_genres: Optional[List[str]] = None
    preferred_types: Optional[List[str]] = None
    max_episodes: Optional[int] = None
    min_rating: Optional[float] = None

class UserRating(BaseModel):
    anime_id: int
    rating: float

class UserRatings(BaseModel):
    user_id: int
    ratings: List[UserRating]

@app.on_event("startup")
async def startup_event():
    """Preprocessing on API startup."""
    pass

@app.get("/genres")
def get_genres():
    """Get available anime genres."""
    return recommender.get_genre_list()

@app.get("/types")
def get_types():
    """Get available anime types."""
    return recommender.get_type_list()

@app.get("/starter-anime")
def get_starter_anime(n: int = 10):
    """Get recommended starter anime for new users."""
    return recommender.get_starter_anime_list(n_recommendations=n)

@app.post("/initial-recommendations")
def get_initial_recommendations(preferences: UserPreferences):
    """Get initial recommendations based on user preferences."""
    preferences_dict = {
        k: v for k, v in preferences.model_dump().items() if v is not None
    }
    return recommender.get_initial_recommendations(preferences_dict)

@app.post("/add-user-ratings")
def add_user_ratings(user_ratings: UserRatings):
    """Add ratings for a new user."""
    try:
        recommender.add_new_user_ratings(
            user_ratings.user_id, 
            [rating.model_dump() for rating in user_ratings.ratings]
        )
        return {"message": "User ratings added successfully"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/recommendations/{user_id}")
def get_recommendations(user_id: int, method: str = "svd", n: int = 5):
    """Get personalized anime recommendations for a user."""
    try:
        if method == "svd":
            return recommender.get_svd_recommendations(user_id, n_recommendations=n)
        elif method == "user_based":
            return recommender.get_user_based_recommendations(user_id, n_recommendations=n)
        else:
            raise HTTPException(status_code=400, detail="Invalid recommendation method")
    except Exception as e:
        raise HTTPException(status_code=404, detail="User not found")


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
