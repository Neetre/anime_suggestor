import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix, vstack
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import warnings
from collections import defaultdict
warnings.filterwarnings('ignore')
from time import time


class AnimeRecommender:
    def __init__(self, anime_path, rating_path):
        """Initialize the recommendation engine with data files."""
        self.anime_df = pd.read_csv(anime_path)
        self.rating_df = pd.read_csv(rating_path)
        self.user_anime_matrix = None
        self.user_means = None
        self.U = None
        self.sigma = None
        self.Vt = None
        self.genre_list = None

    def preprocess_data(self):
        """Clean and preprocess the data."""
        self.anime_df['rating'] = pd.to_numeric(self.anime_df['rating'], errors='coerce')
        self.anime_df['episodes'] = pd.to_numeric(self.anime_df['episodes'], errors='coerce')

        self.anime_df['rating'].fillna(self.anime_df['rating'].mean(), inplace=True)
        self.anime_df['episodes'].fillna(self.anime_df['episodes'].median(), inplace=True)

        scaler = MinMaxScaler()
        self.anime_df['normalized_rating'] = scaler.fit_transform(
            self.anime_df[['rating']])

        self.user_indices = {uid: idx for idx, uid in 
                           enumerate(self.rating_df['user_id'].unique())}
        self.anime_indices = {aid: idx for idx, aid in 
                            enumerate(self.rating_df['anime_id'].unique())}

        row_ind = [self.user_indices[uid] for uid in self.rating_df['user_id']]
        col_ind = [self.anime_indices[aid] for aid in self.rating_df['anime_id']]
        self.user_anime_matrix = csr_matrix(
            (self.rating_df['rating'], (row_ind, col_ind)),
            shape=(len(self.user_indices), len(self.anime_indices))
        )

        self.user_means = np.array(self.user_anime_matrix.mean(axis=1)).flatten()

        all_genres = set()
        for genres in self.anime_df['genre'].dropna():
            all_genres.update(g.strip() for g in genres.split(','))
        self.genre_list = sorted(list(all_genres))
        
        return self
    
    def get_initial_recommendations(self, user_preferences):
        """
        Generate recommendations for new users based on their preferences.
        
        Parameters:
        user_preferences: dict
            - favorite_genres: list of genres the user likes
            - preferred_types: list of preferred anime types (TV, Movie, etc.)
            - max_episodes: maximum number of episodes (optional)
            - min_rating: minimum rating threshold (optional)
        """
        filtered_anime = self.anime_df.copy()

        if user_preferences.get('favorite_genres'):
            genre_mask = filtered_anime['genre'].apply(
                lambda x: any(genre in str(x) 
                            for genre in user_preferences['favorite_genres'])
            )
            filtered_anime = filtered_anime[genre_mask]

        if user_preferences.get('preferred_types'):
            filtered_anime = filtered_anime[
                filtered_anime['type'].isin(user_preferences['preferred_types'])
            ]

        if user_preferences.get('max_episodes'):
            filtered_anime = filtered_anime[
                filtered_anime['episodes'] <= user_preferences['max_episodes']
            ]

        if user_preferences.get('min_rating'):
            filtered_anime = filtered_anime[
                filtered_anime['rating'] >= user_preferences['min_rating']
            ]

        filtered_anime['score'] = (
            0.7 * filtered_anime['normalized_rating'] + 
            0.3 * (filtered_anime['members'] / filtered_anime['members'].max())
        )

        top_anime = filtered_anime.nlargest(10, 'score')
        
        return [
            {
                'name': row['name'],
                'genre': row['genre'],
                'rating': row['rating'],
                'type': row['type'],
                'episodes': row['episodes'],
                'members': row['members']
            }
            for _, row in top_anime.iterrows()
        ]
    
    def get_genre_list(self):
        """Return the list of all available genres."""
        return self.genre_list
    
    def get_type_list(self):
        """Return the list of all available anime types."""
        return [x for x in self.anime_df['type'].unique().tolist() if pd.notna(x)]
    
    def train_collaborative_filtering(self, n_factors=50):
        """Train SVD model with specified number of factors."""
        normalized_matrix = self.user_anime_matrix.copy().astype(np.float64)
        for i in range(normalized_matrix.shape[0]):
            normalized_matrix.data[normalized_matrix.indptr[i]:normalized_matrix.indptr[i+1]] -= self.user_means[i]

        self.U, self.sigma, self.Vt = svds(normalized_matrix, k=min(n_factors, 
                                          min(normalized_matrix.shape)-1))
        
        return self
    
    def _compute_user_similarity(self, user_id, n_similar=10):
        """Compute similarity for a single user efficiently."""
        if user_id not in self.user_indices:
            return None, None
        
        user_idx = self.user_indices[user_id]
        user_ratings = self.user_anime_matrix[user_idx].toarray().flatten()

        rated_items = user_ratings.nonzero()[0]
        if len(rated_items) == 0:
            return None, None

        similar_users = defaultdict(float)
        for item in rated_items:
            item_users = self.user_anime_matrix.getcol(item).nonzero()[0]
            for other_user in item_users:
                if other_user != user_idx:
                    similar_users[other_user] += 1

        similar_users = sorted(similar_users.items(), 
                             key=lambda x: x[1], 
                             reverse=True)[:n_similar]
        
        return user_idx, similar_users
    
    def get_user_based_recommendations(self, user_id, n_recommendations=5):
        """Generate user-based collaborative filtering recommendations."""
        user_idx, similar_users = self._compute_user_similarity(user_id)
        
        if user_idx is None or not similar_users:
            return self._get_popular_recommendations(n_recommendations)

        recommendations = np.zeros(self.user_anime_matrix.shape[1])
        for similar_user, similarity in similar_users:
            similar_user_ratings = self.user_anime_matrix[similar_user].toarray().flatten()
            recommendations += similarity * similar_user_ratings

        user_ratings = self.user_anime_matrix[user_idx].toarray().flatten()
        recommendations[user_ratings.nonzero()[0]] = -1
        
        return self._get_top_anime_details(recommendations, n_recommendations)
    
    def get_svd_recommendations(self, user_id, n_recommendations=5):
        """Generate recommendations using SVD matrix factorization."""
        if user_id not in self.user_indices:
            return self._get_popular_recommendations(n_recommendations)
        
        user_idx = self.user_indices[user_id]
        predicted_ratings = self.user_means[user_idx] + np.dot(
            np.dot(self.U[user_idx, :], np.diag(self.sigma)), 
            self.Vt
        )

        user_ratings = self.user_anime_matrix[user_idx].toarray().flatten()
        predicted_ratings[user_ratings.nonzero()[0]] = -1
        
        return self._get_top_anime_details(predicted_ratings, n_recommendations)
    
    def _get_popular_recommendations(self, n_recommendations):
        """Get recommendations for new users based on popularity."""
        popular_anime = self.anime_df.nlargest(n_recommendations, 'members')
        return [
            {
                'name': row['name'],
                'genre': row['genre'],
                'rating': row['rating'],
                'type': row['type'],
                'episodes': row['episodes']
            }
            for _, row in popular_anime.iterrows()
        ]
    
    def _get_top_anime_details(self, recommendations, n_recommendations):
        """Get detailed information for top recommended anime."""
        top_indices = np.argpartition(recommendations, -n_recommendations)[-n_recommendations:]
        top_indices = top_indices[np.argsort(recommendations[top_indices])][::-1]
        
        top_anime = []
        reverse_anime_indices = {v: k for k, v in self.anime_indices.items()}
        
        for idx in top_indices:
            anime_id = reverse_anime_indices[idx]
            anime_info = self.anime_df[self.anime_df['anime_id'] == anime_id].iloc[0]
            top_anime.append({
                'name': anime_info['name'],
                'genre': anime_info['genre'],
                'rating': anime_info['rating'],
                'type': anime_info['type'],
                'episodes': anime_info['episodes']
            })
        
        return top_anime

    def add_new_user_ratings(self, user_id, ratings):
        """
        Add ratings from a new user to the system.
        
        Parameters:
        user_id: int or str
            Unique identifier for the new user
        ratings: list of dict
            List of dictionaries containing anime_id and rating
            Example: [{'anime_id': 1, 'rating': 5}, ...]
        """
        if user_id in self.user_indices:
            raise ValueError("User ID already exists")

        new_user_idx = len(self.user_indices)
        self.user_indices[user_id] = new_user_idx

        new_row = lil_matrix((1, self.user_anime_matrix.shape[1]))

        for rating_info in ratings:
            if rating_info['anime_id'] in self.anime_indices:
                anime_idx = self.anime_indices[rating_info['anime_id']]
                new_row[0, anime_idx] = rating_info['rating']

        self.user_anime_matrix = vstack([self.user_anime_matrix, new_row.tocsr()])

        self.user_means = np.append(
            self.user_means, 
            new_row.mean() or self.user_means.mean()
        )

        self.train_collaborative_filtering()
        
        return True
    
    def get_starter_anime_list(self, n_recommendations=10):
        """
        Get a list of popular starter anime for new users to rate.
        Returns diverse, highly-rated, and popular anime.
        """
        starter_anime = self.anime_df.copy()
        starter_anime['completion_rate'] = starter_anime['members'] / starter_anime['episodes']
        starter_anime['score'] = (
            0.4 * starter_anime['normalized_rating'] +
            0.4 * (starter_anime['members'] / starter_anime['members'].max()) +
            0.2 * (starter_anime['completion_rate'] / starter_anime['completion_rate'].max())
        )

        selected_anime = []
        seen_genres = set()
        
        for _, anime in starter_anime.nlargest(50, 'score').iterrows():
            genres = set(g.strip() for g in str(anime['genre']).split(','))
            if not genres & seen_genres:
                selected_anime.append({
                    'anime_id': anime['anime_id'],
                    'name': anime['name'],
                    'genre': anime['genre'],
                    'type': anime['type'],
                    'episodes': anime['episodes'],
                    'rating': anime['rating']
                })
                seen_genres.update(genres)
            
            if len(selected_anime) >= n_recommendations:
                break
        
        return selected_anime


recommender = AnimeRecommender('../data/anime.csv', '../data/rating.csv')


def new_user():
    recommender.preprocess_data()

    print("Available genres:", recommender.get_genre_list())
    print("Available types:", recommender.get_type_list())

    starter_anime = recommender.get_starter_anime_list()
    print("\nStarter anime to rate:")
    for anime in starter_anime:
        print(f"- {anime['name']} ({anime['genre']})")

    user_preferences = {
        'favorite_genres': ['Action', 'Adventure'],
        'preferred_types': ['TV', 'Movie'],
        'max_episodes': 26,
        'min_rating': 7.5
    }
    
    initial_recommendations = recommender.get_initial_recommendations(user_preferences)
    print("\nInitial recommendations based on preferences:")
    for anime in initial_recommendations:
        print(f"- {anime['name']} ({anime['genre']})")

    start = time()
    new_user_id = 73517
    new_user_ratings = [
        {'anime_id': 1, 'rating': 5},
        {'anime_id': 2, 'rating': 4}
    ]
    recommender.add_new_user_ratings(new_user_id, new_user_ratings)

    recommendations = recommender.get_svd_recommendations(new_user_id)
    print(f"Prediction time (User-based CF): {time() - start:.2f}s")
    print("\nPersonalized recommendations after ratings:")
    for anime in recommendations:
        print(f"- {anime['name']} ({anime['genre']})")


def main():
    '''
    start = time()
    recommender = AnimeRecommender('../data/anime.csv', '../data/rating.csv')
    recommender.preprocess_data()
    recommender.train_collaborative_filtering(n_factors=30)
    print(f"Training time: {time() - start:.2f}s")

    user_id = 101
    print(f"\nRecommendations for User {user_id}:")
    
    print("\nUser-based Collaborative Filtering Recommendations:")
    start = time()
    recommendations = recommender.get_user_based_recommendations(user_id)
    for idx, anime in enumerate(recommendations, 1):
        print(f"{idx}. Name: {anime['name']}")
        print(f"   Genre: {anime['genre']}")
        print(f"   Rating: {anime['rating']:.2f}")
        print(f"   Type: {anime['type']}")
        print(f"   Episodes: {anime['episodes']}")
    print(f"Prediction time (User-based CF): {time() - start:.2f}s")
    
    print("\nSVD-based Recommendations:")
    start = time()
    recommendations = recommender.get_svd_recommendations(user_id)
    for idx, anime in enumerate(recommendations, 1):
        print(f"{idx}. Name: {anime['name']}")
        print(f"   Genre: {anime['genre']}")
        print(f"   Rating: {anime['rating']:.2f}")
        print(f"   Type: {anime['type']}")
        print(f"   Episodes: {anime['episodes']}")
    print(f"Prediction time (SVD): {time() - start:.2f}s")
    '''
    new_user()


if __name__ == "__main__":
    main()
