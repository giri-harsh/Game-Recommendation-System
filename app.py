from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

try:
    with open("model.pkl", "rb") as file:
        knn_model, X_normalized, feature_columns, df = pickle.load(file)
except Exception as e:
    raise RuntimeError(f"Error loading model.pkl: {e}")

app = FastAPI(title="Game Recommendation API", version="1.0")

class FeatureRequest(BaseModel):
    """Request body for recommending by features."""
    platforms: Optional[List[str]] = None
    genres: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    n: int = 6


class RecommendationItem(BaseModel):
    """Single recommended game item."""
    rank: int
    game: str
    similarity: float


class RecommendationResponse(BaseModel):
    """Response model for feature-based recommendations."""
    recommendations: List[RecommendationItem]


class NameRecommendationResponse(BaseModel):
    """Response model for name-based recommendations."""
    searched_game: str
    recommendations: List[RecommendationItem]

@app.get("/")
def home():
    return {"message": "Welcome to the Game Recommendation API! Use /docs to explore endpoints."}

@app.get("/recommend/name/", response_model=NameRecommendationResponse)
def recommend_by_name(game_name: str, n: int = 6):
    """Recommend similar games by game name."""

    if "name" not in df.columns:
        raise HTTPException(status_code=500, detail="Dataset missing 'name' column.")
    
    matches = df[df['name'].str.contains(game_name, case=False, na=False)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"Game '{game_name}' not found in dataset.")

    game_idx = matches.index[0]
    game_name_exact = matches.iloc[0]['name']

    game_features = X_normalized[game_idx].reshape(1, -1)
    distances, indices = knn_model.kneighbors(game_features, n_neighbors=n)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0][1:], indices[0][1:]), start=1):
        similarity = round((1 - dist) * 100, 2)
        results.append({
            "rank": i,
            "game": df.iloc[idx]['name'],
            "similarity": similarity
        })

    return {"searched_game": game_name_exact, "recommendations": results}

@app.post("/recommend/features/", response_model=RecommendationResponse)
def recommend_by_features(request: FeatureRequest):
    """Recommend games based on selected features."""
    custom_features = np.zeros(len(feature_columns))

    
    if request.platforms:
        for platform in request.platforms:
            col = f"platform_{platform.replace(' ', '_').replace('/', '_').replace('-', '_')}"
            if col in feature_columns:
                custom_features[feature_columns.index(col)] = 1

    if request.genres:
        for genre in request.genres:
            col = f"genre_{genre.replace(' ', '_').replace('-', '_')}"
            if col in feature_columns:
                custom_features[feature_columns.index(col)] = 1

    if request.tags:
        for tag in request.tags:
            clean_tag = tag.replace(" ", "_").replace("-", "_").replace("/", "_").replace("&", "and")
            col = f"tag_{clean_tag}"
            if col in feature_columns:
                custom_features[feature_columns.index(col)] = 1

    custom_features = normalize(custom_features.reshape(1, -1), norm='l2')
    distances, indices = knn_model.kneighbors(custom_features, n_neighbors=request.n)

    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0]), start=1):
        similarity = round((1 - dist) * 100, 2)
        results.append({
            "rank": i,
            "game": df.iloc[idx]['name'],
            "similarity": similarity
        })

    return {"recommendations": results}


