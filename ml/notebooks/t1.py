# %% Cosine-Similarity Based Game Recommendation System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Data ---
df = pd.read_csv(r"C:\Work\Programing Language\task4\Game-Recommendation-System\data\raw\video_game_reviews.csv")

# --- Basic Cleaning ---
df.dropna(subset=['Game Title'], inplace=True)

# --- Encode quality columns ---
graphics_map = {'Low': 1, 'Medium': 2, 'High': 3, 'Ultra': 4}
soundtrack_map = {'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}
story_map = {'Poor': 1, 'Average': 2, 'Good': 3, 'Excellent': 4}

df['Graphics_Score'] = df['Graphics Quality'].map(graphics_map)
df['Soundtrack_Score'] = df['Soundtrack Quality'].map(soundtrack_map)
df['Story_Score'] = df['Story Quality'].map(story_map)
df['Overall_Quality_Score'] = df[['Graphics_Score', 'Soundtrack_Score', 'Story_Score']].mean(axis=1).round(2)

df.drop(['Graphics Quality', 'Soundtrack Quality', 'Story Quality',
         'Min Number of Players', 'User Review Text',
         'Release Year', 'Publisher', 'Developer', 'Price'], axis=1, inplace=True)

# --- Binary Encoding ---
bin_map = {'Yes': 1, 'No': 0}
gm_map = {'Offline': 0, 'Online': 1}

if 'Requires Special Device' in df.columns:
    df['Requires Special Device'] = df['Requires Special Device'].map(bin_map)
if 'Multiplayer' in df.columns:
    df['Multiplayer'] = df['Multiplayer'].map(bin_map)
if 'Game Mode' in df.columns:
    df['Game Mode'] = df['Game Mode'].map(gm_map)

# --- One-Hot Encoding ---
ohe_cols = ['Age Group Targeted', 'Platform', 'Genre']
df_encoded = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# --- Feature Scaling ---
feature_columns = df_encoded.columns.drop(['Game Title', 'Graphics_Score', 'Soundtrack_Score', 'Story_Score'])
df_encoded[feature_columns] = df_encoded[feature_columns].astype(float)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_encoded[feature_columns])

# --- Cosine Similarity Matrix ---
similarity_matrix = cosine_similarity(X_scaled)

# --- Recommendation Function ---
def get_recommendations(game_name, num_recommendations=5):
    """Find similar games using cosine similarity."""
    idx_list = df_encoded.index[df_encoded['Game Title'] == game_name].tolist()
    if not idx_list:
        return f"Game '{game_name}' not found"
    idx = idx_list[0]

    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

    results = []
    for rank, (game_idx, score) in enumerate(sim_scores, start=1):
        results.append({
            'rank': rank,
            'game': df_encoded.iloc[game_idx]['Game Title'],
            'similarity': round(score * 100, 2),
            'rating': df_encoded.iloc[game_idx]['User Rating']
        })
    return results

# --- Test Random Games ---
import random
test_games = random.sample(list(df_encoded['Game Title']), 3)

for test_game in test_games:
    print(f"\n🎮 Games similar to: '{test_game}'")
    print("-" * 60)
    recs = get_recommendations(test_game)
    for rec in recs:
        print(f"{rec['rank']}. {rec['game']}")
        print(f"   Similarity: {rec['similarity']}%")
        print(f"   Rating: {rec['rating']}/50")
