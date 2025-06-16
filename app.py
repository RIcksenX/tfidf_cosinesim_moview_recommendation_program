from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

try:
    movies_df = joblib.load('movies_df.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
except Exception as e:
    logger.error(f"Error loading artifacts: {e}")
    raise

logger.info(f"movies_df shape: {movies_df.shape}")
logger.info(f"movieId type: {movies_df['movieId'].dtype}")
logger.info(f"Rows with movieId=862: \n{movies_df[movies_df['movieId'] == 862]}")
logger.info(f"tfidf_matrix shape: {tfidf_matrix.shape}")

def get_content_recommendations(movie_id, tfidf_matrix, movies_df, top_k=10):
    movie_id = int(movie_id) if isinstance(movie_id, (int, str)) else movie_id
    
    logger.info(f"Processing recommendations for movie_id: {movie_id}")
    idx = movies_df.index[movies_df['movieId'] == movie_id].tolist()
    idx = idx[0]
    
    cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix)[0]

    sim_scores = list(enumerate(cosine_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k+1]
    
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    result = movies_df[['movieId', 'title']].iloc[movie_indices].copy()
    result['similarity_score'] = similarity_scores
    logger.info(f"Found {len(result)} recommendations for movie_id={movie_id}")
    return result

@app.route('/api/movies')
def get_movies():
    try:
        query = request.args.get('q', '').lower()
        logger.info(f"Fetching movies with query: {query}")
        # Filter movies by query (case-insensitive)
        if query:
            filtered_movies = movies_df[movies_df['title'].str.lower().str.contains(query, na=False)]
        else:
            filtered_movies = movies_df
        movie_list = filtered_movies[['movieId', 'title']].sort_values('title').to_dict('records')
        response = [{'id': str(movie['movieId']), 'text': movie['title']} for movie in movie_list]
        logger.info(f"Returning {len(response)} movies for query: {query}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in /api/movies: {e}")
        return jsonify({'error': 'Failed to load movies'}), 500

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    selected_movie = None
    error_message = None
    
    if request.method == 'POST':
        movie_id = request.form.get('movie_id')
        logger.info(f"Received POST request with movie_id: {movie_id}")
        if movie_id:
            recommendations = get_content_recommendations(movie_id, tfidf_matrix, movies_df, top_k=10)
            if not recommendations.empty:
                selected_movie = movies_df[movies_df['movieId'] == int(movie_id)]['title'].iloc[0]
            else:
                error_message = f"No recommendations found for movie ID {movie_id}."
                logger.warning(error_message)
    
    return render_template('index.html', recommendations=recommendations, selected_movie=selected_movie, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)