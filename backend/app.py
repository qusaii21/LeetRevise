# app.py
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import requests
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from flask_cors import CORS
import os
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
LEETCODE_API_BASE_URL = "http://localhost:3000/"  # Local API base URL
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyBFoh57DC91SRffa2qONpHM_7CsDexiUeI")
NUM_RECOMMENDATIONS = 15
TOP_FILTERED = 5

# Initialize dataset - load it once at startup
print("Loading LeetCode dataset...")
dataset = None
leetcode_df = None

try:
    dataset = load_dataset('greengerong/leetcode')
    leetcode_df = dataset['train'].to_pandas()
    print(f"Dataset loaded with {len(leetcode_df)} problems")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")

# Initialize Gemini API
def initialize_gemini():
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel('gemini-2.0-flash')

class OptimizedQuestionRecommender:
    """
    A highly optimized recommender for large question datasets.
    Uses sparse TF-IDF, approximate nearest neighbors for graph construction,
    vectorized potential computation, and sklearn LDA for topic modeling.
    """
    def __init__(self, df=None, top_k=10, n_topics=5):
        # If dataframe is provided, use it; otherwise try to load from file
        if df is not None:
            self.df = df
            self._preprocess_df()
        else:
            try:
                self.df = self._load_and_preprocess("updated_data.json")
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                # Fall back to using the global dataset if available
                if leetcode_df is not None:
                    self.df = leetcode_df.copy()
                    self._preprocess_df()
                else:
                    raise ValueError("No data available for recommender")
        
        self.top_k = top_k
        self._vectorize()
        self._topic_model(n_topics)
        self._compute_potentials()
        self._build_graph()

    def _load_and_preprocess(self, path):
        df = pd.read_json(path)
        scaler = MinMaxScaler()
        df[['likability','accuracy']] = scaler.fit_transform(df[['likability','accuracy']])
        # bin accuracy
        df['accuracy_bin'] = pd.cut(df['accuracy'], bins=4, labels=False, include_lowest=True)
        return df
        
    def _preprocess_df(self):
        """Preprocess dataframe that was loaded from another source"""
        # Create placeholder columns if they don't exist
        if 'likability' not in self.df.columns:
            self.df['likability'] = self.df['difficulty'].map({'Easy': 0.8, 'Medium': 0.5, 'Hard': 0.3})
        if 'accuracy' not in self.df.columns:
            self.df['accuracy'] = 0.75  # default value
            
        scaler = MinMaxScaler()
        self.df[['likability','accuracy']] = scaler.fit_transform(self.df[['likability','accuracy']])
        # bin accuracy
        self.df['accuracy_bin'] = pd.cut(self.df['accuracy'], bins=4, labels=False, include_lowest=True)

    def _vectorize(self):
        # Use content or question field depending on what's available
        text_field = 'question' if 'question' in self.df.columns else 'content'
        # TF-IDF sparse matrix
        self.tfidf = TfidfVectorizer(max_df=0.8, min_df=5).fit_transform(self.df[text_field])

    def _topic_model(self, n_topics):
        # Use content or question field depending on what's available
        text_field = 'question' if 'question' in self.df.columns else 'content'
        # Use CountVectorizer + LDA
        count = CountVectorizer(max_df=0.8, min_df=5).fit_transform(self.df[text_field])
        lda = LatentDirichletAllocation(n_components=n_topics, max_iter=10, learning_method='batch')
        self.topic_matrix = lda.fit_transform(count)

    def _compute_potentials(self):
        # Vectorized joint probability table
        joint = pd.crosstab(self.df['accuracy_bin'], self.df['difficulty'], normalize='all')
        self.potential_matrix = joint.values  # shape (4,3)

    def _build_graph(self):
        # Use NearestNeighbors to find top_k similar questions
        nbrs = NearestNeighbors(n_neighbors=min(self.top_k+1, len(self.df)), metric='cosine', n_jobs=-1)
        nbrs.fit(self.tfidf)
        dists, idxs = nbrs.kneighbors(self.tfidf)

        self.G = nx.Graph()
        # add nodes with attributes
        slug_field = 'titleSlug' if 'titleSlug' in self.df.columns else 'slug'
        
        for i, row in self.df.iterrows():
            self.G.add_node(row[slug_field], difficulty=row['difficulty'],
                            likability=row['likability'], accuracy=row['accuracy'])

        # add edges only among top_k neighbors
        for i, neighbors in enumerate(idxs):
            src = self.df.iloc[i][slug_field]
            for rank, j in enumerate(neighbors[1:], start=1):
                weight = 1 - dists[i, rank]  # higher weight = more similar
                dst = self.df.iloc[j][slug_field]
                self.G.add_edge(src, dst, weight=weight)

    def recommend(self, solved, top_n=15):
        recs = {}
        for slug in solved:
            if slug not in self.G: continue
            for nbr, data in self.G[slug].items():
                if nbr in solved: continue
                recs[nbr] = recs.get(nbr, 0) + data['weight']
        return sorted(recs.items(), key=lambda x: x[1], reverse=True)[:top_n]

def fetch_total_solved_problems(username):
    """
    Fetch the total number of solved problems by the user from the LeetCode API.
    """
    endpoint = f"{username}/solved"
    url = LEETCODE_API_BASE_URL + endpoint
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        user_data = response.json()
        
        # Extract total solved problems count
        total_solved = user_data.get('solvedProblem', 0)
        return total_solved, None

    except requests.exceptions.RequestException as e:
        error_msg = f"Error fetching data for user {username}: {str(e)}"
        return 0, error_msg

def fetch_solved_problems(username, total_solved):
    """
    Fetch all solved problems by the user from the LeetCode API, handling pagination.
    """
    limit = 50  # Set a reasonable batch size
    solved_problems = set()
    
    for offset in range(0, total_solved, limit):
        endpoint = f"{username}/acSubmission?limit={limit}&offset={offset}"
        url = LEETCODE_API_BASE_URL + endpoint
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            user_submissions = response.json()
            
            for submission in user_submissions.get("submission", []):
                if submission.get("statusDisplay") == "Accepted":
                    solved_problems.add(submission.get("titleSlug", ""))
        
        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching data at offset {offset}: {str(e)}"
            return list(solved_problems), error_msg

    return list(solved_problems), None

def analyze_with_gemini(solved_questions, recommendations):
    """Use Gemini API to analyze and filter recommendations"""
    try:
        model = initialize_gemini()
        
        prompt = f"""Analyze these solved problems:
        {solved_questions}

        And these recommendations:
        {recommendations}

        Select top {TOP_FILTERED} that best:
        1. Match patterns from solved problems
        2. Offer progressive difficulty
        3. Cover important DSA concepts
        4. Are high-frequency interview questions

        Format exactly:
        1. Title - Reason 
        2. Title - Reason 
        ..."""

        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        return response.text, None
    except Exception as e:
        error_msg = f"Gemini API Error: {str(e)}"
        return None, error_msg

def recommend_based_on_description(solved_slugs):
    """Generate recommendations based on question descriptions"""
    global leetcode_df
    
    if leetcode_df is None:
        return None, None, "Dataset not available"
    
    df = leetcode_df.copy()
    
    # TF-IDF setup for question content
    tfidf = TfidfVectorizer(stop_words='english')
    content_matrix = tfidf.fit_transform(df['content'])

    # Get solved questions
    solved_mask = df['slug'].isin(solved_slugs)
    solved_questions = df[solved_mask]

    if len(solved_questions) == 0:
        return None, None, "No solved questions found for this user"

    # Get recommendations
    similarity_scores = np.zeros(len(df))
    for idx in solved_questions.index:
        similarity_scores += cosine_similarity(content_matrix[idx], content_matrix).flatten()
    
    if len(solved_questions) > 0:
        similarity_scores /= len(solved_questions)
    
    df['similarity'] = similarity_scores

    recommendations = df[~solved_mask].sort_values('similarity', ascending=False)
    recommendations = recommendations.head(NUM_RECOMMENDATIONS)
    
    return solved_questions, recommendations, None

def recommend_based_on_solutions(solved_slugs):
    """Generate recommendations based on solutions and graph structure"""
    global leetcode_df
    
    if leetcode_df is None:
        return None, None, "Dataset not available"
    
    try:
        recommender = OptimizedQuestionRecommender(df=leetcode_df, top_k=20, n_topics=5)
        recommendation_tuples = recommender.recommend(solved_slugs, top_n=NUM_RECOMMENDATIONS)
        
        # Get recommendation details
        slug_field = 'titleSlug' if 'titleSlug' in leetcode_df.columns else 'slug'
        rec_slugs = [r[0] for r in recommendation_tuples]
        rec_details = leetcode_df[leetcode_df[slug_field].isin(rec_slugs)]
        
        # Add score to recommendation details
        scores_dict = {slug: score for slug, score in recommendation_tuples}
        rec_details['similarity'] = rec_details[slug_field].map(scores_dict)
        
        # Get solved question details
        solved_mask = leetcode_df[slug_field].isin(solved_slugs)
        solved_questions = leetcode_df[solved_mask]
        
        return solved_questions, rec_details, None
        
    except Exception as e:
        error_msg = f"Error in solution-based recommendation: {str(e)}"
        return None, None, error_msg

# API Endpoints

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "dataset_loaded": dataset is not None,
        "num_problems": len(leetcode_df) if leetcode_df is not None else 0
    })

@app.route('/api/fetch-user', methods=['GET'])
def fetch_user():
    """Get user information and solved problems"""
    username = request.args.get('username')
    
    if not username:
        return jsonify({"error": "Username is required"}), 400
    
    # Fetch total solved problems
    total_solved, error = fetch_total_solved_problems(username)
    
    if error:
        return jsonify({"error": error}), 500
        
    if total_solved == 0:
        return jsonify({
            "username": username,
            "total_solved": 0,
            "solved_problems": [],
            "message": "No solved problems found"
        })
    
    # Fetch solved problems
    solved_problems, error = fetch_solved_problems(username, total_solved)
    
    if error:
        return jsonify({"error": error}), 500
    
    return jsonify({
        "username": username,
        "total_solved": total_solved,
        "solved_problems": solved_problems
    })

@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """Generate question recommendations"""
    data = request.json
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    username = data.get('username')
    solved_slugs = data.get('solved_slugs')
    rec_type = data.get('recommendation_type', 'description')  # 'description' or 'solution'
    
    if not username or not solved_slugs:
        return jsonify({"error": "Username and solved problems are required"}), 400
    
    # Generate recommendations based on type
    if rec_type == 'solution':
        solved_questions, recommendations, error = recommend_based_on_solutions(solved_slugs)
    else:  # default to description
        solved_questions, recommendations, error = recommend_based_on_description(solved_slugs)
    
    if error:
        return jsonify({"error": error}), 500
    
    if solved_questions is None or recommendations is None:
        return jsonify({"error": "Failed to generate recommendations"}), 500
    
    # Convert to lists for JSON serialization
    solved_list = solved_questions[['title', 'difficulty', 'content']].to_dict('records')
    rec_list = recommendations[['title', 'difficulty', 'content', 'similarity']].to_dict('records')
    
    # Analyze with Gemini if enabled
    gemini_analysis = None
    gemini_error = None
    
    if data.get('use_gemini', True):
        solved_titles = solved_questions['title'].tolist()
        rec_titles = recommendations['title'].tolist()
        gemini_analysis, gemini_error = analyze_with_gemini(solved_titles, rec_titles)
    
    response = {
        "username": username,
        "recommendation_type": rec_type,
        "total_solved": len(solved_slugs),
        "recommendations": rec_list,
        "top_recommendations": gemini_analysis,
        "error": gemini_error
    }
    
    return jsonify(response)

@app.route('/api/problems', methods=['GET'])
def get_problems():
    """Return a subset of problems for browsing"""
    global leetcode_df
    
    if leetcode_df is None:
        return jsonify({"error": "Dataset not available"}), 500
    
    # Get query parameters
    difficulty = request.args.get('difficulty')
    limit = min(int(request.args.get('limit', 100)), 500)  # Cap at 500 for performance
    
    # Filter by difficulty if specified
    if difficulty and difficulty in ['Easy', 'Medium', 'Hard']:
        filtered_df = leetcode_df[leetcode_df['difficulty'] == difficulty]
    else:
        filtered_df = leetcode_df
    
    # Select a subset
    sample = filtered_df.sample(min(limit, len(filtered_df)))
    
    # Convert to dict for JSON response
    problems = sample[['title', 'difficulty', 'slug']].to_dict('records')
    
    return jsonify({
        "total": len(filtered_df),
        "limit": limit,
        "problems": problems
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

