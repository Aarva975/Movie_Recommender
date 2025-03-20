import streamlit as st
import pandas as pd
import pickle
import requests
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
from streamlit_lottie import st_lottie

# Load Lottie animation
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data
def load_data():
    df = pd.read_csv('tmdb_enriched_dataset.csv')
    df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(2000).astype(int)
    df['recency_boost'] = 2024 - df['release_year']
    with open('tfidf_matrix.pkl', 'rb') as f:
        tfidf_matrix = pickle.load(f)
    return df, tfidf_matrix

movies_df, tfidf_matrix = load_data()

def get_movie_details(movie_id, api_key):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    response = requests.get(url, params={'api_key': api_key, 'language': 'en-US'})
    data = response.json()
    poster_url = f"https://image.tmdb.org/t/p/w500{data['poster_path']}" if data.get('poster_path') else None
    rating = data.get('vote_average')
    title = data.get('title')
    return {'title': title, 'poster_url': poster_url, 'rating': rating}

# ✅ Final smart recommendation function
def smart_recommendation(df, title=None, genre=None, director=None, cast_member=None, keyword=None, top_n=10):
    df = df.copy()
    df['match_bonus'] = 0

    # Normalize quality factors
    df['rating_norm'] = (df['vote_average'] - df['vote_average'].min()) / (df['vote_average'].max() - df['vote_average'].min())
    df['popularity_norm'] = (df['popularity'] - df['popularity'].min()) / (df['popularity'].max() - df['popularity'].min())
    df['revenue_norm'] = (df['revenue'] - df['revenue'].min()) / (df['revenue'].max() - df['revenue'].min())

    # Cast matching
    if cast_member:
        df['cast_match'] = df['top_cast'].apply(lambda x: fuzz.partial_ratio(str(x).lower(), cast_member.lower()))
        df.loc[df['cast_match'] > 60, 'match_bonus'] += (df['cast_match'] / 100) * 30
        if df['cast_match'].max() < 50:
            st.info(f"No strong matches for {cast_member}. Showing fallback results.")
            fallback = df[df['top_cast'].str.contains(cast_member, case=False, na=False)].sort_values(
                by=['vote_average', 'vote_count', 'popularity'], ascending=False
            ).head(top_n)
            return fallback

    # Genre manual boost
    if genre and genre != "Any":
        df.loc[df['genres'].str.contains(genre, case=False, na=False), 'match_bonus'] += 20

    # Keyword weighted by genre compatibility
    if keyword:
        relevant_genres_for_keyword = {
            'christmas': ['Family', 'Holiday', 'Comedy'],
            'heist': ['Crime', 'Thriller'],
            'romance': ['Romance', 'Drama', 'Comedy'],
            'horror': ['Horror', 'Thriller']
            # You can add more mappings as needed
        }

        keyword_lower = keyword.lower()
        keyword_mask = df['keywords'].str.contains(keyword, case=False, na=False)

        if keyword_lower in relevant_genres_for_keyword:
            expected_genres = relevant_genres_for_keyword[keyword_lower]
            for idx, row in df[keyword_mask].iterrows():
                # Check if movie genres align
                genre_match = any(g.strip() in row['genres'] for g in expected_genres)
                if genre_match:
                    df.at[idx, 'match_bonus'] += 15
                else:
                    # Penalize partial relevance
                    df.at[idx, 'match_bonus'] += 7.5
        else:
            # If keyword has no defined mapping, apply default moderate boost
            df.loc[keyword_mask, 'match_bonus'] += 10

    # Director match
    if director:
        df['director_match'] = df['director_name'].apply(lambda x: fuzz.partial_ratio(str(x).lower(), director.lower()))
        df.loc[df['director_match'] > 60, 'match_bonus'] += (df['director_match'] / 100) * 10

    # Title-based attribute matching
    if title:
        matches = df['title'].apply(lambda x: fuzz.ratio(x.lower(), title.lower()))
        idx = matches.idxmax()
        if matches[idx] > 60:
            movie_row = df.iloc[idx]
            title_genres = movie_row['genres'].split(',') if pd.notnull(movie_row['genres']) else []
            title_keywords = movie_row['keywords'].split(',') if pd.notnull(movie_row['keywords']) else []
            title_director = movie_row['director_name']
            title_cast = movie_row['top_cast'].split(',') if pd.notnull(movie_row['top_cast']) else []

            for g in title_genres:
                df.loc[df['genres'].str.contains(g.strip(), case=False, na=False), 'match_bonus'] += 10
            for kw in title_keywords:
                df.loc[df['keywords'].str.contains(kw.strip(), case=False, na=False), 'match_bonus'] += 8
            if title_director:
                df['director_title_match'] = df['director_name'].apply(lambda x: fuzz.partial_ratio(str(x).lower(), title_director.lower()))
                df.loc[df['director_title_match'] > 60, 'match_bonus'] += 10
            for c in title_cast:
                df.loc[df['top_cast'].str.contains(c.strip(), case=False, na=False), 'match_bonus'] += 12

            cosine_similarities = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
            df['semantic_similarity'] = cosine_similarities
        else:
            df['semantic_similarity'] = 0
    else:
        df['semantic_similarity'] = 0

    df['randomness'] = np.random.uniform(0, 1, size=len(df))

    df['base_score'] = (
        df['rating_norm'] * 0.4 +
        df['popularity_norm'] * 0.3 +
        df['revenue_norm'] * 0.2 +
        (df['recency_boost'] / df['recency_boost'].max()) * 0.1
    )

    df['final_score'] = df['base_score'] * (1 + df['match_bonus'] / 20 + df['semantic_similarity'] * 2 + df['randomness'] * 0.5)
    df = df.sort_values(by='final_score', ascending=False)
    return df.head(top_n)


# === Theme styling ===
theme = st.sidebar.radio("🎨 Choose Theme:", ["Dark Mode", "Light Mode"])

if theme == "Dark Mode":
    st.markdown("""
        <style>
        body, .stApp { background-color: #121212 !important; color: white !important; }
        h1, h2, h3 { color: #ffcc70 !important; }
        .movie-card { background-color: rgba(255,255,255,0.08); border-radius: 15px; padding: 15px; margin-bottom: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.4); }
        p, div, span, label, input, .stMarkdown, .stSlider { color: white !important; }
        .stTextInput>div>div>input, .stSelectbox>div>div>select { background-color: #222 !important; color: white !important; border: 1px solid #555 !important; border-radius: 8px; padding: 8px; }
        .stButton>button { background-color: #ff7f50; color: white; border-radius: 50px; padding: 12px 25px; transition: 0.3s; }
        .stButton>button:hover { background-color: #ffa07a; transform: scale(1.05); }
        </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
        <style>
        body, .stApp { background-color: #f9f9f9 !important; color: #333 !important; }
        h1, h2, h3 { color: #ffcc70 !important; }
        .movie-card { background-color: #fff; border-radius: 15px; padding: 15px; margin-bottom: 20px; box-shadow: 0 2px 15px rgba(0,0,0,0.05); }
        p, div, span, label, input, .stMarkdown, .stSlider { color: #333 !important; }
        .stTextInput>div>div>input, .stSelectbox>div>div>select { background-color: #fff !important; color: #333 !important; border: 1px solid #ccc !important; border-radius: 8px; padding: 8px; }
        .stButton>button { background: linear-gradient(90deg, #ffb26b, #ff7b54); color: white; border-radius: 50px; padding: 12px 25px; transition: 0.3s; }
        .stButton>button:hover { background: linear-gradient(90deg, #ffa266, #ff6f4c); transform: scale(1.05); }
        </style>
    """, unsafe_allow_html=True)

st.title("🎬 MovieMatch AI — Find Perfect Recommendations 🍿")

lottie_anim = load_lottieurl("https://assets3.lottiefiles.com/packages/lf20_ydo1amjm.json")
st_lottie(lottie_anim, height=200)

api_key = st.secrets["api_key"]

unique_genres = sorted(set(
    genre.strip() for genres in movies_df['genres'].dropna() for genre in genres.split(',')
))

st.markdown("### 🎯 Tell us what you like:")
col1, col2 = st.columns(2)
with col1:
    movie_title = st.text_input("🎥 Movie you enjoyed:")
    genre_preference = st.selectbox("🎭 Favorite genre:", ["Any"] + unique_genres)
    director_input = st.text_input("🎬 Favorite director:")
    keyword_input = st.text_input("🔎 Keywords:")

with col2:
    cast_input = st.text_input("⭐ Favorite actor/actress:")
    top_n = st.slider("📈 Number of recommendations:", 5, 20, 10)

if st.button("🚀 Show Recommendations"):
    with st.spinner("Finding the perfect matches..."):
        recs = smart_recommendation(
            movies_df,
            title=movie_title,
            genre=genre_preference,
            director=director_input,
            cast_member=cast_input,
            keyword=keyword_input,
            top_n=top_n
        )

    if not recs.empty:
        st.markdown("## 🍿 Here’s what we found for you:")
        cols = st.columns(2)
        for idx, (_, row) in enumerate(recs.iterrows()):
            with cols[idx % 2]:
                details = get_movie_details(row['id'], api_key)
                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                if details['poster_url']:
                    st.image(details['poster_url'], width=200)
                st.markdown(f"### 🎥 {details['title']}")
                st.markdown(f"⭐ **Rating:** {details['rating']}/10")
                st.markdown(f"🎭 **Genres:** {row['genres']}")
                st.markdown(f"🎬 **Director:** {row['director_name']}")
                st.markdown(f"⭐ **Top Cast:** {row['top_cast']}")
                st.markdown(f"📖 **Plot:** {row['overview']}")
                st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("No recommendations found. Try changing your inputs!")

st.sidebar.title("💡 Coming Soon")
st.sidebar.markdown("- 🎞️ Watch trailers")
st.sidebar.markdown("- 📁 Save & share lists")
st.sidebar.markdown("- 📊 Community ratings")

