import streamlit as st
from tmdbv3api import TMDb, Movie, Discover, Person
from sentence_transformers import SentenceTransformer, util
import torch
import os

# ----------------------
# Initialize TMDb
# ----------------------
tmdb = TMDb()
tmdb.api_key = os.environ["TMDB_API_KEY"]  # Read from Streamlit Secrets
tmdb.language = 'en'
movie_api = Movie()
discover = Discover()
person_api = Person()


# ----------------------
# Initialize AI model
# ----------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------
# Helper Functions
# ----------------------
def get_person_id(person_name):
    search_results = person_api.search(person_name)
    if search_results:
        return search_results[0].id
    return None

def get_movie_details(movie_name, release_year=None):
    search_results = movie_api.search(movie_name)
    if not search_results:
        return None

    movie = None
    for m in search_results:
        title_match = m.title.lower() == movie_name.lower()
        year_match = True
        if release_year and hasattr(m, 'release_date') and m.release_date:
            year_match = m.release_date.startswith(str(release_year))
        if title_match and year_match:
            movie = m
            break
    if movie is None:
        movie = search_results[0]

    movie_full = movie_api.details(movie.id)
    genres = [g.name for g in movie_full.genres] if hasattr(movie_full, 'genres') else []

    credits = movie_api.credits(movie.id)
    director = None
    actors = []
    if hasattr(credits, 'crew'):
        for crew_member in credits.crew:
            if crew_member.job == 'Director':
                director = crew_member.name
                break
    if hasattr(credits, 'cast'):
        actors = [c.name for c in list(credits.cast)[:5]]

    return {
        'title': getattr(movie_full, 'title', None),
        'overview': getattr(movie_full, 'overview', None),
        'genres': genres,
        'director': director,
        'actors': actors,
        'release_date': getattr(movie_full, 'release_date', None),
        'runtime': getattr(movie_full, 'runtime', None),
        'rating': getattr(movie_full, 'vote_average', None),
        'language': getattr(movie_full, 'original_language', None)  # original_language
    }

def get_related_movies(movie_details, max_results=20):
    related = {'genre': [], 'director': [], 'actors': []}
    genre_mapping = {'Action':28, 'Adventure':12, 'Drama':18, 'Fantasy':14, 'Animation':16, 'Comedy':35}

    # Genre
    genre_ids = [genre_mapping[g] for g in movie_details['genres'] if g in genre_mapping]
    if genre_ids:
        movies_by_genre = discover.discover_movies({
            'with_genres': ','.join(map(str, genre_ids)),
             'with_original_language': movie_details['language'],  #1  language fix
            'sort_by': 'popularity.desc'
        })
        for m in list(movies_by_genre)[:max_results]:
            related['genre'].append({'title': m.title, 'overview': getattr(m, 'overview','')})

    # Director
    if movie_details['director']:
        director_id = get_person_id(movie_details['director'])
        if director_id:
            movies_by_director = discover.discover_movies({
                'with_crew': str(director_id),
                'sort_by': 'popularity.desc'
            })
            for m in list(movies_by_director)[:max_results]:
                if m.title != movie_details['title']:
                    related['director'].append({'title': m.title, 'overview': getattr(m,'overview','')})

    # Actors
    for actor_name in movie_details['actors'][:3]:
        actor_id = get_person_id(actor_name)
        if actor_id:
            movies_by_actor = discover.discover_movies({
                'with_cast': str(actor_id),
                'sort_by': 'popularity.desc'
            })
            for m in list(movies_by_actor)[:max_results]:
                if m.title != movie_details['title']:
                    related['actors'].append({'title': m.title, 'overview': getattr(m,'overview','')})

    return related

def recommend_by_ai_plot(movie_details, related_movies, top_n=6):
    pool = {}
    for category in ['genre','director','actors']:
        for m in related_movies[category]:
            pool[m['title']] = m['overview']
    pool[movie_details['title']] = movie_details['overview']

    titles = list(pool.keys())
    plots = list(pool.values())
    embeddings = model.encode(plots, convert_to_tensor=True)
    query_idx = titles.index(movie_details['title'])
    cos_scores = util.cos_sim(embeddings[query_idx], embeddings)[0]

    top_results = torch.topk(cos_scores, k=min(top_n+1,len(titles)))
    recommendations = []
    for idx in top_results.indices:
        if titles[idx] != movie_details['title']:
            recommendations.append({'title': titles[idx], 'overview': pool[titles[idx]]})
        if len(recommendations) >= top_n:
            break
    return recommendations

# ----------------------
# Streamlit UI
# ----------------------
st.title("🎬 AI-Powered Movie Recommendations")

movie_input = st.text_input("Enter your favorite movie:")
year_input = st.text_input("Optional: Release Year (e.g., 2024)")

if st.button("Fetch Recommendations") and movie_input:
    movie_details = get_movie_details(movie_input, release_year=year_input if year_input else None)
    if not movie_details:
        st.error("Movie not found!")
    else:
        st.subheader(f"Selected Movie: {movie_details['title']}")
        st.write(movie_details['overview'])
        st.caption(f"Language: {movie_details['language']}")

        related_movies = get_related_movies(movie_details, max_results=6)

        st.subheader("🎭 Genre-based Recommendations")
        for m in related_movies['genre']:
            st.write(f"**{m['title']}**: {m['overview']}")

        st.subheader("🎬 Director-based Recommendations")
        for m in related_movies['director']:
            st.write(f"**{m['title']}**: {m['overview']}")

        st.subheader("⭐ Actor-based Recommendations")
        for m in related_movies['actors']:
            st.write(f"**{m['title']}**: {m['overview']}")

        st.subheader("🤖 AI Plot Similarity Recommendations")
        ai_recs = recommend_by_ai_plot(movie_details, related_movies, top_n=6)
        for m in ai_recs:
            st.write(f"**{m['title']}**: {m['overview']}")
