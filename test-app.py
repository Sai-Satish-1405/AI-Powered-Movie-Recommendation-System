import streamlit as st
from tmdbv3api import TMDb, Movie, Discover, Person
from sentence_transformers import SentenceTransformer, util
import torch
import os

# ----------------------
# Initialize TMDb
# ----------------------
tmdb = TMDb()

if "TMDB_API_KEY" not in os.environ:
    st.error("TMDB_API_KEY not set")
    st.stop()

tmdb.api_key = os.environ["TMDB_API_KEY"]
tmdb.language = 'en'

movie_api = Movie()
discover = Discover()
person_api = Person()

BASE_IMG_URL = "https://image.tmdb.org/t/p/w200"

# ----------------------
# Load AI Model
# ----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------
# Utils
# ----------------------
def clean_text(text):
    return text.strip().lower() if text else ""

def render_movie_cards(movies):
    """Render movies in horizontal cards with posters, overview, and reason"""
    if not movies:
        st.write("No recommendations available.")
        return

    for i in range(0, len(movies), 3):
        cols = st.columns(3)
        for j, movie in enumerate(movies[i:i+3]):
            with cols[j]:
                if 'poster_path' in movie and movie['poster_path']:
                    st.image(BASE_IMG_URL + movie['poster_path'])
                st.markdown(f"**{movie['title']}**")
                if 'reason' in movie:
                    st.caption(movie['reason'])
                st.write(movie['overview'][:200] + ("..." if len(movie['overview']) > 200 else ""))

# ----------------------
# Movie Details
# ----------------------
@st.cache_data
def get_movie_details(movie_name, release_year=None):
    try:
        movie_name_clean = clean_text(movie_name)
        results = movie_api.search(movie_name)
        if not results:
            return None

        movie = None
        for m in results:
            if clean_text(m.title) == movie_name_clean:
                if release_year and getattr(m, 'release_date', None):
                    if m.release_date.startswith(str(release_year)):
                        movie = m
                        break
                else:
                    movie = m
                    break
        if not movie:
            movie = results[0]

        movie_full = movie_api.details(movie.id)
        credits = movie_api.credits(movie.id)

        director = None
        for c in getattr(credits, 'crew', []):
            if c.job == 'Director':
                director = {"name": c.name, "id": c.id}
                break

        actors = [
            {"name": c.name, "id": c.id}
            for c in list(getattr(credits, 'cast', []))[:5]
        ]

        genres = [{"id": g.id, "name": g.name} for g in getattr(movie_full, 'genres', [])]

        return {
            'title': movie_full.title,
            'overview': movie_full.overview or "",
            'genres': genres,
            'director': director,
            'actors': actors,
            'language': getattr(movie_full, 'original_language', None),
            'poster_path': getattr(movie_full, 'poster_path', None)
        }

    except Exception:
        return None

# ----------------------
# Related Movies
# ----------------------
@st.cache_data
def get_related_movies(movie_details):
    related = {'genre': [], 'director': [], 'actors': []}

    # ----------------------
    # Genre
    # ----------------------
    try:
        movie_genres = movie_details.get('genres', [])
        if movie_genres:
            genre_ids = [g['id'] for g in movie_genres]
            genre_map = {g['id']: g['name'] for g in movie_genres}

            results = discover.discover_movies({
                'with_genres': ','.join(map(str, genre_ids)),
                'with_original_language': movie_details['language'],
                'sort_by': 'popularity.desc'
            })

            count = 0
            for m in results:
                if m.title != movie_details['title']:
                    # Identify shared genres
                    m_genres = getattr(m, 'genre_ids', [])
                    shared_genres = [genre_map[g] for g in m_genres if g in genre_map]
                    reason = "Shared Genre: " + ", ".join(shared_genres) if shared_genres else "Similar Genre"

                    related['genre'].append({
                        'title': m.title,
                        'overview': getattr(m, 'overview', ''),
                        'poster_path': getattr(m, 'poster_path', None),
                        'reason': reason
                    })
                    count += 1
                if count == 6:
                    break
    except:
        pass

    # ----------------------
    # Director
    # ----------------------
    try:
        director = movie_details.get('director')
        if director and director.get("id"):
            credits = person_api.movie_credits(director["id"])
            directed_movies = [
                m for m in getattr(credits, 'crew', [])
                if getattr(m, 'job', None) == 'Director'
            ]
            directed_movies = sorted(
                directed_movies,
                key=lambda x: getattr(x, 'popularity', 0),
                reverse=True
            )
            for m in directed_movies[:6]:
                if m.title != movie_details['title']:
                    related['director'].append({
                        'title': m.title,
                        'overview': getattr(m, 'overview', ''),
                        'poster_path': getattr(m, 'poster_path', None),
                        'reason': f"Same Director ({director['name']})"
                    })
    except:
        pass

    # ----------------------
    # Actors
    # ----------------------
    try:
        seen = set()
        for actor in movie_details.get('actors', [])[:3]:
            if not actor.get("id"):
                continue
            credits = person_api.movie_credits(actor["id"])
            cast_movies = sorted(
                getattr(credits, 'cast', []),
                key=lambda x: getattr(x, 'popularity', 0),
                reverse=True
            )
            count = 0
            for m in cast_movies:
                if m.title != movie_details['title'] and m.title not in seen:
                    related['actors'].append({
                        'title': m.title,
                        'overview': getattr(m, 'overview', ''),
                        'poster_path': getattr(m, 'poster_path', None),
                        'reason': f"Actor → {actor['name']}"
                    })
                    seen.add(m.title)
                    count += 1
                if count == 3:
                    break
    except:
        pass

    return related

# ----------------------
# AI Recommendations
# ----------------------
@st.cache_data
def recommend_by_ai_plot(movie_details, related_movies):
    pool = {}
    for category in related_movies:
        for m in related_movies[category]:
            if m['overview']:
                pool[m['title']] = {
                    'overview': m['overview'],
                    'poster_path': m.get('poster_path', None),
                    'reason': f"Similar Storyline (AI)"
                }
    if movie_details['overview']:
        pool[movie_details['title']] = {
            'overview': movie_details['overview'],
            'poster_path': movie_details.get('poster_path', None),
            'reason': "Original Movie"
        }

    if len(pool) < 2:
        return []

    titles = list(pool.keys())
    plots = [pool[t]['overview'] for t in titles]
    embeddings = model.encode(plots, convert_to_tensor=True)
    query_idx = titles.index(movie_details['title'])
    scores = util.cos_sim(embeddings[query_idx], embeddings)[0]
    top = torch.topk(scores, k=min(6, len(titles)))

    recs = []
    for idx in top.indices:
        if titles[idx] != movie_details['title']:
            t = titles[idx]
            recs.append({
                'title': t,
                'overview': pool[t]['overview'],
                'poster_path': pool[t]['poster_path'],
                'reason': pool[t]['reason']
            })
    return recs

# ----------------------
# UI
# ----------------------
st.title("AI Powered Movie Recommendations")

movie_input = st.text_input("Enter your favorite movie:")
year_input = st.text_input("Optional: Release Year")

if st.button("Fetch Recommendations") and movie_input:
    with st.spinner("Fetching recommendations..."):
        movie_details = get_movie_details(
            movie_input,
            year_input if year_input.isdigit() else None
        )

        if not movie_details:
            st.error("Movie not found")
        else:
            st.subheader(f"{movie_details['title']}")
            if movie_details.get('poster_path'):
                st.image(BASE_IMG_URL + movie_details['poster_path'])
            st.write(movie_details['overview'])
            st.caption(f"Language: {movie_details['language']}")

            related = get_related_movies(movie_details)
            ai_recs = recommend_by_ai_plot(movie_details, related)

            st.subheader("Genre-based Recommendations")
            render_movie_cards(related['genre'])

            st.subheader("Director-based Recommendations")
            render_movie_cards(related['director'])

            st.subheader("Actor-based Recommendations")
            render_movie_cards(related['actors'])

            st.subheader("AI Similarity Recommendations")
            render_movie_cards(ai_recs)
