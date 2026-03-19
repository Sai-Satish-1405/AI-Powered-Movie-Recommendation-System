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

        # Better matching
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

        # Director (WITH ID ✅)
        director = None
        for c in getattr(credits, 'crew', []):
            if c.job == 'Director':
                director = {"name": c.name, "id": c.id}
                break

        # Actors (WITH ID ✅)
        actors = [
            {"name": c.name, "id": c.id}
            for c in list(getattr(credits, 'cast', []))[:5]
        ]

        genres = [g.name for g in getattr(movie_full, 'genres', [])]

        return {
            'title': movie_full.title,
            'overview': movie_full.overview or "",
            'genres': genres,
            'director': director,
            'actors': actors,
            'language': getattr(movie_full, 'original_language', None)
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
    # Genre (LESS restrictive)
    # ----------------------
    try:
        if movie_details['genres']:
            results = discover.discover_movies({
                'with_genres': '',
                'sort_by': 'popularity.desc'
            })

            for m in list(results)[:10]:
                if m.title != movie_details['title']:
                    related['genre'].append({
                        'title': m.title,
                        'overview': getattr(m, 'overview', '')
                    })
    except:
        pass

    # ----------------------
    # Director (FIXED ✅)
    # ----------------------
    try:
        director = movie_details['director']

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
                        'overview': getattr(m, 'overview', '')
                    })
    except:
        pass

    # ----------------------
    # Actors (IMPROVED ✅)
    # ----------------------
    try:
        seen = set()

        for actor in movie_details['actors'][:3]:
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
                        'actor': actor['name']
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
            if m['overview']:  # skip empty plots
                pool[m['title']] = m['overview']

    if movie_details['overview']:
        pool[movie_details['title']] = movie_details['overview']

    if len(pool) < 2:
        return []

    titles = list(pool.keys())
    plots = list(pool.values())

    embeddings = model.encode(plots, convert_to_tensor=True)

    query_idx = titles.index(movie_details['title'])
    scores = util.cos_sim(embeddings[query_idx], embeddings)[0]

    top = torch.topk(scores, k=min(6, len(titles)))

    recs = []
    for idx in top.indices:
        if titles[idx] != movie_details['title']:
            recs.append({
                'title': titles[idx],
                'overview': pool[titles[idx]]
            })

    return recs


# ----------------------
# UI
# ----------------------
st.title("🎬 AI Powered Movie Recommendations")

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
            st.subheader(f"🎥 {movie_details['title']}")
            st.write(movie_details['overview'])
            st.caption(f"Language: {movie_details['language']}")

            related = get_related_movies(movie_details)

            # Genre
            st.subheader("🎭 Genre-based")
            for m in related['genre']:
                st.write(f"**{m['title']}**")
                st.caption("Why: Similar Genre")
                st.write(m['overview'])

            # Director
            st.subheader("🎬 Director-based")
            for m in related['director']:
                st.write(f"**{m['title']}**")
                st.caption(f"Why: Same Director ({movie_details['director']['name'] if movie_details['director'] else 'Unknown'})")
                st.write(m['overview'])

            # Actors
            st.subheader("⭐ Actor-based")
            for m in related['actors'][:6]:
                st.write(f"**{m['title']}**")
                st.caption(f"Why: Actor → {m['actor']}")
                st.write(m['overview'])

            # AI
            st.subheader("🤖 AI Similarity")
            ai_recs = recommend_by_ai_plot(movie_details, related)

            for m in ai_recs:
                st.write(f"**{m['title']}**")
                st.caption("Why: Similar Storyline (AI)")
                st.write(m['overview'])
