import streamlit as st
from tmdbv3api import TMDb, Movie, Discover, Person
from sentence_transformers import SentenceTransformer, util
import torch
import os

# ----------------------
# Initialize TMDb
# ----------------------
tmdb = TMDb()
tmdb.api_key = os.environ["TMDB_API_KEY"]
tmdb.language = 'en'

movie_api = Movie()
discover = Discover()
person_api = Person()

# ----------------------
# Load AI Model (cached)
# ----------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ----------------------
# Helper Functions
# ----------------------
@st.cache_data
def get_person_id(person_name):
    res = person_api.search(person_name)
    return res[0].id if res else None


@st.cache_data
def get_movie_details(movie_name, release_year=None):
    search_results = movie_api.search(movie_name)
    if not search_results:
        return None

    movie = None
    for m in search_results:
        if m.title.lower() == movie_name.lower():
            if release_year and m.release_date:
                if m.release_date.startswith(str(release_year)):
                    movie = m
                    break
            else:
                movie = m
                break

    if movie is None:
        movie = search_results[0]

    movie_full = movie_api.details(movie.id)

    genres = [g.name for g in movie_full.genres] if hasattr(movie_full, 'genres') else []

    credits = movie_api.credits(movie.id)

    director = None
    for c in getattr(credits, 'crew', []):
        if c.job == 'Director':
            director = c.name
            break

    actors = [c.name for c in list(getattr(credits, 'cast', []))[:5]]

    return {
        'title': movie_full.title,
        'overview': movie_full.overview,
        'genres': genres,
        'director': director,
        'actors': actors,
        'language': getattr(movie_full, 'original_language', None)
    }


@st.cache_data
def get_related_movies(movie_details):
    related = {'genre': [], 'director': [], 'actors': []}

    genre_mapping = {
        'Action': 28, 'Adventure': 12, 'Drama': 18,
        'Fantasy': 14, 'Animation': 16, 'Comedy': 35
    }

    # ----------------------
    # Genre (same language)
    # ----------------------
    genre_ids = [genre_mapping[g] for g in movie_details['genres'] if g in genre_mapping]

    if genre_ids:
        results = discover.discover_movies({
            'with_genres': ','.join(map(str, genre_ids)),
            'with_original_language': movie_details['language'],
            'sort_by': 'popularity.desc'
        })

        for m in list(results)[:6]:
            related['genre'].append({
                'title': m.title,
                'overview': getattr(m, 'overview', '')
            })

    # ----------------------
    # Director (same language)
    # ----------------------
    if movie_details['director']:
        d_id = get_person_id(movie_details['director'])

        if d_id:
            results = discover.discover_movies({
                'with_crew': str(d_id),
                'with_original_language': movie_details['language'],
                'sort_by': 'popularity.desc'
            })

            for m in list(results)[:6]:
                if m.title != movie_details['title']:
                    related['director'].append({
                        'title': m.title,
                        'overview': getattr(m, 'overview', '')
                    })

    # ----------------------
    # Actors (Balanced + Top 3)
    # ----------------------
    seen = set()

    for actor_name in movie_details['actors'][:3]:
        actor_id = get_person_id(actor_name)

        if actor_id:
            results = discover.discover_movies({
                'with_cast': str(actor_id),
                'with_original_language': movie_details['language'],
                'sort_by': 'popularity.desc'
            })

            count = 0

            for m in list(results):
                if m.title != movie_details['title'] and m.title not in seen:

                    related['actors'].append({
                        'title': m.title,
                        'overview': getattr(m, 'overview', ''),
                        'actor': actor_name
                    })

                    seen.add(m.title)
                    count += 1

                if count == 3:
                    break

    return related


@st.cache_data
def recommend_by_ai_plot(movie_details, related_movies):
    pool = {}

    for category in related_movies:
        for m in related_movies[category]:
            pool[m['title']] = m['overview']

    pool[movie_details['title']] = movie_details['overview']

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

    movie_details = get_movie_details(movie_input, year_input if year_input else None)

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
            st.caption(f"Why: Same Director ({movie_details['director']})")
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
