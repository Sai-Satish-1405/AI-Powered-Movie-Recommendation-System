# AI-Powered-Movie-Recommendation-System
AI-Powered Movie Recommendation System

# 🎬 AI-Powered Movie Recommendation System

An intelligent movie recommendation system that suggests similar movies based on **genre, director, actors, and AI-powered plot similarity** using embeddings.

---

## 🚀 Features

* 🔍 Search any movie using TMDb API
* 🎭 Genre-based recommendations
* 🎬 Director-based recommendations
* ⭐ Actor-based recommendations
* 🤖 AI-powered plot similarity using embeddings
* ⚡ Hybrid recommendation engine (Rule-based + AI scoring)
* 🧠 Explainable recommendations (Why each movie is suggested)
* ⚡ Caching for faster performance
* 🌐 Interactive UI built with Streamlit

---

## 🧠 How It Works

### 1. Data Source

* Uses **TMDb (The Movie Database) API** to fetch:

  * Movie details
  * Genres
  * Director
  * Cast

---

### 2. Recommendation Strategies

#### 🎭 Genre-Based

Suggests movies with similar genres.

#### 🎬 Director-Based

Recommends movies by the same director.

#### ⭐ Actor-Based

Finds movies featuring the same actors.

#### 🤖 AI Plot Similarity

* Converts movie plots into embeddings using **Sentence Transformers**
* Uses **cosine similarity** to find semantically similar movies

---

### 3. Hybrid Scoring System

Combines:

* Rule-based signals (genre, director, actors)
* AI similarity scores

This improves recommendation accuracy and relevance.

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Backend:** Python
* **API:** TMDb API
* **ML/NLP:** Sentence Transformers
* **Libraries:**

  * `streamlit`
  * `tmdbv3api`
  * `sentence-transformers`
  * `torch`

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup API Key

Create a `.env` file or set environment variable:

```bash
TMDB_API_KEY=your_api_key_here
```

Or inside the code:

```python
tmdb.api_key = "your_api_key"
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

## 🌐 Deployment

This app can be deployed easily using:

* **Streamlit Cloud** (Recommended)
* Docker (optional)
* AWS / GCP (advanced)

---

## 📸 Screenshots (Optional)

*Add screenshots of your app here to make your repo stand out*

---

## 🔥 Example

Input:

```
Baahubali
```

Output:

* Genre-based: Similar epic/action movies
* Director-based: Movies by S.S. Rajamouli
* Actor-based: Movies with Prabhas
* AI-based: Movies with similar storyline

---

## 🎯 Future Improvements

* 🎬 Add movie posters
* 👤 User-based personalization
* 📊 Ratings & reviews integration
* 🔎 Advanced filtering (year, rating, language)
* ⚡ Faster vector search using FAISS

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Your Name
GitHub: https://github.com/your-username

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!


Try here https://ai-powered-movie-recommendation-system-by-ssp14.streamlit.app/

