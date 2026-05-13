# AI-Powered-Movie-Recommendation-System
AI-Powered Movie Recommendation System

# 🎬 AI-Powered Movie Recommendation System

An intelligent movie recommendation system that suggests similar movies based on **genre, director, actors, and AI-powered plot similarity** using embeddings.

---
## 🚀 Features

* 🔍 Search any movie using TMDb API
* 🎭 Genre-based recommendations (Genre-based keywords)
* 🎬 Director-based recommendations
* ⭐ Top Actor-based recommendations
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

Create a `.env` file or set an environment variable:

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

Then open: (Local)

```
http://localhost:8501
```

---

## 🌐 Deployment

This app was deployed easily using:

* **Streamlit Cloud** --Deployed Try here -- 🔥 🔗 https://ai-powered-movie-recommendation-system-by-ssp14-test.streamlit.app/ 
* Docker (optional)--N/A
* AWS / GCP (advanced)--N/A

---

## 📸 Screenshots 

<img width="1919" height="816" alt="image" src="https://github.com/user-attachments/assets/e0702a95-955e-4e77-8c2d-c46fdac3693f" />
<img width="1919" height="825" alt="image" src="https://github.com/user-attachments/assets/f5b91deb-0a11-4df6-988b-36e5675b282d" />
<img width="1919" height="816" alt="image" src="https://github.com/user-attachments/assets/dcca81dd-c81c-4022-9082-782adca864b6" />
<img width="1919" height="815" alt="image" src="https://github.com/user-attachments/assets/2e2368b1-c70a-4d00-8965-f0ab7b29c529" />
<img width="1919" height="822" alt="image" src="https://github.com/user-attachments/assets/35f03dcd-6349-47d2-ac4b-97f46ac3943c" />
<img width="1911" height="821" alt="image" src="https://github.com/user-attachments/assets/242bca2a-62ef-4835-9fa4-d2d0916c68e7" />
<img width="1919" height="824" alt="image" src="https://github.com/user-attachments/assets/2b2d29eb-f475-41a9-bbca-fe727dc21c23" />

---

## 🔥 Example

Input:

```
Interstellar
```

Output:

* Genre-based: Similar shared genre: Drama, Adventure, Science Fiction
* Director-based: Movies by the Same Director (Christopher Nolan)
* Actor-based: Movies with Actor → Matthew McConaughey
* AI-based: Movies with Similar Storyline (AI)

---

## 🎯 Future Improvements

* 🎬 Add movie posters - ✔️ 
* 👤 User-based personalization …. Loading
* 📊 Ratings & reviews integration …. Loading
* 🔎 Advanced filtering (year, rating, language) …. Loading
* ⚡ Faster vector search using FAISS …. Loading

---

## 🤝 Contributing 🥳

Contributions are welcome! Feel free to open issues or submit pull requests.

---

## 📄 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Your Name
GitHub: https://github.com/Sai-Satish-1405

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!


Try here  🔗 https://ai-powered-movie-recommendation-system-by-ssp14-test.streamlit.app/ 

