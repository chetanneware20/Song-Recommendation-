import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Arun Date Song Recommendation",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Arun Date Song Recommendation System")
st.write("Content-based recommendation using audio features")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "arundate_songs.csv")
    return pd.read_csv(file_path)

df = load_data()

# ---------------- SHOW DATASET ----------------
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# ---------------- FEATURES ----------------
features = [
    "Danceability",
    "Energy",
    "Loudness",
    "Speechiness",
    "Acousticness",
    "Instrumentalness",
    "Liveness",
    "Valence",
    "Tempo"
]

# ---------------- SCALE + SIMILARITY ----------------
@st.cache_data
def compute_similarity(data):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    similarity = cosine_similarity(scaled_features)
    return similarity

similarity = compute_similarity(df)

# ---------------- SONG SELECTION ----------------
selected_song = st.selectbox(
    "🎶 Select a Song",
    sorted(df["Track Name"].dropna().unique())
)

# ---------------- RECOMMEND FUNCTION ----------------
def recommend(song_name, n=5):
    if song_name not in df["Track Name"].values:
        return []

    index = df[df["Track Name"] == song_name].index[0]
    scores = list(enumerate(similarity[index]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]

    return [df.iloc[i[0]]["Track Name"] for i in scores]

# ---------------- BUTTON ----------------
if st.button("Recommend Songs"):
    st.subheader("🎧 Recommended Songs")
    recommendations = recommend(selected_song)

    if recommendations:
        for song in recommendations:
            st.write("👉", song)
    else:
        st.warning("No recommendations found.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
