import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(
    page_title="Arun Date Song Recommendation",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 Arun Date Song Recommendation System")
st.write("Content-based recommendation using audio features")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Arundate songs - Sheet1.csv")  
    return df

df = load_data()

# Display dataset
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Features for recommendation
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

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features])

# Cosine similarity
similarity = cosine_similarity(scaled_features)

# Song selection
selected_song = st.selectbox(
    "🎶 Select a Song",
    df["Track Name"].values
)

# Recommendation function
def recommend(song_name):
    index = df[df["Track Name"] == song_name].index[0]
    distances = similarity[index]
    song_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommendations = []
    for i in song_list:
        recommendations.append(df.iloc[i[0]]["Track Name"])
    return recommendations

# Button
if st.button("Recommend Songs"):
    st.subheader("🎧 Recommended Songs")
    for song in recommend(selected_song):
        st.write("👉", song)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
