import sqlite3
import re
import random
import pickle
import numpy as np
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import assemblyai as aai
import tempfile
import os

# Set up AssemblyAI API key
aai.settings.api_key = "426f3fbd27f946dd8e87df11195759d2"  # Replace with your API key
transcriber = aai.Transcriber()

# -----------------------------
# Preprocessing Data (Initial Setup)
# -----------------------------

# Step 1: Connect to the SQLite database
def get_data_from_db():
    conn = sqlite3.connect('eng_subtitles_database.db')
    cursor = conn.cursor()
    cursor.execute("SELECT num, name, content FROM zipfiles")  # Include subtitle name
    rows = cursor.fetchall()
    conn.close()
    return rows

# Step 2: Clean subtitle content
def clean_subtitle(content):
    text = content.decode('latin-1', errors='ignore')  # Decode binary content

    # Remove timestamps and non-alphanumeric characters
    text = re.sub(r'\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}', '', text)
    text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    return text

# Step 3: Get a sample of subtitles (for large datasets)
def get_sample_data(rows, sample_size=0.2):
    sample_rows = random.sample(rows, int(len(rows) * sample_size))
    return [(num, name, clean_subtitle(content)) for num, name, content in sample_rows]

# Step 4: Vectorization using TF-IDF and save results
def preprocess_and_save_data():
    try:
        # Fetch data from the database
        rows = get_data_from_db()

        # Sample and clean the data
        subtitles = get_sample_data(rows)

        # Extract subtitle content and IDs
        documents = [content for _, _, content in subtitles]
        subtitle_ids = [num for num, _, _ in subtitles]
        subtitle_names = [name for _, name, _ in subtitles]

        # Vectorize the subtitles
        vectorizer = TfidfVectorizer(stop_words='english')
        doc_vectors = vectorizer.fit_transform(documents)

        # Safely save processed data
        with open('processed_data.pkl', 'wb') as f:
            pickle.dump({
                'vectorizer': vectorizer,
                'doc_vectors': doc_vectors,
                'subtitle_ids': subtitle_ids,
                'subtitle_names': subtitle_names,
                'subtitles': subtitles
            }, f, protocol=pickle.HIGHEST_PROTOCOL)

        print("✅ Data preprocessing and saving completed successfully!")

    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")

# ---------------------------
# Verifying Pickle File
# ---------------------------
def verify_pickle_file():
    try:
        with open('processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        print("✅ Pickle file loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")

# ---------------------------
# Querying Functionality (for Streamlit UI)
# ---------------------------

# Load preprocessed data
def load_preprocessed_data():
    try:
        with open('processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading preprocessed data: {e}")
        return None

# Vectorize user query
def vectorize_query(query, vectorizer):
    return vectorizer.transform([query])

# Compute cosine similarity
def compute_similarity(query_vector, doc_vectors):
    similarities = cosine_similarity(query_vector, doc_vectors)
    return similarities.flatten()

# Get top subtitles based on similarity
def get_top_subtitles(similarity_scores, subtitles, subtitle_names, top_n=10):
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return [(subtitle_names[i], subtitles[i][0], similarity_scores[i], subtitles[i][2]) for i in top_indices]

# ---------------------------
# Audio Transcription Functionality
# ---------------------------

def transcribe_audio(uploaded_file):
    with st.spinner('Transcribing your audio...'):
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_audio_path = temp_file.name

        transcript = transcriber.transcribe(temp_audio_path)
        os.remove(temp_audio_path)

        st.success("Transcription complete!")
        return transcript.text

# ---------------------------
# Streamlit UI
# ---------------------------

def create_ui():
    st.title("🎧 Shazam Clone: Subtitle Search Engine & Audio Transcription")

    # Load preprocessed data
    data = load_preprocessed_data()
    if not data:
        st.warning("Data not preprocessed yet. Run the preprocessing function first.")
        if st.button("Preprocess and Save Data"):
            preprocess_and_save_data()
            verify_pickle_file()
        return

    vectorizer = data['vectorizer']
    doc_vectors = data['doc_vectors']
    subtitle_names = data['subtitle_names']
    subtitles = data['subtitles']

    # Audio Upload and Transcription
    st.header("🔊 Upload Audio for Transcription")
    uploaded_file = st.file_uploader("Choose an audio file (mp3, wav, flac)", type=["mp3", "wav", "flac"])

    if uploaded_file:
        st.audio(uploaded_file, format=uploaded_file.type)
        if st.button("Transcribe and Search"):
            transcribed_text = transcribe_audio(uploaded_file)
            st.text_area("📝 Transcribed Text:", transcribed_text, height=150)

            query_vector = vectorize_query(transcribed_text, vectorizer)
            similarity_scores = compute_similarity(query_vector, doc_vectors)
            top_subtitles = get_top_subtitles(similarity_scores, subtitles, subtitle_names)

            st.subheader("🔍 Search Results:")
            if top_subtitles:
                for name, sub_id, score, content in top_subtitles:
                    st.write(f"**Subtitle Name:** {name}")
                    st.write(f"**Subtitle ID:** {sub_id}")
                    st.write(f"**Similarity Score:** {score:.4f}")
                    st.markdown("---")
            else:
                st.write("No relevant subtitles found.")

    # Text-Based Search
    st.header("🔍 Search Subtitles by Text")
    query = st.text_input("Enter your search query:")

    if query:
        query_vector = vectorize_query(query, vectorizer)
        similarity_scores = compute_similarity(query_vector, doc_vectors)
        top_subtitles = get_top_subtitles(similarity_scores, subtitles, subtitle_names)

        st.subheader("🔍 Search Results:")
        if top_subtitles:
            for name, sub_id, score, content in top_subtitles:
                st.write(f"**Subtitle Name:** {name}")
                st.write(f"**Subtitle ID:** {sub_id}")
                st.write(f"**Similarity Score:** {score:.4f}")
                st.markdown("---")
        else:
            st.write("No relevant subtitles found.")

# ---------------------------
# Main Function to Run the App
# ---------------------------
if __name__ == "__main__":
    #preprocess_and_save_data()
    create_ui()
