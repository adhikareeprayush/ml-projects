
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(
    page_title="Movie Rating Prediction",
    layout="wide"
)

# Load model and scaler
@st.cache_resource
def load_model():
    model = joblib.load("movie_rating_model.joblib")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

st.title("Movie Rating Prediction")
st.markdown("This app predicts movie ratings based on popularity, vote count, and release date.")

if not model_loaded:
    st.error("Model files not found. Please run the notebook first to train and save the model.")
    st.stop()

st.header("Predict Movie Rating")

col1, col2 = st.columns(2)

with col1:
    vote_count = st.number_input(
        "Vote Count",
        min_value=0,
        max_value=50000,
        value=500,
        help="Number of votes the movie has received"
    )
    popularity = st.number_input(
        "Popularity Score",
        min_value=0.0,
        max_value=1000.0,
        value=50.0,
        step=1.0,
        help="TMDB popularity score"
    )

with col2:
    release_date = st.date_input(
        "Release Date",
        value=datetime.now(),
        help="Movie release date"
    )
    release_year = release_date.year
    release_month = release_date.month
    release_day = release_date.day
    st.info(f"Year: {release_year} | Month: {release_month} | Day: {release_day}")

st.divider()

if st.button("Predict Rating", type="primary", use_container_width=True):
    # Prepare input data
    input_data = pd.DataFrame({
        'vote_count': [vote_count],
        'popularity': [popularity],
        'release_year': [release_year],
        'release_month': [release_month],
        'release_day': [release_day]
    })
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    # Clamp prediction to valid range
    prediction = max(0, min(10, prediction))
    # Display results
    st.subheader("Prediction Result")
    st.metric("Predicted Rating", f"{prediction:.2f} / 10")
    st.progress(prediction / 10, text=f"Rating: {prediction:.2f} / 10")
