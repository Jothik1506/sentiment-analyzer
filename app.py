import streamlit as st
from model import predict_sentiment

st.title("Sentiment Analyzer (IMDB / Twitter)")
dataset = st.selectbox("Choose Dataset:", ["imdb", "twitter"])
user_input = st.text_area("Enter a sentence or review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence to analyze.")
    else:
        result = predict_sentiment(user_input, dataset)
        st.success(f"Predicted Sentiment: {result}")
        st.success(f"Predicted Sentiment: 😀 {result}" if result == "Positive" else f"Predicted Sentiment: 😠 {result}")



