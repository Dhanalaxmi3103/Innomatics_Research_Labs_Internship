# ==========================================
# 1️⃣ Import Libraries
# ==========================================

import streamlit as st
import joblib
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')


# ==========================================
# 2️⃣ Load Saved Model & Vectorizer
# ==========================================

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("best_model.pkl")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ==========================================
# 3️⃣ Text Cleaning Function
# ==========================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)


# ==========================================
# 4️⃣ Streamlit UI
# ==========================================

st.set_page_config(page_title="Sentiment Analysis App")

st.title("Sentiment Analysis of Customer Review ")
st.write("Enter a customer review below to analyze sentiment.")


review_text = st.text_area("Enter Customer Review")


# ==========================================
# 5️⃣ Prediction Logic
# ==========================================

if st.button("Analyze Sentiment"):

    if review_text.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_text)

        vectorized = vectorizer.transform([cleaned])

        prediction = model.predict(vectorized)[0]
        probability = model.predict_proba(vectorized)[0]

        confidence = np.max(probability) * 100

        st.subheader("🔎 Result")

        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")

        st.write(f"Confidence Score: **{confidence:.2f}%**")
