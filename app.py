import pickle
from pathlib import Path

import streamlit as st


APP_DIR = Path(__file__).resolve().parent


st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered",
)


@st.cache_resource
def load_model():
    with (APP_DIR / "model.pkl").open("rb") as file:
        model = pickle.load(file)

    with (APP_DIR / "tfidf_vectorizer.pkl").open("rb") as file:
        vectorizer = pickle.load(file)

    return model, vectorizer


st.title("📰 Fake News Detector")
st.write(
    "Enter a news article below and the machine learning model "
    "will classify it as **REAL** or **FAKE**."
)
st.divider()

title = st.text_input("News Title", placeholder="Enter the news headline...")
author = st.text_input("Author", placeholder="Enter author name...")
article = st.text_area(
    "News Article",
    placeholder="Paste the news article here...",
    height=250,
)

if st.button("🔍 Check News", use_container_width=True):
    if not title.strip() and not article.strip():
        st.warning("Please enter a news title or article.")
    else:
        model, vectorizer = load_model()
        total_text = f"{title} {author} {article}"
        transformed_text = vectorizer.transform([total_text])
        prediction = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = probabilities[int(prediction)] * 100

        if prediction == 1:
            st.error("🚨 FAKE NEWS")
        else:
            st.success("✅ REAL NEWS")

        st.metric("Model Confidence", f"{confidence:.2f}%")
        st.info(
            "This prediction is based on patterns learned from the training "
            "dataset. It should not be treated as independent fact verification."
        )

with st.sidebar:
    st.header("🤖 Model Information")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Feature Extraction:** TF-IDF")
    st.write("**N-grams:** Unigrams + Bigrams")
    st.write("**Dataset:** Kaggle Fake News Dataset")
    st.divider()
    st.caption("NLP-based Fake News Classification")