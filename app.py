import streamlit as st

from services.summarizer_service import generate_article

st.set_page_config(page_title="Youtube Video To Article Generator", layout="wide")

st.title("🚀 YouTube → Article Generator")

# ---------------- INPUT ----------------
url = st.text_input("Enter YouTube URL")

generate_article_btn = st.button("Generate Article")

# ---------------- GENERATE ARTICLE ----------------
if generate_article_btn and url:
    with st.spinner("Generating article..."):
        article = generate_article(url)
        st.session_state["article"] = article

# ---------------- SHOW ARTICLE ----------------
if "article" in st.session_state:
    st.markdown(st.session_state["article"])