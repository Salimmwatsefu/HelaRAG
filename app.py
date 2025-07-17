import streamlit as st
import pandas as pd
from main import run_pipeline, load_config, create_faiss_index, load_faiss_index, load_helarag_model, load_or_create_helarag_embeddings

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="HelaRAG: Financial Insights",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Custom Financial-Themed CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

body, .stApp {background-color: #F2F2F2; color: #070B0D; font-family: 'Inter', sans-serif;}

.helarag-header {text-align: center; margin: 2rem auto;}
.helarag-title {font-size: 3.5rem; font-weight: 700; color: #204359; letter-spacing: 1px;}
.helarag-subtitle {font-size: 1.4rem; color: #204959;  margin: 1rem auto;}

.stTextInput>div>div>input {background:#FFFFFF;border:1px solid #84C9D9;color:#070B0D;font-size:1.1rem;padding:1rem;border-radius:12px;}
.stTextInput>div>div>input:focus {border-color:#6AB9D9;box-shadow:0 0 10px rgba(106,185,217,0.3);}

.stButton>button {background:#204359;color:white;font-size:1.2rem;font-weight:600;padding:0.8rem 1.8rem;border-radius:8px;border:none;transition:all 0.3s ease;}
.stButton>button:hover {background:#204959;transform:translateY(-2px);box-shadow:0 4px 12px rgba(32,73,89,0.4);}

.result-card,.stExpander {background:white;border-radius:12px;padding:1.5rem;margin:1.5rem auto;box-shadow:0 4px 12px rgba(0,0,0,0.1);border:1px solid #E2E8F0;}
.answer-content {font-size:1.2rem;color:#204359;line-height:1.7;}
.knowledge-tag {background:#6AB9D9;color:#070B0D;padding:0.4rem 1rem;border-radius:16px;font-weight:600;}
.context-item {background:#F8FAFC;padding:0.8rem;border-left:4px solid #204359;border-radius:10px;margin:0.5rem 0;}
.footer {text-align:center;color:#204959;padding:2rem 0;font-size:0.9rem;}
.power-badge {display:inline-block;margin:0.4rem;padding:0.4rem 1rem;border-radius:20px;background:#84C9D9;color:#070B0D;border:none;}

@media(max-width:768px){.helarag-title{font-size:2.5rem;}.helarag-subtitle{font-size:1rem;}}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class='helarag-header'>
    <h1 class='helarag-title'>💹 HelaRAG Financial Insights</h1>
    <p class='helarag-subtitle'>Welcome! This tool leverages **Retrieval-Augmented Generation (RAG)** to provide insights into user feedback across various financial applications.</p>
</div>
""", unsafe_allow_html=True)

# Load Config
config = load_config()

# Cache Resources
@st.cache_resource
def load_data_and_models(_config):
    try:
        reviews_df = pd.read_csv(_config['data']['reviews_path'])
        domain_documents = reviews_df['text'].tolist()
        if not domain_documents:
            st.error("❌ No reviews found."); st.stop()
    except FileNotFoundError:
        st.error("❌ 'reviews.csv' not found."); st.stop()

    embeddings_path = _config['data']['embeddings_path']
    try:
        faiss_index = load_faiss_index(embeddings_path)
    except FileNotFoundError:
        with st.spinner("Creating new FAISS index..."):
            faiss_index = create_faiss_index(domain_documents, embeddings_path)

    helarag_model = load_helarag_model(_config)
    load_or_create_helarag_embeddings(domain_documents, _config)
    return domain_documents, faiss_index, helarag_model

with st.spinner("🔍 Initializing knowledge base..."):
    domain_documents, faiss_index, helarag_model = load_data_and_models(config)

# Input Section
st.subheader("🔍 Ask Your Financial Question")
query_input = st.text_input("Your Query", placeholder="e.g., What are customers saying about payment processing?", label_visibility="collapsed")
submit_button = st.button("Analyze")

if submit_button and query_input:
    with st.spinner("✨ Generating insights..."):
        final_output, context_docs, kb2_used = run_pipeline(query=query_input)
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.subheader("📊 Analysis Results")
        if kb2_used:
            st.markdown("<div class='knowledge-tag'>General Knowledge Used</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='answer-content'>{final_output}</div>", unsafe_allow_html=True)
        with st.expander("📚 Source Context"):
            if context_docs:
                for i, doc in enumerate(context_docs, 1):
                    st.markdown(f"<div class='context-item'><b>Source {i}:</b> {doc}</div>", unsafe_allow_html=True)
            else:
                st.info("No specific context retrieved")
        st.markdown("</div>", unsafe_allow_html=True)
elif submit_button:
    st.warning("Please enter a question to analyze.")

# Footer
st.markdown("""
<div class='footer'>
    <div>Powered by advanced AI technologies</div>
    <div>
        <span class='power-badge'>Gemini API</span>
        <span class='power-badge'>Sentence Transformers</span>
        <span class='power-badge'>FAISS</span>
        <span class='power-badge'>Streamlit</span>
    </div>
    <div style='margin-top:1rem;color:#64748b;font-size:0.8rem'>HelaRAG Financial Insights v1.3</div>
</div>
""", unsafe_allow_html=True)
