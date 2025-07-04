import streamlit as st
from main import run_pipeline

# Set page title and layout
st.set_page_config(page_title="HelaRAG Financial Insights", layout="wide")


# Main content
st.title("HelaRAG Multilingual RAG Pipeline")
st.markdown("""
Welcome to the HelaRAG Pipeline for Tala Financial Insights. Ask about customer feedback
or product sentiments to get insights from 8,393 reviews.
""")

# Input query
query = st.text_input("Enter your query:", value="What are the most positive reviews?")

# Button to run the pipeline
if st.button("Get Insights"):
    with st.spinner("Running HelaRAG pipeline..."):
        result, faiss_context, helarag_context = run_pipeline(query)
        
        # Display response
        st.subheader("Response")
        if isinstance(result, str) and result.startswith("❌"):
            st.error(result)
        else:
            st.write(result)
        
        # Display retrieved reviews
        if faiss_context:
            st.subheader("Retrieved Reviews (FAISS - Vector-Based)")
            for i, doc in enumerate(faiss_context, 1):
                st.write(f"{i}. {doc}")
        
        if helarag_context:
            st.subheader("Retrieved Reviews (HelaRAG - Keyword-Based)")
            for i, doc in enumerate(helarag_context, 1):
                st.write(f"{i}. {doc}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and HelaRAG. Powered by xAI.")