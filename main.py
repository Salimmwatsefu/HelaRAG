
import os
import pandas as pd
from src.retriever import load_config, load_faiss_index, retrieve_documents
from src.generator import run_generation

def run_pipeline():
    print("Running pipeline...")
    
    # Load configuration
    config = load_config()
    
    # Load documents (reviews from CSV)
    print("Loading reviews...")
    try:
        reviews_df = pd.read_csv(config['data']['reviews_path'])
        documents = reviews_df['text'].tolist()
    except Exception as e:
        print(f"Error loading reviews.csv: {e}")
        return
    
    if not documents:
        print("Error: No Tala reviews found in reviews.csv")
        return
    
    print(f"Loaded {len(documents)} reviews")
    
    # Load FAISS index
    print("Loading FAISS index...")
    if not os.path.exists(config['data']['embeddings_path']):
        print("Error: FAISS index not found at data/processed/embeddings.faiss")
        return
    index = load_faiss_index(config['data']['embeddings_path'])
    
    # Test query
    test_query = "How do users feel about customer service?"
    
    # Retrieve reviews
    print("Retrieving reviews...")
    context = "\n".join(retrieve_documents(test_query, documents, config))

    
    if not context:
        print("Error: No relevant reviews retrieved")
        return
    
    # Generate response
    response, scores = run_generation(test_query, context, config)
    
    print(f"Response: {response}")
    print(f"Scores: {scores}")

if __name__ == "__main__":
    run_pipeline()
