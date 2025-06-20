import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import yaml

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def preprocess_data(config):
    print("Indexing reviews...")
    
    # Load reviews
    reviews = pd.read_csv(config['data']['reviews_path'])
    documents = list(reviews['text'])
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config['model']['embedding_model'])
    model = AutoModel.from_pretrained(config['model']['embedding_model'])
    
    # Generate embeddings
    embeddings = []
    for doc in documents:
        inputs = tokenizer(doc, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.append(embedding)
    
    # Create FAISS index
    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    
    # Save index
    save_faiss_index(index, config['data']['embeddings_path'])
    print("Indexing done.")
    
    return documents

if __name__ == "__main__":
    config = load_config()
    preprocess_data(config)