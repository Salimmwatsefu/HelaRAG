
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
import yaml

def load_config():
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

def load_faiss_index(path):
    return faiss.read_index(path)

def get_embedding(query, tokenizer, model):
    inputs = tokenizer(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding

def retrieve_documents(query, documents, config):
    print("Retrieving reviews...")
    
    tokenizer = AutoTokenizer.from_pretrained(config['model']['embedding_model'])
    model = AutoModel.from_pretrained(config['model']['embedding_model'])
    
    # Get query embedding
    query_embedding = get_embedding(query, tokenizer, model)
    
    # Load FAISS index
    index = load_faiss_index(config['data']['embeddings_path'])
    
    # Retrieve top-k documents
    k = config['retrieval']['top_k']
    distances, indices = index.search(np.array([query_embedding]), k)
    print(f"Retrieved indices: {indices[0]}")
    
    # Validate indices
    valid_indices = [idx for idx in indices[0] if idx >= 0 and idx < len(documents)]
    if not valid_indices:
        print("No valid documents retrieved")
        return []
    
    retrieved_docs = [documents[idx] for idx in valid_indices]
    print(f"Retrieved documents: {retrieved_docs}")
    return retrieved_docs

if __name__ == "__main__":
    config = load_config()
    documents = []  # Placeholder
    test_query = "Hidden fees"
    docs = retrieve_documents(test_query, documents, config)
    print("Retrieved:", docs)
