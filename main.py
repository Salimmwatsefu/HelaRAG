import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

def load_config():
    # Config with user-specific file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'data': {
            'reviews_path': os.path.join(base_dir, 'data', 'raw', 'reviews.csv'),
            'embeddings_path': os.path.join(base_dir, 'data', 'processed', 'embeddings.faiss')
        },
        'model': {
            'name': 'helarag-finetuned-model'
        }
    }

def create_faiss_index(documents, embeddings_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    faiss.write_index(index, embeddings_path)
    print(f"✅ Saved FAISS index to {embeddings_path}")
    return index

def load_faiss_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    index = faiss.read_index(path)
    print(f"✅ Loaded FAISS index from {path}")
    return index

def retrieve_documents(query, documents, config, faiss_index, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], show_progress_bar=False).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def load_helarag_model(config):
    print(f"Loading domain model: {config['model']['name']}")
    return "HelaRAGModelObject"

def helarag_retrieve(query, documents, model, config):
    return [doc for doc in documents if query.lower() in doc.lower()]

def run_generation(query, context, config):
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large")
        # Split context into chunks if too long (BART has a max length limit)
        max_input_length = 1024
        context_chunks = [context[i:i+max_input_length] for i in range(0, len(context), max_input_length)]
        summaries = [summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] for chunk in context_chunks if chunk.strip()]
        response = " ".join(summaries) if summaries else "No relevant context to summarize."
        return response, [0.9]
    except Exception as e:
        print(f"⚠️ Summarization failed: {e}")
        return "Failed to generate a summary due to an error.", [0.5]

def optimize_response(query, response, config):
    return f"Optimized: {response}"

def check_knowledgebase2(query):
    return None

def run_pipeline(query="How do users feel about customer service?"):
    print("\n🔄 Running HelaRAG Parallel Pipeline...\n")

    config = load_config()
    
    print("📄 Loading reviews...")
    try:
        reviews_df = pd.read_csv(config['data']['reviews_path'])
        domain_documents = reviews_df['text'].tolist()
    except Exception as e:
        return f"❌ Error loading CSV: {e}", None, None

    if not domain_documents:
        return "❌ No reviews loaded.", None, None
    
    print(f"✅ Loaded {len(domain_documents)} reviews.\n")
    
    embeddings_path = config['data']['embeddings_path']
    if not os.path.exists(embeddings_path):
        print("📊 Creating FAISS index...")
        faiss_index = create_faiss_index(domain_documents, embeddings_path)
    else:
        faiss_index = load_faiss_index(embeddings_path)

    helarag_model = load_helarag_model(config)

    print(f"🔍 User Query: {query}\n")

    print("⚙️ Parallel retrieval...")
    faiss_context = retrieve_documents(query, domain_documents, config, faiss_index)
    helarag_context = helarag_retrieve(query, domain_documents, helarag_model, config)

    if not faiss_context and not helarag_context:
        return "❌ Retrieval failed.", None, None

    print("📊 Performing sentiment analysis for positive review queries...")
    if "positive reviews" in query.lower():
        try:
            sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
            sentiments = [sentiment_analyzer(doc)[0] for doc in domain_documents]
            positive_docs = [
                doc for doc, sent in zip(domain_documents, sentiments)
                if sent['label'] == 'POSITIVE' and ("loan" in doc.lower() or "wallet" in doc.lower())
            ]
            if positive_docs:
                response = f"Found {len(positive_docs)} positive reviews for Tala products. Sample positive reviews:\n" + "\n".join(positive_docs[:2])
            else:
                response = "No positive reviews for Tala products found."
            return f"Optimized: {response}", faiss_context, helarag_context
        except Exception as e:
            print(f"⚠️ Sentiment analysis failed: {e}")
            return f"Failed to analyze sentiment: {e}", faiss_context, helarag_context

    print("🤝 Negotiating context...")
    merged_context = "\n".join(faiss_context + helarag_context)

    print("🧠 Generating draft response...")
    draft_response, scores = run_generation(query, merged_context, config)
    print(f"\n✍️ Draft:\n{draft_response}")

    optimized_response = optimize_response(query, draft_response, config)

    kb2_response = check_knowledgebase2(query)
    if kb2_response:
        final_output = f"{optimized_response}\n\n[From KB2:]\n{kb2_response}"
    else:
        final_output = optimized_response

    return final_output, faiss_context, helarag_context

if __name__ == "__main__":
    final_output, faiss_context, helarag_context = run_pipeline()
    print(f"\n✅ Final Output:\n{final_output}")
    if faiss_context:
        print("\n📄 FAISS Retrieved Reviews:")
        for doc in faiss_context:
            print(f"- {doc}")
    if helarag_context:
        print("\n📄 HelaRAG Retrieved Reviews:")
        for doc in helarag_context:
            print(f"- {doc}")