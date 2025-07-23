import os
import csv
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import google.generativeai as genai
from openai import OpenAI
from difflib import get_close_matches
import google.api_core.exceptions

# --- Load Environment Variables ---
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Configure OpenAI client to use OpenRouter API endpoint
openai_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# --- Global Cache ---
helarag_doc_embeddings = None

# --- KB2 ---
kb2_fallbacks = {
    "refund process": "For specific refund inquiries, please refer to the support section or FAQ within your financial app, or contact its customer service directly.",
    "how to reset pin": "To reset your PIN or password, look for a 'Forgot PIN/Password' option on your app's login screen, or check the app's help section.",
    "customer care number": "Specific customer care numbers vary by app. Please open your financial app and navigate to the 'Help', 'Contact Us', or 'Support' section to find the relevant contact information.",
    "account closure": "To close an account, typically you need to contact the customer support of the specific financial app you are using, as procedures vary.",
    "loan eligibility": "Loan eligibility criteria differ for each financial app. Please check the terms and conditions or the FAQ section within your chosen app for details.",
    "data privacy": "All reputable financial apps adhere to strict data privacy policies. You can usually find their detailed privacy policy within the app's settings or on their official website.",
    "app not working": "If your app is not working, try restarting your device, clearing the app's cache, or reinstalling the app. If the issue persists, contact the app's specific customer support."
}

# --- Config ---
def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'data': {
            'reviews_path': os.path.join(base_dir, 'data', 'raw', 'reviews.csv'),
            'embeddings_path': os.path.join(base_dir, 'data', 'processed', 'embeddings.faiss'),
            'helarag_embeddings_path': os.path.join(base_dir, 'data', 'processed', 'helarag_embeddings.npy')
        },
        'model': {
            'name': 'helarag-finetuned-model',
            'gpt_name': 'gemini-2.5-flash'
        }
    }

# --- Indexing ---
def create_faiss_index(documents, embeddings_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(documents, show_progress_bar=True).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    faiss.write_index(index, embeddings_path)
    return index

def load_faiss_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found at {path}")
    return faiss.read_index(path)

# --- Retrieval ---
def retrieve_documents(query, documents, config, faiss_index, top_k=5):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query], show_progress_bar=False).astype('float32')
    distances, indices = faiss_index.search(query_embedding, top_k)
    results = []
    for i, dist in zip(indices[0], distances[0]):
        similarity = 1 / (1 + dist)
        results.append((documents[i], similarity))
    return results

def load_helarag_model(config):
    return SentenceTransformer('all-MiniLM-L6-v2')

def load_or_create_helarag_embeddings(documents, config):
    global helarag_doc_embeddings
    embeddings_path = config['data']['helarag_embeddings_path']
    if os.path.exists(embeddings_path):
        print(f"Loading cached HelaRAG embeddings from {embeddings_path}")
        helarag_doc_embeddings = np.load(embeddings_path)
    else:
        print("Creating new HelaRAG embeddings...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        helarag_doc_embeddings = model.encode(documents, show_progress_bar=True)
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        np.save(embeddings_path, helarag_doc_embeddings)
    return helarag_doc_embeddings

def helarag_retrieve(query, documents, model, config, top_k=5):
    global helarag_doc_embeddings
    if helarag_doc_embeddings is None:
        helarag_doc_embeddings = load_or_create_helarag_embeddings(documents, config)
    query_embedding = model.encode([query], show_progress_bar=False)
    similarities = cosine_similarity(query_embedding, helarag_doc_embeddings)[0]
    top_indices = similarities.argsort()[::-1][:top_k]
    return [(documents[i], float(similarities[i])) for i in top_indices]

# --- AI Model Retrievals ---
def gemini_retrieve(query, model="gemini-2.5-flash"):
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(query)
        return [(response.text.strip(), 1.0)]
    except Exception as e:
        print(f"⚠️ Gemini retrieval failed: {e}")
        return []

def openai_retrieve(query, model="openai/gpt-4o"):
    extra_headers = {
        "HTTP-Referer": "https://yourwebsite.com",
        "X-Title": "HelaRAG Pipeline"
    }
    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert assistant helping with customer feedback analysis."},
                {"role": "user", "content": query}
            ],
            max_tokens=300,
            temperature=0.3,
            extra_headers=extra_headers
        )
        answer = response.choices[0].message.content
        return [(answer.strip(), 1.0)]
    except Exception as e:
        print(f"⚠️ OpenAI retrieval failed: {e}")
        return []

# --- Negotiation Layer ---
def negotiate_context(faiss_results, helarag_results, gemini_results=None, openai_results=None, strategy="weighted_top"):
    combined = faiss_results + helarag_results
    if gemini_results:
        combined += gemini_results
    if openai_results:
        combined += openai_results

    if strategy == "weighted_top":
        max_score = max([score for _, score in combined]) or 1.0
        combined = [(doc, score / max_score) for doc, score in combined]
        seen = set()
        sorted_docs = []
        for doc, score in sorted(combined, key=lambda x: x[1], reverse=True):
            if doc not in seen:
                sorted_docs.append((doc, score))
                seen.add(doc)
        return [doc for doc, score in sorted_docs[:5]]
    return [doc for doc, _ in combined]

# --- Ensemble Answer Synthesis ---
def ensemble_synthesis(query, faiss_docs, hela_docs, gemini_ans, openai_ans, model="gemini-2.5-flash"):
    context = "\n".join([doc for doc, _ in faiss_docs + hela_docs])
    combined = f"""
Query: {query}

Context from FAISS + HelaRAG:
{context}

Gemini Answer:
{gemini_ans[0][0] if gemini_ans else 'N/A'}

OpenAI Answer:
{openai_ans[0][0] if openai_ans else 'N/A'}

Please synthesize the final answer by combining the most relevant information.
"""
    try:
        model_instance = genai.GenerativeModel(model)
        response = model_instance.generate_content(combined)
        return response.text.strip()
    except Exception as e:
        print(f"⚠️ Ensemble synthesis failed: {e}")
        return gemini_ans[0][0] if gemini_ans else "No answer available."

# --- KB2 Hook ---
def check_knowledgebase2(query, threshold=0.7):
    query_lower = query.lower()
    best_match = get_close_matches(query_lower, kb2_fallbacks.keys(), n=1, cutoff=threshold)
    if best_match:
        return kb2_fallbacks[best_match[0]]
    return None

# --- Feedback Logger ---
LOG_FILE = "data/feedback_logs.csv"

def log_feedback(query, context_docs, response, user_feedback=None, kb2_used=False):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().isoformat(),
            query,
            " | ".join(context_docs),
            response,
            user_feedback or "",
            kb2_used
        ])

# --- Full Pipeline ---
def run_pipeline(query="What is the typical feedback on the loan application process across different apps?"):
    print("\n🔄 Running HelaRAG Pipeline with Gemini + OpenRouter (OpenAI)...\n")

    config = load_config()
    try:
        reviews_df = pd.read_csv(config['data']['reviews_path'])
        domain_documents = reviews_df['text'].tolist()
    except Exception as e:
        return f"❌ Error loading CSV: {e}", None, None

    if not domain_documents:
        return "❌ No reviews loaded.", None, None

    embeddings_path = config['data']['embeddings_path']
    faiss_index = load_faiss_index(embeddings_path) if os.path.exists(embeddings_path) else create_faiss_index(domain_documents, embeddings_path)
    helarag_model = load_helarag_model(config)

    print(f"🔍 User Query: {query}\n")

    faiss_results = retrieve_documents(query, domain_documents, config, faiss_index)
    helarag_results = helarag_retrieve(query, domain_documents, helarag_model, config)
    gemini_results = gemini_retrieve(query)
    openai_results = openai_retrieve(query)  # routed via OpenRouter

    negotiated_context_docs = negotiate_context(
        faiss_results, helarag_results,
        gemini_results=gemini_results, openai_results=openai_results
    )

    ensemble_answer = ensemble_synthesis(query, faiss_results, helarag_results, gemini_results, openai_results)

    kb2_response = check_knowledgebase2(query)
    final_output = kb2_response if kb2_response else ensemble_answer

    log_feedback(query, negotiated_context_docs, final_output, kb2_used=bool(kb2_response))
    return final_output, negotiated_context_docs, kb2_response

# --- CLI Entry ---
if __name__ == "__main__":
    final_output, context_docs, kb2_used = run_pipeline()
    print(f"\n✅ Final Output:\n{final_output}")
