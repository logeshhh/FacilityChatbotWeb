from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_ollama import ChatOllama   # ✅ Updated import
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import initialize_vector_store
from langchain_ollama import OllamaEmbeddings   # ✅ Keep embeddings import updated
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# ----------------- Load CSV -----------------
CSV_FILE = "facility_maintenance_safety_guidelines_realtime.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df = None
    print(f"⚠️ CSV file '{CSV_FILE}' not found. Exact/fuzzy matching disabled.")

# ----------------- Embeddings for fuzzy matching -----------------
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Precompute embeddings for CSV Q&A
question_embeddings = []
if df is not None:
    print("✅ Precomputing embeddings for CSV questions...")
    for _, row in df.iterrows():
        emb = embeddings_model.embed_query(row["Question"])
        question_embeddings.append((row["Question"], row["Answer"], emb))


# ----------------- CSV Matching Helpers -----------------
def check_exact_match(user_message: str):
    if df is not None:
        row = df[df["Question"].str.lower().str.strip() == user_message.lower().strip()]
        if not row.empty:
            return row.iloc[0]["Answer"]
    return None


def semantic_fuzzy_match(user_message: str, threshold=0.8):
    if not question_embeddings:
        return None
    query_embedding = embeddings_model.embed_query(user_message)
    best_score = 0
    best_answer = None
    for q, a, emb in question_embeddings:
        score = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        if score > best_score:
            best_score = score
            best_answer = a
    if best_score >= threshold:
        return best_answer
    return None


# ----------------- Question Normalization Helper -----------------
def normalize_question(q: str) -> str:
    """Normalize layman-style questions into retrieval-friendly forms."""
    replacements = {
        "know when to": "how does it",
        "why do": "what is the reason",
        "when does": "under what condition does",
        "how come": "why does",
        "does it know": "how does it detect",
    }
    q_norm = q.lower()
    for k, v in replacements.items():
        q_norm = q_norm.replace(k, v)
    return q_norm


# ----------------- RAG Pipeline -----------------
def initialize_rag_pipeline():
    try:
        vector_store = initialize_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.1},  # ✅ Lower threshold, more robust
        )

        prompt_template = ChatPromptTemplate.from_template("""
You are a helpful facility management and safety assistant. 
Use ONLY the information in the provided context to answer the question. 

Guidelines:
- If the context contains relevant details (even if wording is slightly different), rephrase and explain clearly. 
- For layman questions, use simple language. 
- For technical questions, give exact values (temperature, pressure, size) as shown. 
- If the context has absolutely no relevant information, then and only then reply: 
  "I cannot find the answer in the provided documents." 
- Do NOT guess or add outside info. Stay grounded in the context. 

Context:
{context}

Question:
{input}

Answer:
""")

        llm = ChatOllama(model="mistral:7b-instruct-q2_K", temperature=0.1)
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        return None


retrieval_chain = initialize_rag_pipeline()


# ----------------- Flask Routes -----------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    if not retrieval_chain:
        return jsonify({"response": "Chatbot is not ready. Please check the server logs."})

    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "No message received."})

    # 1. Exact Match (CSV)
    answer = check_exact_match(user_message)
    if answer:
        return jsonify({"response": answer })

    # 2. Fuzzy Match (CSV)
    fuzzy_answer = semantic_fuzzy_match(user_message, threshold=0.8)
    if fuzzy_answer:
        return jsonify({"response": fuzzy_answer })

    # 3. RAG (PDFs + CSV)
    try:
        # 🔧 Normalize layman question before retrieval
        normalized_question = normalize_question(user_message)

        response = retrieval_chain.invoke({"input": normalized_question})

        # Debug: show retrieved context
        print("🔎 Retrieved context:")
        if "context" in response:
            print(response["context"])
        else:
            print("No context returned")

        return jsonify({"response": response["answer"] + " (📄 RAG: CSV+PDF)"})
    except Exception as e:
        return jsonify({"response": f"❌ An error occurred: {e}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
