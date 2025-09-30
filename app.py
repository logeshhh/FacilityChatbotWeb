from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import initialize_vector_store
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load CSV once
CSV_FILE = "facility_maintenance_safety_guidelines_realtime.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df = None
    print(f"⚠️ CSV file '{CSV_FILE}' not found. Exact/fuzzy matching disabled.")

# Embeddings for fuzzy matching
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

# Precompute embeddings for all CSV questions
question_embeddings = []
if df is not None:
    print("✅ Precomputing embeddings for CSV questions...")
    for _, row in df.iterrows():
        emb = embeddings_model.embed_query(row["Question"])
        question_embeddings.append((row["Question"], row["Answer"], emb))


# ---------- CSV Matching Helpers ----------
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


# ---------- RAG Pipeline ----------
def initialize_rag_pipeline():
    try:
        vector_store = initialize_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.2}  # relaxed threshold
        )

        prompt_template = ChatPromptTemplate.from_template("""
        You are an expert in facility management and safety manuals.

        Use ONLY the provided context to answer the question.

        - If the answer includes technical values (like pressure, temperature, size), extract them exactly as shown.
        - If multiple values are present, list them clearly.
        - If the answer truly is not present, reply: "I cannot find the answer in the provided documents."

        Context: {context}

        Question: {input}
        """)


        llm = ChatOllama(model="gemma:2b")

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        return None


retrieval_chain = initialize_rag_pipeline()


# ---------- Flask Routes ----------
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
        return jsonify({"response": answer + " (✅ CSV Exact)"})

    # 2. Fuzzy Match (CSV)
    fuzzy_answer = semantic_fuzzy_match(user_message, threshold=0.8)
    if fuzzy_answer:
        return jsonify({"response": fuzzy_answer + " (✅ CSV Fuzzy)"})

    # 3. RAG (CSV + PDFs)
    try:
        response = retrieval_chain.invoke({"input": user_message})

        # 🔎 Debug: print retrieved context in logs
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
