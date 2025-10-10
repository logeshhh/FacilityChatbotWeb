from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import initialize_vector_store
from langchain_huggingface import HuggingFaceEmbeddings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from uuid import uuid4

app = Flask(__name__)
CORS(app)

# ----------------- CSV -----------------
CSV_FILE = "facility_maintenance_safety_guidelines_realtime.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df = None
    print(f"⚠️ CSV file '{CSV_FILE}' not found. Exact/fuzzy matching disabled.")

# ----------------- Embeddings -----------------
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

question_embeddings = []
if df is not None:
    print("✅ Precomputing embeddings for CSV questions...")
    for _, row in df.iterrows():
        emb = embeddings_model.embed_query(row["Question"])
        question_embeddings.append((row["Question"], row["Answer"], emb))

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
        score = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding)*np.linalg.norm(emb))
        if score > best_score:
            best_score = score
            best_answer = a
    if best_score >= threshold:
        return best_answer
    return None

def normalize_question(q: str) -> str:
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

# ----------------- RAG -----------------
def initialize_rag_pipeline():
    try:
        vector_store = initialize_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 4, "score_threshold": 0.1},
        )

        prompt_template = ChatPromptTemplate.from_template("""
You are a helpful facility management and safety assistant.
Use ONLY the information in the provided context to answer the question.

Guidelines:
- If the context contains relevant details, rephrase and explain clearly.
- For layman questions, use simple language.
- For technical questions, give exact values (temperature, pressure, size).
- If no relevant information exists, reply: "I cannot find the answer in the provided documents."
- Do NOT guess or add outside info.

Context:
{context}

Question:
{input}

Answer:
""")

        llm = ChatOllama(
            model="mistral:7b",
            temperature=0.1,
            num_ctx=4096,
            num_predict=256
        )

        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        return retrieval_chain
    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        return None

retrieval_chain = initialize_rag_pipeline()

# ----------------- Chat Sessions -----------------
chat_sessions = []  # list of dicts: {id, created_at, messages: [{"user":..., "bot":...}]}

def create_new_chat():
    new_chat = {
        "id": str(uuid4()),
        "created_at": datetime.now(),
        "messages": []
    }
    chat_sessions.insert(0, new_chat)  # newest first
    cleanup_chats()
    return new_chat

def cleanup_chats():
    global chat_sessions
    cutoff = datetime.now() - timedelta(days=1)
    chat_sessions = [c for c in chat_sessions if c["created_at"] > cutoff]

def add_message(chat_id, user_msg, bot_resp):
    for chat in chat_sessions:
        if chat["id"] == chat_id:
            chat["messages"].append({"user": user_msg, "bot": bot_resp})
            break
    cleanup_chats()

# ----------------- Routes -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/new_chat", methods=["POST"])
def new_chat():
    chat = create_new_chat()
    return jsonify({"id": chat["id"]})

@app.route("/get_chats", methods=["GET"])
def get_chats():
    cleanup_chats()
    previews = []
    for chat in chat_sessions:
        first_question = chat["messages"][0]["user"] if chat["messages"] else ""
        previews.append({
            "id": chat["id"],
            "first_question": first_question,
            "created_at": chat["created_at"].isoformat()
        })
    return jsonify(previews)

@app.route("/chat/<chat_id>", methods=["POST"])
def chat_in_session(chat_id):
    user_message = request.json.get("message")
    if not user_message:
        return jsonify({"response": "No message received."})

    # Exact Match
    answer = check_exact_match(user_message)
    # Fuzzy Match
    if not answer:
        answer = semantic_fuzzy_match(user_message)
    # RAG
    if not answer:
        normalized_question = normalize_question(user_message)
        response = retrieval_chain.invoke({"input": normalized_question})
        answer = response["answer"] + " (📄 RAG)"

    add_message(chat_id, user_message, answer)
    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
