from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from vector import initialize_vector_store
import pandas as pd
from difflib import get_close_matches

app = Flask(__name__)
CORS(app)

# Load CSV once
CSV_FILE = "facility_maintenance_safety_guidelines_realtime.csv"
try:
    df = pd.read_csv(CSV_FILE)
except FileNotFoundError:
    df = None
    print(f"❌ CSV file '{CSV_FILE}' not found. Exact/fuzzy matching disabled.")


# ---------- Matching Helpers ----------
def check_exact_match(user_message: str):
    """Check if the question matches exactly in CSV."""
    if df is not None:
        row = df[df["Question"].str.lower().str.strip() == user_message.lower().strip()]
        if not row.empty:
            return row.iloc[0]["Answer"]
    return None


def check_fuzzy_match(user_message: str, cutoff=0.7):
    """Check for approximate string match in CSV questions."""
    if df is not None:
        questions = df["Question"].str.lower().tolist()
        matches = get_close_matches(user_message.lower(), questions, n=1, cutoff=cutoff)
        if matches:
            matched_row = df[df["Question"].str.lower() == matches[0]]
            if not matched_row.empty:
                return matched_row.iloc[0]["Answer"]
    return None


# ---------- RAG Pipeline ----------
def initialize_rag_pipeline():
    try:
        # Load vector store & retriever
        vector_store = initialize_vector_store()
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.4}
        )

        # Prompt template
        prompt_template = ChatPromptTemplate.from_template("""
        You are an expert in answering questions about facility management.
        Use ONLY the following context to answer the question.
        If the answer is not in the context,
        you MUST say: "I cannot find the answer to that question in the provided documents."

        Context: {context}

        Question: {input}
        """)

        # Load LLM (choose a lightweight model for speed)
        llm = ChatOllama(model="gemma:2b")   # very fast, good for Q&A

        # Build chain
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return retrieval_chain

    except Exception as e:
        print(f"❌ Error initializing RAG pipeline: {e}")
        return None


# Initialize RAG pipeline once
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

    # 1. Exact Match
    answer = check_exact_match(user_message)
    if answer:
        return jsonify({"response": answer})

    # 2. Fuzzy Match
    fuzzy_answer = check_fuzzy_match(user_message)
    if fuzzy_answer:
        return jsonify({"response": fuzzy_answer})

    # 3. Semantic Match via RAG
    try:
        response = retrieval_chain.invoke({"input": user_message})
        return jsonify({"response": response["answer"]})
    except Exception as e:
        return jsonify({"response": f"❌ An error occurred: {e}"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
