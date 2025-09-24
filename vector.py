from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

def initialize_vector_store():
    """
    Initializes or loads a Chroma vector store with embeddings.
    Reads data from 'facility_maintenance_safety_guidelines_realtime.csv'.
    """
    db_location = "./chroma_langchain_db"

    # Load CSV dataset
    try:
        df = pd.read_csv("facility_maintenance_safety_guidelines_realtime.csv")
    except FileNotFoundError:
        raise FileNotFoundError("❌ CSV file 'facility_maintenance_safety_guidelines_realtime.csv' not found.")

    # Initialize embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # fast + efficient

    # Check if DB exists
    add_documents = not os.path.exists(db_location)

    if add_documents:
        print("📌 Chroma DB not found. Creating new one and adding documents...")
        documents = []
        ids = []

        for i, row in df.iterrows():
            document = Document(
                page_content=f"Q: {row['Question']} A: {row['Answer']}",
                metadata={"question": row["Question"]},
            )
            documents.append(document)
            ids.append(str(i))

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=db_location,
            collection_name="qa_queries",
            ids=ids
        )
    else:
        print("✅ Chroma DB found. Loading existing database...")
        vector_store = Chroma(
            collection_name="qa_queries",
            persist_directory=db_location,
            embedding_function=embeddings
        )

    return vector_store

# Debug run
if __name__ == "__main__":
    vs = initialize_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 2})
    print("✅ Vector store initialized and retriever ready.")
