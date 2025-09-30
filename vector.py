from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pandas as pd
import shutil


def load_csv_documents(csv_file="facility_maintenance_safety_guidelines_realtime.csv"):
    docs = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            docs.append(
                Document(
                    page_content=f"Q: {row['Question']} A: {row['Answer']}",
                    metadata={"source": "csv"}
                )
            )
    else:
        print(f"⚠️ CSV file '{csv_file}' not found. Skipping CSV import.")
    return docs


def load_pdf_documents(pdf_folder="pdfs"):
    docs = []
    if not os.path.exists(pdf_folder):
        print(f"⚠️ PDF folder '{pdf_folder}' not found. Skipping PDF import.")
        return docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)

    for file in os.listdir(pdf_folder):
        if file.endswith(".pdf"):
            try:
                loader = PyPDFLoader(os.path.join(pdf_folder, file))
                pages = loader.load()
                chunks = splitter.split_documents(pages)

                # 🧹 Clean chunk text: remove extra newlines/spaces
                for c in chunks:
                    c.page_content = " ".join(c.page_content.split())
                    c.metadata["source"] = file

                docs.extend(chunks)
                print(f"📄 Loaded {len(chunks)} cleaned chunks from {file}")
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    return docs


def initialize_vector_store():
    db_location = "./chroma_langchain_db"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")  # 768-dim, lightweight

    # If DB already exists, try to load it
    if os.path.exists(db_location):
        try:
            print("✅ Chroma DB found. Loading existing database...")
            vector_store = Chroma(
                collection_name="qa_queries",
                persist_directory=db_location,
                embedding_function=embeddings
            )
            return vector_store
        except Exception as e:
            print(f"⚠️ Error loading DB, rebuilding: {e}")
            shutil.rmtree(db_location)

    # Build new DB
    print("📌 Building Chroma DB from CSV + PDFs...")
    all_docs = []
    all_docs.extend(load_csv_documents())
    all_docs.extend(load_pdf_documents("pdfs"))

    if not all_docs:
        raise ValueError("❌ No documents found to index. Add CSV or PDFs.")

    vector_store = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=db_location,
        collection_name="qa_queries"
    )

    print(f"✅ Indexed {len(all_docs)} documents in Chroma DB.")
    return vector_store


if __name__ == "__main__":
    vs = initialize_vector_store()
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    print("✅ Vector store initialized and retriever ready.")
