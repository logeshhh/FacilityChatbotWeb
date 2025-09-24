from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever  # Make sure retriever is defined in vector.py

# Initialize your LLM
model = OllamaLLM(model="llama3.2")

# Define the prompt template
template = """
You are an expert in answering questions about facility management queries.

Here are some relevant Q&A entries: {reviews}

Here is the question to answer: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Chain the prompt with the LLM
chain = prompt | model

# Start interactive loop
while True:
    print("\n\n-------------------------------")
    question = input("Ask your question (q to quit): ")
    
    if question.lower() == "q":
        break

    # Use retriever to get top relevant documents
    docs = retriever.get_relevant_documents(question)  # LangChain retriever method
    reviews_text = "\n".join([doc.page_content for doc in docs])  # Combine content

    # Invoke LLM chain
    result = chain.invoke({"reviews": reviews_text, "question": question})
    print("\nAnswer:\n", result)
