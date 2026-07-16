from src.load_docs import load_documents
from src.chunk_docs import chunk_documents
from src.vector_store import create_vector_store
from langchain_ollama import ChatOllama

docs = load_documents("data/documents/sample_ai_document_for_rag.pdf")

chunks = chunk_documents(docs)

retriever = create_vector_store(chunks)
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, num_ctx=1024)

system_prompt = """
You are a PDF question-answering agent.
Answer ONLY using the provided PDF context.
If the tool results do not contain the answer, say:
"I cannot find the answer in the document."
"""

def get_rag_response(question):
    docs_found = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs_found])

    response = llm.invoke([
        ("system", system_prompt),
        ("user", f"PDF Context\n{context}\n\nQuestion: {question}")
    ])

    return response.content, [d.page_content for d in docs_found]

while True:
    user_input = input("\nAsk question (or 'exit'): ")

    if user_input.lower() == "exit":
        break

    answer, chunks = get_rag_response(user_input)
    print(answer)