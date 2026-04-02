from src.load_docs import load_documents
from src.chunk_docs import chunk_documents
from src.vector_store import create_vector_store
from src.llm import generate_answer

docs = load_documents("data/documents/sample_ai_document_for_rag.pdf")

chunks = chunk_documents(docs)

vectorstore = create_vector_store(chunks)

while True:
    query = input("\nAsk question (or type 'exit'): ")
    
    if query.lower() == "exit":
        break

    results = vectorstore.similarity_search_with_score(query, k=3)

    relevant_docs = []

    for doc, score in results:
        if score < 1.0:
            relevant_docs.append(doc)

    if len(relevant_docs) == 0 or all(score > 1.0 for _, score in results):
        answer = "I cannot find the answer in the document." 
    else:
        context = "\n".join([doc.page_content[:300] for doc in relevant_docs[:2]])
        answer = generate_answer(context, query)

    print("\nAnswer:\n")
    print(answer)

    if len(relevant_docs) > 0:
        print("\nSource Chunks:\n")
        for i, doc in enumerate(relevant_docs):
            print(f"Source {i+1}")
            print(doc.page_content[:150])
            print("-"*30)