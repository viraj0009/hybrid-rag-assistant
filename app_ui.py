import streamlit as st
from src.load_docs import load_documents
from src.chunk_docs import chunk_documents
from src.vector_store import create_vector_store
from src.llm import generate_answer

docs = load_documents("data/documents/sample_ai_document_for_rag.pdf")
chunks = chunk_documents(docs)
vectorstore = create_vector_store(chunks)

st.title("📄 RAG PDF Assistant")

query = st.text_input("Ask a question:")

if query:
    st.write("Processing....")

    results = vectorstore.similarity_search_with_score(query, k=2)

    relevant_docs = []

    for doc, score in results:
        if score < 2.5:
            relevant_docs.append(doc)

    if len(relevant_docs) == 0 or all(score > 1.0 for _, score in results):
        answer = "I cannot find the answer in the document." 
    else:
        context = "\n".join([doc.page_content[:300] for doc in relevant_docs[:2]])
        answer = generate_answer(context, query)


    st.write(answer)

    if answer != "I cannot find the answer in the document.":
        st.write("Source: ")

        for i, doc in enumerate(relevant_docs):
            st.write(f"Source{i+1}")
            st.write(doc.page_content[:300])