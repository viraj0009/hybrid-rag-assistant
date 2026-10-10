import streamlit as st
from langchain_ollama import ChatOllama
from src.chunk_docs import chunk_documents
from src.load_docs import load_documents
from src.vector_store import create_vector_store
import time

@st.cache_resource
def setup_pipeline():      
    docs = load_documents("data/documents/sample_ai_document_for_rag.pdf")
    chunks = chunk_documents(docs)
    retriever = create_vector_store(chunks)
    llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, num_ctx=1024)
    return retriever, llm, len(chunks)

retriever, llm, total_chunks = setup_pipeline()

system_prompt = """
You are a PDF question-answering agent.
Answer ONLY using the provided PDF context.
If the tool results do not contain the answer, say:
"I cannot find the answer in the document."
"""

st.title("Hybrid RAG Assistant")
st.caption("Hybrid Search (FAISS + BM25) | Local LLM | Thought Process Visualization")

st.sidebar.header("System Info")
st.sidebar.metric("Total Chunks Indexed", total_chunks)
st.sidebar.metric("Retrieval Mode", "Hybrid (FAISS + BM25)")
st.sidebar.metric("LLM", "Qwen2.5:1.5b")

query = st.text_input("Ask a question about the document:", placeholder="e.g. What is Retrieval Augmented Generation?")

if query:
    with st.expander("RAG Pipeline Execution", expanded=True):

        step1 = st.empty()
        step1.info("**Step 1 - Query Analysis**")
        time.sleep(0.3)
        step1.success(f"**Step 1 - Query Analysis** \n\nQuery received: *\"{query}\"*\nRouting to Hybrid Reriever... ")

        step2 = st.empty()
        step2.info("Step 2 - **Hybrid Retrieval (FASIS + BM25)**")

        t0 = time.time()
        context_items = retriever.invoke(query)
        retrieval_time = time.time() - t0

        step2.success(
            f"**Step 2 - Hybrid Retrieval** \n\n"
            f"Retrieved **{len(context_items)} chunks** in {retrieval_time: .2f}s \n\n"
            f"Strategy: BM25 (keyword) + FAISS (semantic) → merged with equal weights"
        )

        step3 = st.empty()
        context = "\n\n".join([doc.page_content for doc in context_items])
        token_estimate = len(context.split())
        step3.success(
            f"**Step 3 - Context Assembly** \n\n"
            f"Combined context: ~ **{token_estimate} tokens** passed to LLM"
        )

        step4 = st.empty()
        step4.info("**Step 4 - Generating Answer** \n\n")

        t1 = time.time()
        response = llm.invoke([
            ("system", system_prompt),
            ("user", f"PDF context:\n{context}\n\nQuestion: {query}"),
        ])

        answer = response.content
        gen_time = time.time() - t1

        step4.success(
            f"**Step 4 - Answer Generated** \n\n"
            f"Generation time: {gen_time: .2f}s"
        )

    st.markdown("### Answer")
    st.write(answer)

    if answer != "I cannot find the answer in the document.":
        st.markdown("### Retrieved Sources")
        for i, doc in enumerate(context_items):
            with st.expander(f"Chunk{i+1}"):
                st.write(doc.page_content)