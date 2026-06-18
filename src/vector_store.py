from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

def create_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.5, 0.5]
    )

    return ensemble_retriever