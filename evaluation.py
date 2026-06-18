import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from src.load_docs import load_documents
from src.chunk_docs import chunk_documents
from src.vector_store import create_vector_store

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

docs = load_documents("data/documents/sample_ai_document_for_rag.pdf")
chunks = chunk_documents(docs)
retriever = create_vector_store(chunks)

answer_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, num_ctx=1024)

system_prompt = """
You are a PDF question-answering agent.
Answer ONLY using the provided PDF context.
If the tool results do not contain the answer, say:
"I cannot find the answer in the document."
"""

judge = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0,
)

test_questions = [
    "What is Artificial Intelligence?",
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What are Large Language Models?",
    "What is Retrieval Augmented Generation?",
]

def score_faithfulness(context, answer):
    prompt = f"""
You are an evaluator. Score how faithful the answer is to the context.
Faithful means the answer only uses information from the context.
Score from 0.0 to 1.0. Reply with ONLY a number like 0.85

Context: {context}
Answer: {answer}
Score:"""
    response = judge.invoke(prompt)
    try:
        return float(response.content.strip())
    except:
        return 0.0

def score_relevancy(question, answer):
    prompt = f"""
You are an evaluator. Score how relevant the answer is to the question.
Score from 0.0 to 1.0. Reply with ONLY a number like 0.85

Question: {question}
Answer: {answer}
Score:"""
    response = judge.invoke(prompt)
    try:
        return float(response.content.strip())
    except:
        return 0.0

print("\n Generating answers and scoring...\n")

faithfulness_scores = []
relevancy_scores = []

for question in test_questions:
    print(f"Q: {question}")

    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in retrieved_docs])

    response = answer_llm.invoke([
        ("system", system_prompt),
        ("user", f"PDF context:\n{context}\n\nQuestion: {question}")
    ])
    answer = response.content
    print(f"A: {answer}")

    f_score = score_faithfulness(context, answer)
    r_score = score_relevancy(question, answer)
    faithfulness_scores.append(f_score)
    relevancy_scores.append(r_score)
    print(f"Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}\n")


avg_faith = sum(faithfulness_scores) / len(faithfulness_scores)
avg_relev = sum(relevancy_scores) / len(relevancy_scores)

print("="*50)
print("       EVALUATION RESULTS")
print("="*50)
print(f"  Faithfulness:      {avg_faith:.2f}")
print(f"  Answer Relevancy:  {avg_relev:.2f}")
print("="*50)
print(f"\n Achieved Faithfulness: {avg_faith:.2f}, Relevancy: {avg_relev:.2f} via RAG evaluation'")