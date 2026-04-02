from langchain_community.llms import Ollama

llm = Ollama(
    model="tinyllama",
    temperature=0
)

def generate_answer(context, question):
    prompt = f"""
Answer ONLY using the context.

Give a clear, direct definition.
Do NOT say 'Sure', 'Here is answer', or anything extra.

If not found, say:
"I cannot find the answer in the document."

Context:
{context}

Question:
{question}

Answer:
"""
    
    response = llm.invoke(prompt)

    return response