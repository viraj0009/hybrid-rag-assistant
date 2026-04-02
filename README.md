# 📄 RAG PDF Assistant

This project is a **local Retrieval-Augmented Generation (RAG) system** that allows users to ask questions about a PDF and get answers using a local LLM.

The system retrieves relevant content from the document and generates answers based only on that context.

---

## 🚀 What This Project Does

* Takes a PDF as input
* Splits it into smaller chunks
* Converts text into embeddings
* Stores embeddings using FAISS
* Retrieves relevant chunks for a question
* Uses a local LLM (TinyLlama) to generate answers
* Displays the answer along with source chunks

---

## 🛠 Tech Stack

* Python
* LangChain
* FAISS
* Sentence Transformers (MiniLM)
* Ollama (TinyLlama)
* Streamlit

---

## ▶️ How to Run

**1. Activate environment**

```
venv\Scripts\activate
```

**2. Run terminal version**

```
python app.py
```

**3. Run Streamlit UI**

```
streamlit run app_ui.py
```

---

## 📷 Demo
```
![App Screenshot](screenshot.png)

```
---

## ⚠️ Notes

* This project uses a small local model, so responses may not always be perfect
* Sometimes irrelevant questions may still produce incorrect answers
* The focus of this project is on building a working RAG pipeline

---

## 🎯 What I Learned

* How RAG systems work end-to-end
* How embeddings and vector databases are used
* How to control LLM hallucination using context
* How to integrate a local LLM with real data

---

## 📌 Future Improvements

* Improve prompt quality
* Add support for multiple documents
* Use a stronger model for better responses
* Improve UI and user experience

---

## 👨‍💻 About Me

I am transitioning from a mechanical engineering background into AI/ML, focusing on Generative AI and building practical projects like this.