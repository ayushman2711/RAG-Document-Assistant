# RAG Document Assistant using Groq + ONNX

##  Overview

This project is a Retrieval-Augmented Generation (RAG) based document assistant that allows users to query PDF documents using natural language.

It combines **semantic search** with **LLM reasoning** to provide accurate, context-aware answers from unstructured documents.

---

##  Key Features

*  Upload and process PDF documents
*  Semantic search using vector embeddings
*  Fast inference using Groq LLM
*  Local embedding generation using ONNX (no PyTorch dependency)
*  Context-aware answers based only on document content

---

##  Architecture

User Query
↓
Embedding (ONNX - MiniLM)
↓
Vector Search (ChromaDB)
↓
Top-K Relevant Chunks
↓
Groq LLM (LLaMA 3.1)
↓
Final Answer

---

##  Tech Stack

* Python
* Streamlit (UI)
* ChromaDB (Vector Database)
* Groq API (LLM inference)
* ONNX Runtime (Embedding optimization)
* Transformers (Tokenizer)
* PyPDF2 (PDF parsing)

---

## 📂 Project Structure

```
rag-document-assistant/
│
├── app.py
├── rag_pipeline.ipynb
├── requirements.txt
├── .gitignore
├── README.md
├── sample_data/
│   └── sample.pdf
```

---

##  How to Run

### 1. Clone the repository

```
git clone https://github.com/your-username/rag-document-assistant.git
cd rag-document-assistant
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Set environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

---

### 4. Run the app

```
streamlit run app.py
```

---

##  Example Queries

* "Summarize this document"
* "What are the key insights?"
* "Explain a specific section in simple terms"

---

##  Optimization Highlight

Instead of using traditional PyTorch-based embeddings, this project uses **ONNX Runtime** for faster and lightweight embedding generation, reducing latency and improving performance.

---

##  Future Improvements

* Support multiple documents
* Add hybrid search (keyword + semantic)
* Deploy as a web application
* Add chat history memory

---

##  Author

Ayushman Sharma
📧 [sharmaayushman296@gmail.com](mailto:sharmaayushman296@gmail.com)
🔗 https://www.linkedin.com/in/ayushman-sharma-ayushmantech/
