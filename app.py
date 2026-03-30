# app.py  (ONNX embeddings, Groq LLM, Chroma)
import streamlit as st
import chromadb
from groq import Groq
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import onnxruntime as ort
from transformers import AutoTokenizer
import math

# ---------------------------
# CONFIG — change only this if your ONNX path differs
# ---------------------------
ONNX_PATH = r"E:\RAG Project\models\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\c9745ed1d9f207416be6d2e6f8de32d1f16199bf\onnx\model.onnx"  # <- update if different
EMBED_DIM = 384  # expected dimension for MiniLM
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "rag_docs"

# ---------------------------
# LOAD ENV & CLIENTS
# ---------------------------
load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Chroma persistent client
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
except Exception:
    collection = chroma_client.create_collection(COLLECTION_NAME)

# ---------------------------
# LOAD ONNX EMBEDDING SESSION & TOKENIZER
# ---------------------------
if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(f"ONNX model not found at {ONNX_PATH}. Please download it and set ONNX_PATH accordingly.")

# Create ONNX session
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 1
ort_session = ort.InferenceSession(ONNX_PATH, sess_options, providers=['CPUExecutionProvider'])

# Load tokenizer (no torch)
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", use_fast=True)

# Helper: compute embeddings using ONNX session
def onnx_embed(texts):
    """
    texts: list[str]
    returns: list[list[float]] length = len(texts), each inner list length = EMBED_DIM
    """
    batch_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="np")
    # ONNX model input names may vary; inspect session inputs
    input_names = {inp.name: inp.name for inp in ort_session.get_inputs()}
    feed = {}
    # common names
    if "input_ids" in input_names:
        feed["input_ids"] = batch_inputs["input_ids"]
    if "attention_mask" in input_names:
        feed["attention_mask"] = batch_inputs["attention_mask"]
    # run
    outputs = ort_session.run(None, feed)
    # find the last_hidden_state output (most models return an array[batch, seq_len, hidden])
    # We will try to detect shape and then mean-pool tokens by attention_mask
    last_hidden = None
    for out in outputs:
        if out.ndim == 3:
            last_hidden = out
            break
    if last_hidden is None:
        # fallback: take first output and try to handle
        last_hidden = outputs[0]
    embeddings = []
    masks = batch_inputs["attention_mask"]
    for i in range(last_hidden.shape[0]):
        vecs = last_hidden[i]  # seq_len x hidden
        mask = masks[i].astype(bool)
        if mask.any():
            pooled = vecs[mask].mean(axis=0)
        else:
            pooled = vecs.mean(axis=0)
        # L2 normalize
        norm = (pooled ** 2).sum() ** 0.5
        if norm > 0:
            pooled = pooled / norm
        embeddings.append(pooled.tolist())
    return embeddings

# ---------------------------
# PDF -> text
# ---------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    pages = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            pages.append((i + 1, page_text))
            text += page_text + "\n"
    return text, pages

# ---------------------------
# chunking (simple but safe)
# ---------------------------
def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        start = max(end - overlap, end)
    return chunks

# ---------------------------
# insert into chroma: uses onnx_embed
# ---------------------------
def insert_chunks_to_chroma(chunks, filename):
    ids, docs, embs, metas = [], [], [], []
    # create embeddings in small batches
    B = 16
    for i in range(0, len(chunks), B):
        batch = chunks[i:i+B]
        batch_embs = onnx_embed(batch)
        for j, ch in enumerate(batch):
            idx = i + j
            ids.append(f"{filename}_chunk_{idx}")
            docs.append(ch)
            embs.append(batch_embs[j])
            metas.append({"source": filename, "chunk_index": idx})
    collection.add(ids=ids, documents=docs, embeddings=embs, metadatas=metas)

# ---------------------------
# RAG Query: embed question via ONNX, query chroma, call Groq LLM
# ---------------------------
def rag_query(question, k=3):
    q_emb = onnx_embed([question])[0]
    results = collection.query(query_embeddings=[q_emb], n_results=k)
    docs = results["documents"][0]
    metas = results.get("metadatas", [[]])[0] if "metadatas" in results else [{}]*len(docs)
    # Build context with safe metadata
    context_pieces = []
    for d, m in zip(docs, metas):
        m = m or {}
        context_pieces.append(f"[source:{m.get('source','N/A')} chunk:{m.get('chunk_index','N/A')}]\n{d}")
    context = "\n\n---\n\n".join(context_pieces)
    prompt = f"""Answer using ONLY the CONTEXT below. If not present, reply: 'The document does not contain this information.'\n\nCONTEXT:\n{context}\n\nQUESTION: {question}"""
    resp = client.chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role":"user","content":prompt}])
    return resp.choices[0].message.content

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.set_page_config(page_title="RAG (ONNX) Document Assistant", layout="wide")
st.title("RAG Document Assistant — ONNX embeddings (local)")

uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_pdf:
    st.success("PDF uploaded")

    # full reset: delete the collection and recreate to avoid duplicates
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(COLLECTION_NAME)

    text, pages = extract_text_from_pdf(uploaded_pdf)
    chunks = chunk_text(text)
    st.info(f"Creating {len(chunks)} embeddings locally (ONNX) — this may take a short while...")
    insert_chunks_to_chroma(chunks, uploaded_pdf.name)
    st.success("Ingest complete! You can now ask questions.")

st.header("Ask a question about uploaded document")
question = st.text_input("Question")
if st.button("Ask"):
    if not question.strip():
        st.error("Type a question")
    else:
        with st.spinner("Retrieving answer..."):
            ans = rag_query(question)
        st.subheader("Answer")
        st.write(ans)
        st.subheader("Context used (top results)")
        q_emb = onnx_embed([question])[0]
        results = collection.query(query_embeddings=[q_emb], n_results=3)
        docs = results["documents"][0]
        metas = results.get("metadatas", [[]])[0]
        for i, (d, m) in enumerate(zip(docs, metas)):
            m = m or {}
            st.markdown(f"**Result {i+1}** — {m.get('source','N/A')} chunk {m.get('chunk_index','N/A')}")
            st.write(d[:800])
