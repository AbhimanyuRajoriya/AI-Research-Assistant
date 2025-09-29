import streamlit as st
import numpy as np
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import PyPDF2

def pdf_to_text(pdf_bytes):
    return "\n".join([p.extract_text() or "" for p in PyPDF2.PdfReader(BytesIO(pdf_bytes)).pages])

def chunk_text(text, size=400, overlap=80):
    words = text.split()
    if len(words) <= size: return [text]
    return [" ".join(words[i:i+size]) for i in range(0, len(words), size - overlap)]

def embed(model, texts): return model.encode(texts, convert_to_numpy=True)

def top_k(query_emb, chunk_embs, k=3):
    sims = cosine_similarity([query_emb], chunk_embs)[0]
    idx = np.argsort(sims)[::-1][:k]
    return [idx, sims[idx]]

@st.cache_resource
def load_models():
    return {
        "embedder": SentenceTransformer("all-MiniLM-L6-v2"),
        "summarizer": pipeline("summarization", model="google/pegasus-xsum"),
        "qa": pipeline("question-answering", model="distilbert-base-cased-distilled-squad"),
    }

st.set_page_config(page_title="AI Research Assistant", layout="wide")

st.markdown("""
    <style>
    body {
        background-color: #f9f9f9;
    }
    .main-title {
        text-align: center;
        font-size: 60px !important;
        color: #2E86C1;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 25px !important;
        color: gray;
        margin-bottom: 30px;
    }
    .stButton button {
        border-radius: 30px;
        padding: 0.6em 1.2em;
        font-weight: bold;
        background: linear-gradient(90deg, #2E86C1, #5DADE2);
        color: white;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">AI Research Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Summarize • Analyze • Ask Questions — directly from your PDF</p>', unsafe_allow_html=True)

file = st.file_uploader("Upload your PDF", type="pdf")

if not file: 
    st.stop()

text = pdf_to_text(file.read())
if not text.strip():
    st.error("No text found in PDF.")
    st.stop()

models = load_models()
chunks = chunk_text(text)
chunk_embs = embed(models["embedder"], chunks)

tab1, tab3 = st.tabs(["Summarization", "Q&A"])

with tab1:
    st.subheader("Document Summary")
    if st.button("Generate Summary", use_container_width=True):
        input_text = " ".join(chunks[:10])
        summary = models["summarizer"](input_text, max_length=150, min_length=100, truncation=True)[0]["summary_text"]
        st.markdown(summary, unsafe_allow_html=True)

with tab3:
    st.subheader("Ask a Question")
    q = st.text_input("Type your question here:")
    k = st.slider("Context Size (chunks)", 1, 6, 3)

    if st.button("Get Answer", use_container_width=True) and q:
        q_emb = models["embedder"].encode(q, convert_to_numpy=True)
        idxs, _ = top_k(q_emb, chunk_embs, k)
        ctx = "\n".join([chunks[i] for i in idxs])[:3000]
        ans = models["qa"](question=q, context=ctx)
        st.markdown(ans["answer"], unsafe_allow_html=True)