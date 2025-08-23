import os
import streamlit as st
from ingest import ingest_path
from rag import generate_answer, get_vectorstore, VECTOR_DIR
from utils import now_ms, timer_step

st.set_page_config(page_title="Local RAG (Ollama + LangChain)", layout="wide")
st.title("📚 Local RAG Q&A — Ollama + LangChain + Chroma")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Settings")
    llm_name = st.selectbox(
        "Choose LLM",
        ["mistral", "llama3.2:3b", "gemma:2b"],
        index=1  # default llama3.2:3b since it's fastest on CPU
    )
    emb_backend = st.selectbox("Embeddings backend", ["sentence", "ollama"], index=0)
    emb_model = st.text_input("OLLAMA_EMBED_MODEL", os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    top_k = st.slider("TOP_K (retriever)", 1, 12, int(os.getenv("TOP_K", "3")))
    chunk_size = st.number_input("CHUNK_SIZE", value=int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap = st.number_input("CHUNK_OVERLAP", value=int(os.getenv("CHUNK_OVERLAP", "100")))

    if st.button("Apply env"):
        os.environ["OLLAMA_LLM"] = llm_name
        os.environ["EMBEDDINGS_BACKEND"] = emb_backend
        os.environ["OLLAMA_EMBED_MODEL"] = emb_model
        os.environ["CHUNK_SIZE"] = str(chunk_size)
        os.environ["CHUNK_OVERLAP"] = str(chunk_overlap)
        st.success(f"Environment updated — model: {llm_name}")

st.write("**Vector store path:**", VECTOR_DIR)

# ---------------- Upload & Ingest ----------------
st.subheader("Upload and Ingest Documents")
uploaded_files = st.file_uploader(
    "Upload PDFs/TXT/MD", type=["pdf", "txt", "md", "markdown"], accept_multiple_files=True
)

if st.button("Ingest Uploaded Files"):
    if not uploaded_files:
        st.warning("Please upload at least one file.")
    else:
        os.makedirs("data/samples", exist_ok=True)
        tmp_paths = []
        for uf in uploaded_files:
            path = os.path.join("data/samples", uf.name)
            with open(path, "wb") as f:
                f.write(uf.getbuffer())
            tmp_paths.append(path)

        t0 = now_ms()
        for p in tmp_paths:
            ingest_path(p)

        # ✅ force reload vectorstore so new docs are usable immediately
        vs = get_vectorstore()
        st.info(f"Vector store now contains {vs._collection.count()} chunks.")

        t_total = timer_step(t0)
        st.success(f"Ingested {len(tmp_paths)} files in {t_total} ms.")

st.divider()

# ---------------- Ask Question ----------------
st.subheader("Ask a Question")
q = st.text_input("Your question")

if st.button("Ask") and q.strip():
    t0 = now_ms()
    out = generate_answer(q, k=top_k)  # ✅ always uses fresh retriever
    t_total = timer_step(t0)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Answer")
        st.markdown(out["answer"])
        st.caption(f"Total latency: {t_total} ms")

    with col2:
        st.markdown("### Cited Chunks")
        for c in out["citations"]:
            st.code(f'{c["source"]}:{c["page"]}')

st.caption("⚡ Tip: If answers are weak, increase TOP_K or reduce CHUNK_SIZE.")
