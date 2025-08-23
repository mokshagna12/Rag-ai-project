# Local RAG Q&A Bot (Ollama + LangChain + Chroma)

A fully offline Retrieval-Augmented Generation (RAG) chatbot that answers questions from your documents.
Runs with Ollama locally and uses a local vector store (ChromaDB).

> Built for the internship task: ingestion, retrieval, cited answers, latency reporting, dynamic uploads, and both UI (Streamlit) and CLI.

## Prerequisites

- Python 3.10+
- Ollama installed and running: `ollama serve`
- Pull an LLM: `ollama pull mistral` (or llama3 / gemma)
- (Optional) Embeddings via Ollama: `ollama pull nomic-embed-text` or `ollama pull all-minilm`

Default embeddings use SentenceTransformers, so Ollama embeddings are optional.

## Setup

```bash
cd local_rag_ollama
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run

1) Ingest docs from `data/samples` (or pass a path):
```bash
python ingest.py --doc-path data/samples
```

2) Start UI:
```bash
streamlit run app_streamlit.py
```

3) Or CLI:
```bash
python app_cli.py
```

## Env Vars

- `OLLAMA_LLM` (default: `mistral`) (gemma:2b or llama3)
- `EMBEDDINGS_BACKEND` = `sentence` or `ollama` (default: `sentence`)
- `OLLAMA_EMBED_MODEL` (when EMBEDDINGS_BACKEND=ollama, e.g., `nomic-embed-text`)
- `VECTOR_DIR` (default: `data/chroma`)
- `CHUNK_SIZE` (default: 800)
- `CHUNK_OVERLAP` (default: 150)
- `TOP_K` (default: 4)
