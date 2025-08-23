import os
import argparse
from langchain_community.vectorstores import Chroma
from utils import load_documents, chunk_documents, now_ms, timer_step
from rag import get_embeddings, VECTOR_DIR

COLLECTION_NAME = "local_rag_docs"   # ✅ fixed collection name for all docs

def ingest_path(doc_path: str) -> int:
    t0 = now_ms()
    docs = load_documents(doc_path)
    t_load = timer_step(t0)

    chunks = chunk_documents(docs)
    t_chunk = timer_step(t0) - t_load

    embeddings = get_embeddings()
    # ✅ open existing collection instead of overwriting
    vs = Chroma(
        persist_directory=VECTOR_DIR,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    vs.add_documents(chunks)   # ✅ append new docs
    vs.persist()

    t_total = timer_step(t0)
    print(f"Loaded {len(docs)} docs; Chunked into {len(chunks)} chunks.")
    print(f"Timings: load={t_load}ms, chunk={t_chunk}ms, total={t_total}ms")
    return len(chunks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-path", type=str, default="data/samples", help="File or directory containing PDFs/TXT/MD")
    args = parser.parse_args()
    os.makedirs("data/chroma", exist_ok=True)
    os.makedirs("data/samples", exist_ok=True)
    ingest_path(args.doc_path)
