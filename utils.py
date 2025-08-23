import os
import time
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document
import pathlib

DEFAULT_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

def now_ms() -> int:
    return int(time.perf_counter() * 1000)

def timer_step(start_ms: int) -> int:
    return now_ms() - start_ms

def load_documents(path: str) -> List[Document]:
    """Load PDFs, TXT, MD from a path (file or directory)."""
    p = pathlib.Path(path)
    paths = []
    if p.is_file():
        paths = [p]
    else:
        for ext in ["**/*.pdf", "**/*.txt", "**/*.md", "**/*.markdown"]:
            paths.extend(p.glob(ext))

    docs: List[Document] = []
    for fp in paths:
        suffix = fp.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(fp))
            pages = loader.load()
            for i, d in enumerate(pages):
                meta = d.metadata or {}
                meta["source"] = fp.name
                meta["page"] = meta.get("page", i + 1)
                d.metadata = meta
            docs.extend(pages)
        else:
            loader = TextLoader(str(fp), encoding="utf-8")
            loaded = loader.load()
            for d in loaded:
                meta = d.metadata or {}
                meta["source"] = fp.name
                meta["page"] = 1
                d.metadata = meta
                docs.append(d)
    return docs

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    return chunks
