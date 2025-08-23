import os
from typing import List, Dict, Any, Tuple, Generator
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from prompts import SYSTEM_PROMPT, USER_PROMPT

VECTOR_DIR = os.getenv("VECTOR_DIR", "data/chroma")
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "sentence").lower()
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM = os.getenv("OLLAMA_LLM", "mistral")

COLLECTION_NAME = "local_rag_docs"   # ✅ same as ingest.py

def get_embeddings():
    if EMBEDDINGS_BACKEND == "ollama":
        return OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def get_vectorstore(persist_directory: str = VECTOR_DIR):
    embeddings = get_embeddings()
    vs = Chroma(
        persist_directory=persist_directory,
        collection_name=COLLECTION_NAME,   # ✅ ensures we always query same collection
        embedding_function=embeddings
    )
    return vs

def build_retriever(k: int = 4):
    vs = get_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})

def format_context(docs: List[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", "1")
        parts.append(f"[{src}:{page}] {d.page_content}")
    return "\n\n".join(parts)

def _final_prompt(question: str, context: str) -> str:
    prompt = PromptTemplate.from_template(USER_PROMPT)
    user_filled = prompt.format(question=question, context=context)
    return f"{SYSTEM_PROMPT}\n\n{user_filled}"

def prepare_prompt(question: str, k: int = 4) -> Tuple[str, List[Dict[str, Any]], List[Document]]:
    retriever = build_retriever(k=k)
    docs = retriever.get_relevant_documents(question)
    context = format_context(docs) if docs else ""
    final_prompt = _final_prompt(question, context)
    citations = [{"source": d.metadata.get("source", "?"), "page": d.metadata.get("page", 1)} for d in docs]
    return final_prompt, citations, docs

def generate_answer(question: str, k: int = 4, temperature: float = 0.2) -> Dict[str, Any]:
    final_prompt, citations, docs = prepare_prompt(question, k=k)
    llm = Ollama(model=OLLAMA_LLM, temperature=temperature)
    answer_text = llm.invoke(final_prompt)
    return {"answer": answer_text, "citations": citations, "docs": docs}

def stream_answer(
    question: str,
    k: int = 4,
    temperature: float = 0.2,
) -> Tuple[Generator[str, None, None], Dict[str, Any]]:
    final_prompt, citations, docs = prepare_prompt(question, k=k)
    llm = Ollama(model=OLLAMA_LLM, temperature=temperature)

    def gen() -> Generator[str, None, None]:
        for chunk in llm.stream(final_prompt):
            yield chunk

    meta = {"citations": citations, "docs": docs}
    return gen(), meta
