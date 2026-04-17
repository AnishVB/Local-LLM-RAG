import os
from typing import Iterable

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings


load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "gemma:2b")
DB_PATH = os.getenv("DB_PATH", "./chroma_db")
COLLECTION_NAME = "local_rag_documents"


def _get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=MODEL_NAME)


def _get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=DB_PATH,
        embedding_function=_get_embeddings(),
    )


def _format_docs(documents: Iterable[Document]) -> str:
    return "\n\n".join(document.page_content for document in documents)


def add_documents(documents: list[Document]) -> int:
    if not documents:
        return 0

    vectorstore = _get_vectorstore()
    vectorstore.add_documents(documents)
    return len(documents)


def query_llm(user_input: str) -> str:
    vectorstore = _get_vectorstore()

    try:
        document_count = vectorstore._collection.count()
    except Exception:
        document_count = 0

    if document_count == 0:
        return "No indexed documents yet. Upload a PDF first."

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    related_documents = retriever.invoke(user_input)
    context = _format_docs(related_documents)

    prompt = (
        "You are a local, private RAG assistant. Answer only from the provided context. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {user_input}\n"
        "Answer:"
    )

    llm = ChatOllama(model=MODEL_NAME, temperature=0.2)
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response)).strip()