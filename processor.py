from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path: str | Path) -> list[Document]:
    loader = PyPDFLoader(str(pdf_path))
    return loader.load()


def split_text(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


def process_pdf(pdf_path: str | Path) -> list[Document]:
    documents = load_pdf(pdf_path)
    return split_text(documents)