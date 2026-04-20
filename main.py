from __future__ import annotations

import argparse
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings, OllamaLLM

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "gemma4")
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv"}
COLLECTION_NAME = "bdl_docs"


def _seed_data_if_empty() -> None:
	DATA_DIR.mkdir(parents=True, exist_ok=True)
	has_files = any(
		path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
		for path in DATA_DIR.iterdir()
	)
	if has_files:
		return

	seed_file = DATA_DIR / "bdl_overview.txt"
	seed_file.write_text(
		"BDL is an organization focused on data and business intelligence solutions.\n"
		"This is starter content for local RAG testing. Replace with real files in data/.\n",
		encoding="utf-8",
	)


def _load_documents() -> list[Document]:
	docs: list[Document] = []

	for path in sorted(DATA_DIR.iterdir()):
		if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
			continue

		text = path.read_text(encoding="utf-8", errors="ignore").strip()
		if not text:
			continue

		docs.append(
			Document(
				page_content=text,
				metadata={"source": path.name},
			)
		)

	return docs


def _chunk_documents(documents: list[Document], chunk_size: int = 800, overlap: int = 120) -> list[Document]:
	chunks: list[Document] = []

	for doc in documents:
		text = doc.page_content
		start = 0
		while start < len(text):
			end = min(start + chunk_size, len(text))
			chunk_text = text[start:end]
			chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
			if end == len(text):
				break
			start = max(0, end - overlap)

	return chunks


def _build_vector_store(reindex: bool) -> Chroma:
	docs = _load_documents()
	if not docs:
		raise RuntimeError("No readable files found in data/. Add .txt, .md, or .csv files.")

	chunks = _chunk_documents(docs)
	embeddings = OllamaEmbeddings(model=EMBED_MODEL)

	if reindex:
		try:
			existing_store = Chroma(
				collection_name=COLLECTION_NAME,
				persist_directory=str(CHROMA_DIR),
				embedding_function=embeddings,
			)
			existing_store.delete_collection()
		except Exception:
			pass

	try:
		vector_store = Chroma.from_documents(
			documents=chunks,
			embedding=embeddings,
			collection_name=COLLECTION_NAME,
			persist_directory=str(CHROMA_DIR),
		)
	except Exception as exc:
		raise RuntimeError(
			"Failed to build embeddings. Pull an embeddings model first, for example: "
			"ollama pull nomic-embed-text"
		) from exc

	vector_store.persist()
	return vector_store


def _answer_question(vector_store: Chroma, question: str) -> str:
	retriever = vector_store.as_retriever(search_kwargs={"k": 3})
	context_docs = retriever.invoke(question)

	context = "\n\n".join(
		f"Source: {doc.metadata.get('source', 'unknown')}\n{doc.page_content}"
		for doc in context_docs
	)

	prompt = ChatPromptTemplate.from_template(
		"""
You are a concise support assistant for BDL.
Use only the retrieved context to answer.
If the answer is not in the context, clearly say you do not know.

Question: {question}

Retrieved Context:
{context}
"""
	)

	llm = OllamaLLM(model=CHAT_MODEL)
	chain = prompt | llm
	return chain.invoke({"question": question, "context": context})


def main() -> None:
	parser = argparse.ArgumentParser(description="Local Ollama + Chroma RAG demo")
	parser.add_argument(
		"--question",
		type=str,
		default="What is BDL?",
		help="Question to ask the local RAG pipeline",
	)
	parser.add_argument(
		"--reindex",
		action="store_true",
		help="Rebuild chroma_db from files in data/",
	)
	args = parser.parse_args()

	_seed_data_if_empty()
	vector_store = _build_vector_store(reindex=args.reindex)
	answer = _answer_question(vector_store, args.question)

	print("\nQuestion:")
	print(args.question)
	print("\nAnswer:")
	print(answer)


if __name__ == "__main__":
	main()
