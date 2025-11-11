from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


def ensure_index_directory(directory: Path, reset: bool = False) -> None:
    if reset and directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True, exist_ok=True)


def load_or_create_faiss_index(
    directory: Path,
    documents: List[Document],
    embedding_model: HuggingFaceEmbeddings,
) -> FAISS:
    index_path = directory / "index.faiss"
    store_path = directory / "index.pkl"

    if index_path.exists() and store_path.exists():
        logger.info("📁 Loading existing FAISS index from %s", directory)
        return FAISS.load_local(
            directory.as_posix(),
            embedding_model,
            allow_dangerous_deserialization=True,
        )

    logger.info("✨ Building new FAISS index at %s", directory)
    vectordb = FAISS.from_documents(documents, embedding_model)
    vectordb.save_local(directory.as_posix())
    return vectordb

