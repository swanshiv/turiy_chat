from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

_BM25_MODELS: Dict[Path, BM25Okapi] = {}
_BM25_CHUNK_IDS: Dict[Path, List[str]] = {}
_CHUNK_ID_DOCS: Dict[Path, Dict[str, Document]] = {}


def register_chunk_documents(directory: Path, documents: List[Document]) -> None:
    _CHUNK_ID_DOCS[directory] = {
        doc.metadata.get("chunk_id"): doc
        for doc in documents
        if doc.metadata.get("chunk_id")
    }


def get_registered_documents(directory: Path) -> Dict[str, Document]:
    return _CHUNK_ID_DOCS.get(directory, {})


def build_bm25_index(directory: Path, documents: List[Document]) -> None:
    tokens = [_tokenize_text(doc.page_content) for doc in documents]
    chunk_ids = [doc.metadata.get("chunk_id") for doc in documents]
    bm25 = BM25Okapi(tokens)
    _BM25_MODELS[directory] = bm25
    _BM25_CHUNK_IDS[directory] = chunk_ids
    register_chunk_documents(directory, documents)

    payload = {"tokens": tokens, "chunk_ids": chunk_ids}
    with (directory / "bm25.pkl").open("wb") as handle:
        pickle.dump(payload, handle)


def ensure_bm25_index(directory: Path, documents: Optional[List[Document]] = None) -> None:
    if directory in _BM25_MODELS:
        return

    bm25_path = directory / "bm25.pkl"
    if bm25_path.exists():
        with bm25_path.open("rb") as handle:
            payload = pickle.load(handle)
        tokens = payload["tokens"]
        chunk_ids = payload["chunk_ids"]
        _BM25_MODELS[directory] = BM25Okapi(tokens)
        _BM25_CHUNK_IDS[directory] = chunk_ids
        return

    if documents is None:
        raise RuntimeError("BM25 index missing and documents not provided to rebuild it.")

    build_bm25_index(directory, documents)


def bm25_search(directory: Path, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    ensure_bm25_index(directory)
    bm25 = _BM25_MODELS[directory]
    chunk_ids = _BM25_CHUNK_IDS[directory]
    scores = bm25.get_scores(_tokenize_text(query))
    scored_chunks = sorted(
        zip(chunk_ids, scores),
        key=lambda item: item[1],
        reverse=True,
    )
    return scored_chunks[:top_k]


def _tokenize_text(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())

