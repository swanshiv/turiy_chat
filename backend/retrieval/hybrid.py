from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

from langchain_community.vectorstores import FAISS
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

try:
    from config_loader import load_agent_config
    from vector_store import bm25_index
except ModuleNotFoundError:  # pragma: no cover
    from backend.config_loader import load_agent_config  # type: ignore
    from backend.vector_store import bm25_index  # type: ignore

logger = logging.getLogger(__name__)
CONFIG = load_agent_config()


class HybridRetriever(BaseRetriever):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(
        self,
        vectorstore: FAISS,
        directory: Path,
        chunk_map: Dict[str, Document],
        allowed_doc_ids: Optional[Set[str]],
        dense_k: int,
        bm25_k: int,
        rrf_k: int,
        top_n: int,
    ):
        super().__init__()
        self.vectorstore = vectorstore
        self.directory = directory
        self.chunk_map = chunk_map
        self.allowed_doc_ids = allowed_doc_ids
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.top_n = top_n

    def _filter_allowed(self, doc: Document) -> bool:
        if not self.allowed_doc_ids:
            return True
        return doc.metadata.get("doc_id") in self.allowed_doc_ids

    def _hybrid_search(self, query: str) -> List[Document]:
        dense_results = self.vectorstore.similarity_search_with_relevance_scores(
            query,
            k=self.dense_k,
        )
        dense_ranked = [
            doc.metadata.get("chunk_id")
            for doc, _ in dense_results
            if doc.metadata.get("chunk_id") and self._filter_allowed(doc)
        ]

        bm25_candidates = bm25_index.bm25_search(self.directory, query, top_k=self.bm25_k)
        bm25_ranked = []
        for chunk_id, _ in bm25_candidates:
            doc = self.chunk_map.get(chunk_id)
            if doc and self._filter_allowed(doc):
                bm25_ranked.append(chunk_id)

        scores = defaultdict(float)
        for rank, chunk_id in enumerate(dense_ranked, start=1):
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)
        for rank, chunk_id in enumerate(bm25_ranked, start=1):
            scores[chunk_id] += 1.0 / (self.rrf_k + rank)

        if not scores:
            return [doc for doc, _ in dense_results[: self.top_n]]

        ranked_chunk_ids = [
            chunk_id
            for chunk_id, _ in sorted(
                scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ][: self.top_n]

        results: List[Document] = []
        for chunk_id in ranked_chunk_ids:
            doc = self.chunk_map.get(chunk_id)
            if not doc:
                continue
            enriched = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "retrieval_rrf_score": scores[chunk_id],
                },
            )
            results.append(enriched)

        return results or [doc for doc, _ in dense_results[: self.top_n]]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self._hybrid_search(query)

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._hybrid_search(query)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return self._hybrid_search(query)


def create_hybrid_retriever(
    vectorstore: FAISS,
    directory: Path,
    allowed_doc_ids: Optional[Set[str]] = None,
) -> HybridRetriever:
    retrieval_cfg = CONFIG.get("retrieval", {})
    dense_k = retrieval_cfg.get("dense", {}).get("k", 5)
    bm25_k = retrieval_cfg.get("bm25", {}).get("k", 10)
    rrf_cfg = retrieval_cfg.get("rrf", {})
    rrf_k = rrf_cfg.get("k_constant", 60)
    top_n = rrf_cfg.get("top_n", 10)

    bm25_index.ensure_bm25_index(directory)
    chunk_map = bm25_index.get_registered_documents(directory)

    logger.info(
        "🔁 HybridRetriever configured (dense=%s, bm25=%s, top_n=%s)",
        dense_k,
        bm25_k,
        top_n,
    )

    return HybridRetriever(
        vectorstore=vectorstore,
        directory=directory,
        chunk_map=chunk_map,
        allowed_doc_ids=allowed_doc_ids,
        dense_k=dense_k,
        bm25_k=bm25_k,
        rrf_k=rrf_k,
        top_n=top_n,
    )
