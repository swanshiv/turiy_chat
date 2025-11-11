from __future__ import annotations

import logging
import uuid
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document

try:
    from config_loader import load_agent_config
except ModuleNotFoundError:  # pragma: no cover
    from backend.config_loader import load_agent_config  # type: ignore
from .models import ChunkedDocument

logger = logging.getLogger(__name__)

CONFIG = load_agent_config()


def split_documents_standard(documents: List[Document]) -> List[ChunkedDocument]:
    chunk_cfg = CONFIG.get("chunking", {}).get("standard", {})
    chunk_size = chunk_cfg.get("chunk_size", 1000)
    chunk_overlap = chunk_cfg.get("chunk_overlap", 250)

    logger.info(
        "✂️ Initializing standard text splitter (chunk_size=%s, chunk_overlap=%s)...",
        chunk_size,
        chunk_overlap,
    )
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    results: List[ChunkedDocument] = []
    for doc in documents:
        chunk_type = doc.metadata.get("chunk_type", "text")
        if chunk_type == "table":
            doc.metadata["chunk_id"] = doc.metadata.get("chunk_id") or uuid.uuid4().hex
            results.append(ChunkedDocument.from_document(doc))
            continue

        splits = splitter.split_documents([doc])
        for idx, piece in enumerate(splits, start=1):
            piece.metadata["doc_id"] = doc.metadata.get("doc_id")
            piece.metadata["chunk_type"] = "text"
            base_page = doc.metadata.get("page")
            piece.metadata["page_start"] = piece.metadata.get("page_start", base_page)
            piece.metadata["page_end"] = piece.metadata.get("page_end", base_page)
            piece.metadata["chunk_order"] = idx
            piece.metadata["chunk_id"] = piece.metadata.get("chunk_id") or uuid.uuid4().hex
            results.append(ChunkedDocument.from_document(piece))

    logger.info(
        "✂️ Standard split complete: %s document(s) -> %s chunk(s)",
        len(documents),
        len(results),
    )
    return results


def split_documents_llm_propositional(documents: List[Document], google_api_key: str) -> List[ChunkedDocument]:
    chunk_cfg = CONFIG.get("chunking", {}).get("smart", {})
    primary_chunk_size = chunk_cfg.get("primary_chunk_size", 2000)
    primary_chunk_overlap = chunk_cfg.get("primary_chunk_overlap", 300)
    delimiter = chunk_cfg.get("refinement_delimiter", "<CHUNK>")

    logger.info("🤖 Initializing LLM-based structural refinement pipeline (Gemini 2.5 Pro)...")
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        temperature=0,
        google_api_key=google_api_key,
    )

    refinement_prompt = PromptTemplate(
        input_variables=["text"],
        template=f"""
You are reorganizing document excerpts into coherent, self-contained knowledge chunks.

Guidelines:
- Merge adjacent sentences that describe the same topic.
- Split apart unrelated concepts so each chunk covers a single idea.
- Preserve numeric and tabular context verbatim.
- Do not add commentary.

Return the refined chunks using the delimiter {delimiter} on its own line before each chunk. Do not include any other text.

Source Text:
{{text}}
""",
    )

    text_docs = [doc for doc in documents if doc.metadata.get("chunk_type") != "table"]
    table_docs = [doc for doc in documents if doc.metadata.get("chunk_type") == "table"]

    initial_splitter = RecursiveCharacterTextSplitter(
        chunk_size=primary_chunk_size,
        chunk_overlap=primary_chunk_overlap,
    )
    base_segments = initial_splitter.split_documents(text_docs)
    logger.info("🤖 Primary split created %s base segment(s)", len(base_segments))

    chain = refinement_prompt | llm
    refined_text_chunks: List[ChunkedDocument] = []
    table_chunks: List[ChunkedDocument] = []

    for index, segment in enumerate(base_segments):
        source = segment.metadata.get("source", "unknown")
        try:
            response = chain.invoke({"text": segment.page_content})
            raw_output = getattr(response, "content", str(response))
        except Exception as exc:
            logger.error("❌ LLM refinement failed for %s segment %s: %s", source, index, exc)
            raise

        pieces = [
            piece.strip()
            for piece in raw_output.split(delimiter)
            if piece.strip()
        ]

        if not pieces:
            logger.debug("ℹ️ No refined pieces returned for segment %s; keeping original text.", index)
            pieces = [segment.page_content.strip()]

        base_page = segment.metadata.get("page")
        page_numbers = segment.metadata.get("page_numbers") or (
            [base_page] if base_page else []
        )
        page_start = min(page_numbers) if page_numbers else base_page
        page_end = max(page_numbers) if page_numbers else base_page

        for chunk_order, chunk_text in enumerate(pieces, start=1):
            metadata = {
                **segment.metadata,
                "chunk_type": "text",
                "refinement_source_index": index,
                "refinement_chunk_order": chunk_order,
                "page_start": page_start,
                "page_end": page_end,
                "chunk_id": uuid.uuid4().hex,
            }
            refined_text_chunks.append(ChunkedDocument(chunk_text, metadata))

    for table_doc in table_docs:
        table_doc.metadata["chunk_id"] = table_doc.metadata.get("chunk_id") or uuid.uuid4().hex
        table_chunks.append(ChunkedDocument.from_document(table_doc))

    merge_cfg = chunk_cfg.get("secondary_merge", {})
    max_length = merge_cfg.get("max_combined_length", primary_chunk_size)
    allow_cross_page = merge_cfg.get("allow_cross_page", False)

    if refined_text_chunks and max_length:
        merged_text_chunks = merge_adjacent_chunks(
            refined_text_chunks,
            max_length=max_length,
            allow_cross_page=allow_cross_page,
        )
    else:
        merged_text_chunks = refined_text_chunks

    logger.info(
        "🤖 Structural refinement complete: %s base segment(s) -> %s refined chunk(s) (+%s tables)",
        len(base_segments),
        len(merged_text_chunks),
        len(table_chunks),
    )
    return merged_text_chunks + table_chunks


def merge_adjacent_chunks(
    chunks: List[ChunkedDocument],
    max_length: int,
    allow_cross_page: bool = False,
) -> List[ChunkedDocument]:
    merged: List[ChunkedDocument] = []
    buffer: List[ChunkedDocument] = []

    for chunk in chunks:
        if not buffer:
            buffer.append(chunk)
            continue

        last = buffer[-1]
        same_doc = chunk.metadata.get("doc_id") == last.metadata.get("doc_id")
        contiguous_page = chunk.metadata.get("page_start") == last.metadata.get("page_end")

        if same_doc and (allow_cross_page or contiguous_page):
            combined_length = sum(len(item.page_content) for item in buffer) + len(chunk.page_content)
            if combined_length <= max_length:
                buffer.append(chunk)
                continue

        merged.append(_collapse_buffer(buffer))
        buffer = [chunk]

    if buffer:
        merged.append(_collapse_buffer(buffer))

    return merged


def _collapse_buffer(buffer: List[ChunkedDocument]) -> ChunkedDocument:
    if len(buffer) == 1:
        return buffer[0]

    combined_text = "\n\n".join(item.page_content for item in buffer)
    metadata = dict(buffer[0].metadata)
    metadata["page_start"] = buffer[0].metadata.get("page_start")
    metadata["page_end"] = buffer[-1].metadata.get("page_end")
    metadata["chunk_id"] = uuid.uuid4().hex
    metadata["merged_chunk_count"] = len(buffer)
    return ChunkedDocument(combined_text, metadata)

