from __future__ import annotations

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import camelot  # type: ignore
import fitz  # PyMuPDF
from langchain_core.documents import Document

try:
    from config_loader import load_agent_config
except ModuleNotFoundError:  # pragma: no cover - package import context
    from backend.config_loader import load_agent_config  # type: ignore

try:
    from storage import create_metadata_store, DocumentRecord
except ModuleNotFoundError:  # pragma: no cover
    from backend.storage import create_metadata_store, DocumentRecord  # type: ignore
from .models import ChunkedDocument

logger = logging.getLogger(__name__)

CONFIG = load_agent_config()
DATA_DIR = Path(__file__).resolve().parent.parent.joinpath("data")
DATA_DIR.mkdir(exist_ok=True)

FAQ_PATH = DATA_DIR.joinpath("faq.json")
METADATA_STORE = create_metadata_store(CONFIG)


def list_pdf_files() -> List[str]:
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(DATA_DIR)
        for file in files
        if file.lower().endswith(".pdf")
    ]


def ensure_faq_initialized() -> None:
    if not FAQ_PATH.exists():
        faq_payload = {
            # Category 1: Meta-Questions
            "what is this app": "I am the 'GRI-Agent,' an AI assistant designed to help you upload, search, and analyze your documents. You can ask me questions, and I will find the most relevant information. You can also ask me to analyze data, for example: 'Graph the Scope 1 emissions.'",
            "what can you do": "I am the 'GRI-Agent,' an AI assistant designed to help you upload, search, and analyze your documents. You can ask me questions, and I will find the most relevant information. You can also ask me to analyze data, for example: 'Graph the Scope 1 emissions.'",
            "who are you": "I am the 'GRI-Agent,' an AI assistant designed to help you upload, search, and analyze your documents. You can ask me questions, and I will find the most relevant information. You can also ask me to analyze data, for example: 'Graph the Scope 1 emissions.'",
            "what is this document": "This is the 2024 Global Reporting Initiative (GRI) Report for UPS. It's a formal document that details the company's performance on environmental, social, and governance (ESG) topics for the 2024 fiscal year.",
            "what is the gri report": "This is the 2024 Global Reporting Initiative (GRI) Report for UPS. It's a formal document that details the company's performance on environmental, social, and governance (ESG) topics for the 2024 fiscal year.",
            "what period does this report cover": "This report covers the 2024 reporting year, from January 1, 2024, to December 31, 2024.",
            "what year is this data from": "This report covers the 2024 reporting year, from January 1, 2024, to December 31, 2024.",
            "who audited this report": "Yes, Deloitte & Touche LLP performed a review (limited assurance) on the GRI Content Index and an examination (reasonable assurance) on the Statement of Greenhouse Gas Emissions.",
            "was this report externally assured": "Yes, Deloitte & Touche LLP performed a review (limited assurance) on the GRI Content Index and an examination (reasonable assurance) on the Statement of Greenhouse Gas Emissions.",
            "what does gri stand for": "GRI stands for 'Global Reporting Initiative.' It's an independent, international organization that provides a common global standard for sustainability reporting.",

            # Category 2: High-Frequency Company Facts
            "who can i contact about this report": "You can send comments or questions to sustainability@ups.com or in writing to: UPS, Attention: Sustainability, 55 Glenlake Parkway, Atlanta, Georgia 30328.",
            "sustainability email": "You can send comments or questions to sustainability@ups.com or in writing to: UPS, Attention: Sustainability, 55 Glenlake Parkway, Atlanta, Georgia 30328.",
            "contact": "You can send comments or questions to sustainability@ups.com or in writing to: UPS, Attention: Sustainability, 55 Glenlake Parkway, Atlanta, Georgia 30328.",
            "where is ups headquartered": "UPS's principal executive offices are in Atlanta, Georgia, USA.",
            "main office": "UPS's principal executive offices are in Atlanta, Georgia, USA.",
            "what was the total revenue in 2024": "The total revenue for UPS in 2024 was $91.1 billion.",
            "total revenue 2024": "The total revenue for UPS in 2024 was $91.1 billion.",
            "how many countries does ups operate in": "UPS serves over 200 countries and territories.",
            "global reach": "UPS serves over 200 countries and territories.",
            "how many packages did ups deliver in 2024": "UPS delivered an average of 22.4 million packages per day in 2024, totaling 5.7 billion packages for the year.",
            "packages delivered": "UPS delivered an average of 22.4 million packages per day in 2024, totaling 5.7 billion packages for the year.",
            "how many employees does ups have": "As of September 30, 2024, UPS had 494,040 total permanent employees (full-time and part-time).",
            "employee count": "As of September 30, 2024, UPS had 494,040 total permanent employees (full-time and part-time).",
            "what are the ups business segments": "UPS has two main reporting segments: U.S. Domestic Package and International Package. The remaining businesses are reported as Supply Chain Solutions.",

            # Category 3: Strategy & Goals
            "what is ups's company strategy": "The UPS strategy is 'Customer First, People Led, Innovation Driven.'",
            "company strategy": "The UPS strategy is 'Customer First, People Led, Innovation Driven.'",
            "what is ups's main sustainability goal": "UPS's overarching sustainability goal is to achieve carbon neutrality by 2050.",
            "what is the 2050 goal": "UPS's overarching sustainability goal is to achieve carbon neutrality by 2050.",
            "what are the company's interim climate goals": "UPS has interim 2035 goals of 100% renewable electricity in its facilities, 30% sustainable aviation fuel in its air network, and a 50% reduction in CO2e per global small package.",
            "interim climate goals": "UPS has interim 2035 goals of 100% renewable electricity in its facilities, 30% sustainable aviation fuel in its air network, and a 50% reduction in CO2e per global small package.",
            "what is the ups foundation": "The UPS Foundation (TUPSF) is the company's philanthropic arm. It focuses on four areas: Health & Humanitarian Relief, Equity & Economic Empowerment, Local Community Engagement, and Planet Protection.",
            "ups foundation": "The UPS Foundation (TUPSF) is the company's philanthropic arm. It focuses on four areas: Health & Humanitarian Relief, Equity & Economic Empowerment, Local Community Engagement, and Planet Protection.",
            "what is ups's goal for volunteering": "UPS aims to provide 30 million employee volunteer hours by 2030.",
            "volunteer goal": "UPS aims to provide 30 million employee volunteer hours by 2030.",

            # Category 4: Conversational Handlers
            "hello": "Hello! How can I help you with your documents today?",
            "hi": "Hello! How can I help you with your documents today?",
            "hey": "Hello! How can I help you with your documents today?",
            "thank you": "You're welcome! Is there anything else I can help you with?",
            "thanks": "You're welcome! Is there anything else I can help you with?",
            "appreciate it": "You're welcome! Is there anything else I can help you with?",
            "goodbye": "Goodbye! Have a great day.",
            "bye": "Goodbye! Have a great day.",
            "done": "Goodbye! Have a great day.",
            "good job": "Thank you! I'm glad I could help.",
            "great answer": "Thank you! I'm glad I could help.",
            "helpful": "Thank you! I'm glad I could help.",
        }
        FAQ_PATH.write_text(json.dumps(faq_payload, indent=2), encoding="utf-8")


def load_documents() -> List[Document]:
    docs: List[Document] = []
    pdf_paths = list_pdf_files()
    logger.info("📋 Found %s PDF file(s) to load", len(pdf_paths))

    for path in pdf_paths:
        logger.info("📄 Loading: %s", os.path.basename(path))
        doc_id = make_document_id(path)
        try:
            text_docs, page_count = _preprocess_pdf_with_pymupdf(path, doc_id)
            docs.extend(doc.to_document() for doc in text_docs)

            table_docs = _extract_tables_with_camelot(path, doc_id)
            docs.extend(doc.to_document() for doc in table_docs)

            METADATA_STORE.upsert_document(
                DocumentRecord(
                    doc_id=doc_id,
                    file_name=os.path.basename(path),
                    page_count=page_count,
                )
            )
        except Exception as exc:
            logger.error("❌ Error loading %s: %s", os.path.basename(path), exc)
            raise

    logger.info("📚 Total pages loaded: %s", len(docs))
    return docs


def make_document_id(pdf_path: str) -> str:
    stem = Path(pdf_path).stem
    normalized = stem.lower().replace(" ", "_")
    return normalized


def _identify_repeating_lines(
    page_line_sets: List[List[str]],
    top_k: int,
    bottom_k: int,
) -> Tuple[set[str], set[str]]:
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    for lines in page_line_sets:
        for line in lines[:top_k]:
            header_counter[line] += 1
        for line in lines[-bottom_k:]:
            footer_counter[line] += 1

    threshold = max(3, int(0.6 * len(page_line_sets)))
    common_headers = {line for line, count in header_counter.items() if count >= threshold}
    common_footers = {line for line, count in footer_counter.items() if count >= threshold}
    return common_headers, common_footers


def _should_drop_line(
    line: str,
    common_headers: set[str],
    common_footers: set[str],
) -> bool:
    if line in common_headers or line in common_footers:
        return True

    normalized = line.lower()
    header_keywords = ("gri", "global reporting", "united parcel service", "ups")
    if any(keyword in normalized for keyword in header_keywords):
        return True

    footer_patterns = (
        r"^page\s+\d+",
        r"^\d+\s*/\s*\d+$",
        r"^\d+$",
        r"^©",
    )
    return any(re.match(pattern, normalized) for pattern in footer_patterns)


def _preprocess_pdf_with_pymupdf(pdf_path: str, doc_id: str) -> Tuple[List[ChunkedDocument], int]:
    doc = fitz.open(pdf_path)
    page_line_sets: List[List[str]] = []
    for page in doc:
        text = page.get_text("text")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        page_line_sets.append(lines)

    common_headers, common_footers = _identify_repeating_lines(page_line_sets, top_k=3, bottom_k=3)

    cleaned_documents: List[ChunkedDocument] = []
    for index, lines in enumerate(page_line_sets):
        filtered_lines = [
            line
            for line in lines
            if not _should_drop_line(line, common_headers, common_footers)
        ]
        cleaned_text = "\n".join(filtered_lines).strip()
        metadata = {
            "source": pdf_path,
            "file_name": os.path.basename(pdf_path),
            "page": index + 1,
            "page_numbers": [index + 1],
            "doc_id": doc_id,
            "chunk_type": "text",
        }
        cleaned_documents.append(ChunkedDocument(cleaned_text, metadata))

    page_count = doc.page_count
    doc.close()
    return cleaned_documents, page_count


def _dataframe_to_markdown(df) -> str:
    headers = [str(col).strip() for col in df.columns]
    rows = [
        [str(cell).strip() if str(cell).strip() else "" for cell in row.tolist()]
        for _, row in df.iterrows()
    ]

    def format_row(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    header_line = format_row(headers)
    separator_line = "| " + " | ".join("---" for _ in headers) + " |"
    body_lines = [format_row(row) for row in rows]
    if body_lines:
        return "\n".join([header_line, separator_line, *body_lines])
    return "\n".join([header_line, separator_line])


def _extract_tables_with_camelot(pdf_path: str, doc_id: str) -> List[ChunkedDocument]:
    table_documents: List[ChunkedDocument] = []
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="lattice")
    except Exception as lattice_error:
        logger.warning("⚠️ Camelot lattice extraction failed for %s: %s", pdf_path, lattice_error)
        tables = []

    if not tables:
        try:
            tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        except Exception as stream_error:
            logger.warning("⚠️ Camelot stream extraction failed for %s: %s", pdf_path, stream_error)
            tables = []

    for table in tables:
        try:
            markdown = _dataframe_to_markdown(table.df)
        except Exception as table_error:
            logger.warning("⚠️ Failed converting table to markdown on page %s: %s", table.page, table_error)
            continue

        metadata = {
            "source": pdf_path,
            "file_name": os.path.basename(pdf_path),
            "page": int(table.page) if str(table.page).isdigit() else table.page,
            "page_numbers": [int(table.page)] if str(table.page).isdigit() else [],
            "chunk_type": "table",
            "doc_id": doc_id,
        }
        table_documents.append(ChunkedDocument(markdown, metadata))

    logger.info(
        "📊 Extracted %s table chunk(s) from %s",
        len(table_documents),
        os.path.basename(pdf_path),
    )
    return table_documents

