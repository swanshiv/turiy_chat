import json
import os
import logging
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, List, Dict, Any

import psycopg2
from psycopg2 import sql
from supabase import create_client, Client

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    doc_id: str
    file_name: str
    page_count: int


@dataclass
class ChunkRecord:
    chunk_id: str
    doc_id: str
    page_start: Optional[int]
    page_end: Optional[int]
    chunk_type: str
    metadata: dict
    content: str


@dataclass
class ConversationRecord:
    timestamp: str
    api: str
    mode: str
    tool: str
    latency_ms: float
    user_query: str
    answer: str
    sources: List[Dict[str, Any]]
    chart: Optional[Dict[str, Any]]
    chat_history: List[List[str]]
    selected_documents: Optional[List[str]]


class MetadataStore:
    def upsert_document(self, record: DocumentRecord) -> None:
        raise NotImplementedError

    def upsert_chunks(self, records: Iterable[ChunkRecord]) -> None:
        raise NotImplementedError

    def log_conversation(self, record: ConversationRecord) -> None:
        raise NotImplementedError


class LocalSQLiteStore(MetadataStore):
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _initialize(self):
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    page_count INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    page_start INTEGER,
                    page_end INTEGER,
                    chunk_type TEXT,
                    metadata TEXT,
                    content TEXT,
                    FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    api TEXT,
                    mode TEXT,
                    tool TEXT,
                    latency_ms REAL,
                    query TEXT,
                    answer TEXT,
                    sources TEXT,
                    chart TEXT,
                    chat_history TEXT,
                    selected_documents TEXT
                )
                """
            )
            conn.commit()

    def upsert_document(self, record: DocumentRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(doc_id, file_name, page_count)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    file_name = excluded.file_name,
                    page_count = excluded.page_count
                """,
                (record.doc_id, record.file_name, record.page_count),
            )
            conn.commit()

    def upsert_chunks(self, records: Iterable[ChunkRecord]) -> None:
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO chunks(chunk_id, doc_id, page_start, page_end, chunk_type, metadata, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    doc_id = excluded.doc_id,
                    page_start = excluded.page_start,
                    page_end = excluded.page_end,
                    chunk_type = excluded.chunk_type,
                    metadata = excluded.metadata,
                    content = excluded.content
                """,
                [
                    (
                        record.chunk_id,
                        record.doc_id,
                        record.page_start,
                        record.page_end,
                        record.chunk_type,
                        json.dumps(record.metadata),
                        record.content,
                    )
                    for record in records
                ],
            )
            conn.commit()

    def log_conversation(self, record: ConversationRecord) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(
                    timestamp,
                    api,
                    mode,
                    tool,
                    latency_ms,
                    query,
                    answer,
                    sources,
                    chart,
                    chat_history,
                    selected_documents
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.timestamp,
                    record.api,
                    record.mode,
                    record.tool,
                    record.latency_ms,
                    record.user_query,
                    record.answer,
                    json.dumps(record.sources),
                    json.dumps(record.chart),
                    json.dumps(record.chat_history),
                    json.dumps(record.selected_documents),
                ),
            )
            conn.commit()


class SupabaseStore(MetadataStore):
    def __init__(self, client: Client, table_documents: str, table_chunks: str, table_conversations: str):
        self.client = client
        self.table_documents = table_documents
        self.table_chunks = table_chunks
        self.table_conversations = table_conversations

    def upsert_document(self, record: DocumentRecord) -> None:
        payload = asdict(record)
        try:
            response = self.client.table(self.table_documents).upsert(payload).execute()
        except Exception as exc:
            raise RuntimeError(f"Supabase upsert_document failed: {exc}") from exc
        status_code = getattr(response, "status_code", 200)
        if status_code and status_code >= 400:
            raise RuntimeError(f"Supabase upsert_document returned status {status_code}")

    def upsert_chunks(self, records: Iterable[ChunkRecord]) -> None:
        payload = [
            {
                "chunk_id": record.chunk_id,
                "doc_id": record.doc_id,
                "page_start": record.page_start,
                "page_end": record.page_end,
                "chunk_type": record.chunk_type,
                "metadata": record.metadata,
                "content": record.content,
            }
            for record in records
        ]
        if not payload:
            return
        try:
            response = self.client.table(self.table_chunks).upsert(payload).execute()
        except Exception as exc:
            raise RuntimeError(f"Supabase upsert_chunks failed: {exc}") from exc
        status_code = getattr(response, "status_code", 200)
        if status_code and status_code >= 400:
            raise RuntimeError(f"Supabase upsert_chunks returned status {status_code}")

    def log_conversation(self, record: ConversationRecord) -> None:
        payload = {
            "timestamp": record.timestamp,
            "api": record.api,
            "mode": record.mode,
            "tool": record.tool,
            "latency_ms": record.latency_ms,
            "query": record.user_query,
            "answer": record.answer,
            "sources": record.sources,
            "chart": record.chart,
            "chat_history": record.chat_history,
            "selected_documents": record.selected_documents,
        }
        try:
            response = self.client.table(self.table_conversations).insert(payload).execute()
        except Exception as exc:
            raise RuntimeError(f"Supabase log_conversation failed: {exc}") from exc
        status_code = getattr(response, "status_code", 200)
        if status_code and status_code >= 400:
            raise RuntimeError(f"Supabase log_conversation returned status {status_code}")


def _identifier(path: str) -> sql.Identifier:
    parts = path.split(".")
    if len(parts) == 2:
        return sql.Identifier(parts[0], parts[1])
    return sql.Identifier(path)


def _sanitize_identifier_for_index(path: str) -> str:
    return path.replace(".", "_")


def _ensure_supabase_tables(
    connection_url: str,
    documents_table: str,
    chunks_table: str,
    conversations_table: str,
) -> None:
    try:
        with psycopg2.connect(connection_url) as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto")
                doc_ident = _identifier(documents_table)
                chunk_ident = _identifier(chunks_table)
                convo_ident = _identifier(conversations_table)
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            doc_id TEXT PRIMARY KEY,
                            file_name TEXT NOT NULL,
                            page_count INTEGER
                        )
                        """
                    ).format(doc_ident)
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            chunk_id TEXT PRIMARY KEY,
                            doc_id TEXT NOT NULL REFERENCES {}(doc_id) ON DELETE CASCADE,
                            page_start INTEGER,
                            page_end INTEGER,
                            chunk_type TEXT,
                            metadata JSONB,
                            content TEXT
                        )
                        """
                    ).format(chunk_ident, doc_ident)
                )
                index_name = sql.Identifier(f"{_sanitize_identifier_for_index(chunks_table)}_doc_id_idx")
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (doc_id)").format(index_name, chunk_ident)
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {} (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            timestamp TIMESTAMPTZ NOT NULL,
                            api TEXT,
                            mode TEXT,
                            tool TEXT,
                            latency_ms DOUBLE PRECISION,
                            query TEXT,
                            answer TEXT,
                            sources JSONB,
                            chart JSONB,
                            chat_history JSONB,
                            selected_documents JSONB
                        )
                        """
                    ).format(convo_ident)
                )
    except Exception as exc:  # pragma: no cover - fall back silently
        logger.warning("Unable to ensure Supabase tables exist automatically: %s", exc)


def create_metadata_store(config: dict) -> MetadataStore:
    supabase_cfg = config.get("supabase", {})
    if supabase_cfg.get("enable"):
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if url and key:
            documents_table = supabase_cfg.get("table_documents", "documents")
            chunks_table = supabase_cfg.get("table_chunks", "chunks")
            conversations_table = supabase_cfg.get("table_conversations", "conversations")
            db_url = os.getenv("SUPABASE_DB_URL")
            if db_url:
                _ensure_supabase_tables(db_url, documents_table, chunks_table, conversations_table)
            else:
                logger.warning(
                    "SUPABASE_DB_URL not provided; automatic table creation skipped. "
                    "Ensure tables '%s', '%s', and '%s' exist.",
                    documents_table,
                    chunks_table,
                    conversations_table,
                )
            client = create_client(url, key)
            logger.info("📦 Using Supabase metadata store")
            return SupabaseStore(
                client,
                table_documents=documents_table,
                table_chunks=chunks_table,
                table_conversations=conversations_table,
            )
        logger.warning("Supabase enabled in config but environment variables missing; falling back to SQLite store.")
    logger.info("📦 Using local SQLite metadata store")
    return LocalSQLiteStore(Path(config.get("faiss", {}).get("index_path", "backend/data/faiss")) / "metadata.db")

