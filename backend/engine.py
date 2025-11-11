import json
import os
import shutil  # <-- 1. IMPORT THIS
import logging
import uuid
from pathlib import Path
from typing import Literal, Optional, Iterable, List, Set, Dict, Tuple
import time
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from config_loader import load_agent_config
from storage import (
    create_metadata_store,
    ChunkRecord,
    ConversationRecord,
)
from ingestion import (
    ensure_faq_initialized as ingestion_ensure_faq_initialized,
    list_pdf_files as ingestion_list_pdf_files,
    load_documents as ingestion_load_documents,
    make_document_id,
    split_documents_llm_propositional as ingestion_split_documents_llm_propositional,
    split_documents_standard as ingestion_split_documents_standard,
)
from retrieval.hybrid import create_hybrid_retriever
from vector_store import bm25_index
from orchestration import (
    ToolOrchestrator,
    faq_search_tool,
    table_to_graph_tool,
    synthesize_answer,
    maybe_generate_clarification,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Directories ---
DATA_DIR = Path(__file__).resolve().parent.joinpath("data")
VECTOR_STORE_BASE_DIR = DATA_DIR.joinpath("vector_store")
FAQ_PATH = DATA_DIR.joinpath("faq.json")
DATA_DIR.mkdir(exist_ok=True)
VECTOR_STORE_BASE_DIR.mkdir(exist_ok=True)

# --- Environment ---
ENV_PATH = Path(__file__).resolve().parent.joinpath(".env")
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

CONFIG = load_agent_config()
METADATA_STORE = create_metadata_store(CONFIG)
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name=CONFIG.get("embeddings", {}).get("text_model", "nomic-ai/nomic-embed-text-v1.5"),
    model_kwargs={"device": "cpu", "trust_remote_code": True},
    encode_kwargs={"normalize_embeddings": CONFIG.get("faiss", {}).get("normalize", True)},
)
vector_store_override = CONFIG.get("faiss", {}).get("index_path")
if vector_store_override:
    VECTOR_STORE_BASE_DIR = Path(vector_store_override)
    VECTOR_STORE_BASE_DIR.mkdir(parents=True, exist_ok=True)
# --- FAQ Cache ---
_FAQ_CACHE: dict[str, str] = {}


def _sanitize_metadata(metadata: dict) -> dict:
    sanitized = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            sanitized[key] = ",".join(str(item) for item in value)
        elif isinstance(value, dict):
            sanitized[key] = json.dumps(value, ensure_ascii=False)
        else:
            sanitized[key] = value
    return sanitized


def _serialize_sources(documents: Iterable[Document]) -> List[Dict[str, object]]:
    serialized: List[Dict[str, object]] = []
    for doc in documents:
        meta = dict(doc.metadata)
        entry: Dict[str, object] = {}
        source = meta.get("source") or meta.get("file_name") or meta.get("doc_id")
        if source is not None:
            entry["source"] = source
        for key in ("page", "page_start", "page_end", "chunk_id", "score"):
            value = meta.get(key)
            if value is not None:
                entry[key] = value
        serialized.append(entry)
    return serialized


def _sanitize_chart_payload(chart_payload: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not chart_payload:
        return None
    sanitized = {key: value for key, value in chart_payload.items() if key != "data"}
    return sanitized


def _log_conversation_entry(
    *,
    query: str,
    answer: str,
    api: Literal["google", "openai"],
    mode: ModeLiteral,
    tool: str,
    documents: Iterable[Document],
    chart_payload: Optional[Dict[str, object]],
    chat_history: List[Tuple[str, str]],
    selected_documents: Optional[List[str]],
    latency_seconds: float,
) -> None:
    try:
        record = ConversationRecord(
            timestamp=datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            api=api,
            mode=mode,
            tool=tool,
            latency_ms=latency_seconds * 1000.0,
            user_query=query,
            answer=answer,
            sources=_serialize_sources(list(documents)),
            chart=_sanitize_chart_payload(chart_payload),
            chat_history=[list(turn) for turn in chat_history],
            selected_documents=selected_documents or [],
        )
        METADATA_STORE.log_conversation(record)
    except NotImplementedError:
        logger.debug("Metadata store does not support conversation logging.")
    except Exception as exc:
        logger.warning("Failed to log conversation entry: %s", exc)


def _make_chunk_record(chunk: Document) -> ChunkRecord:
    metadata = dict(chunk.metadata)
    doc_id = metadata.get("doc_id")
    if not doc_id:
        raise ValueError("Chunk metadata missing doc_id")

    chunk_id = metadata.get("chunk_id") or uuid.uuid4().hex
    metadata["chunk_id"] = chunk_id
    chunk.metadata["chunk_id"] = chunk_id

    page_start = metadata.get("page_start") or metadata.get("page")
    page_end = metadata.get("page_end") or metadata.get("page")
    chunk_type = metadata.get("chunk_type", "text")

    sanitized_metadata = _sanitize_metadata(metadata)
    return ChunkRecord(
        chunk_id=chunk_id,
        doc_id=doc_id,
        page_start=page_start,
        page_end=page_end,
        chunk_type=chunk_type,
        metadata=sanitized_metadata,
        content=chunk.page_content,
    )


# --- Modes ---
ModeLiteral = Literal["turiya-standard", "turiya-thinking"]

# --- LLMs ---
google_chatmodel = lambda api_key: ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key
)
openai_chatmodel = lambda api_key: ChatOpenAI(
    api_key=api_key, model="gpt-4", temperature=0.1
)


def _get_api_key(api: Literal["google", "openai"]):
    env_var = "GEMINI_API_KEY" if api == "google" else "OPENAI_API_KEY"
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(
            f"Missing required environment variable '{env_var}'. "
            "Please set the API key on the backend before running queries."
        )
    return api_key


def api_key_available(api: Literal["google", "openai"]) -> bool:
    try:
        _get_api_key(api)
        return True
    except ValueError:
        return False

# --- Custom Prompt Template ---
# --- Document Loader ---
def list_pdf_files():
    return ingestion_list_pdf_files()


def load_documents():
    return ingestion_load_documents()


def ensure_faq_initialized() -> None:
    global _FAQ_CACHE
    ingestion_ensure_faq_initialized()
    try:
        with FAQ_PATH.open("r", encoding="utf-8") as faq_file:
            _FAQ_CACHE = json.load(faq_file)
        logger.info("ℹ️ Loaded %s FAQ entries", len(_FAQ_CACHE))
    except Exception as faq_error:
        logger.error("❌ Failed to load FAQ file: %s", faq_error)
        raise


def get_faq_entries() -> dict[str, str]:
    if not _FAQ_CACHE:
        ensure_faq_initialized()
    return _FAQ_CACHE.copy()

# --- Text Splitter ---
def split_documents_standard(documents: list[Document]) -> list[Document]:
    return ingestion_split_documents_standard(documents)


def split_documents_llm_propositional(
    documents: list[Document],
    google_api_key: str,
) -> list[Document]:
    return ingestion_split_documents_llm_propositional(documents, google_api_key)

# --- Retriever ---
def get_vector_store_dir(mode: ModeLiteral, api: Literal["google", "openai"]) -> Path:
    subdir_mode = "turiya_standard" if mode == "turiya-standard" else "turiya_thinking"
    subdir_api = "google" if api == "google" else "openai"
    subdir = f"{subdir_mode}_{subdir_api}"
    directory = VECTOR_STORE_BASE_DIR.joinpath(subdir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _vector_store_exists(directory: Path) -> bool:
    if not directory.exists():
        return False
    index_file = directory.joinpath("index.faiss")
    store_file = directory.joinpath("index.pkl")
    return index_file.exists() and store_file.exists()
def load_or_create_retriever(
    api: Literal["google", "openai"],
    mode: ModeLiteral,
    force_reload: bool = False,
    selected_documents: Optional[list] = None,
):
    store_directory = get_vector_store_dir(mode, api)
    logger.info(
        "🔄 load_or_create_retriever called - api=%s mode=%s force_reload=%s",
        api,
        mode,
        force_reload,
    )
    logger.info("📂 Vector store directory: %s", store_directory)
    logger.info("📂 Vector store exists: %s", _vector_store_exists(store_directory))
    _get_api_key(api)  # Validate availability early

    if _vector_store_exists(store_directory) and not force_reload:
        logger.info("📁 Loading existing FAISS index from disk.")
        try:
            vectordb = FAISS.load_local(
                store_directory.as_posix(),
                EMBEDDING_MODEL,
                allow_dangerous_deserialization=True,
            )
            logger.info("✅ Successfully loaded existing FAISS index from disk.")
        except Exception as e:
            logger.error(f"❌ Error loading existing FAISS index: {e}")
            raise
    else:
        if store_directory.exists():
            logger.info("🗑️ Deleting existing vector store...")
            try:
                shutil.rmtree(store_directory)
            except Exception as e:
                logger.warning("⚠️ Unable to remove directory normally (%s). Attempting manual cleanup.", e)
                for file_path in store_directory.glob("*"):
                    try:
                        file_path.unlink()
                    except Exception as unlink_error:
                        logger.error("❌ Failed to delete %s: %s", file_path, unlink_error)
            logger.info("✅ Cleared vector store at: %s", store_directory)

        logger.info("✨ Creating new FAISS index...")
        try:
            store_directory.mkdir(parents=True, exist_ok=True)
            logger.info("📄 Starting to load documents...")
            documents = load_documents()
            logger.info(f"📄 Loaded {len(documents)} document(s) total")
            if not documents:
                logger.error("❌ No PDF files found in the data folder.")
                raise ValueError("No PDF files found in the data folder.")
            
            logger.info("✂️ Starting to split documents into chunks...")
            if mode == "turiya-thinking":
                try:
                    google_api_key = _get_api_key("google")
                except ValueError as exc:
                    logger.error(
                        "❌ Cannot run propositional chunking without GOOGLE_API_KEY: %s", exc
                    )
                    raise
                raw_chunks = split_documents_llm_propositional(documents, google_api_key)
            else:
                raw_chunks = split_documents_standard(documents)
            logger.info("✂️ Split %s documents into %s chunks.", len(documents), len(raw_chunks))

            chunk_records = []
            sanitized_documents = []
            for chunk in raw_chunks:
                record = _make_chunk_record(chunk)
                chunk_records.append(record)
                sanitized_documents.append(
                    Document(page_content=chunk.page_content, metadata=record.metadata)
                )

            METADATA_STORE.upsert_chunks(chunk_records)

            logger.info("🔧 Building FAISS index with local embeddings...")
            vectordb = FAISS.from_documents(
                sanitized_documents,
                EMBEDDING_MODEL,
            )
            vectordb.save_local(store_directory.as_posix())
            logger.info("✅ FAISS index created and persisted at: %s", store_directory)
        except Exception as e:
            logger.error(f"❌ Error creating FAISS index: {e}")
            raise

    documents = list(vectordb.docstore._dict.values())
    bm25_index.register_chunk_documents(store_directory, documents)
    bm25_index.ensure_bm25_index(store_directory, documents)

    logger.info("🎯 Creating hybrid retriever...")

    allowed_doc_ids = None
    if selected_documents:
        allowed_doc_ids = {
            make_document_id(str(DATA_DIR / doc_name))
            for doc_name in selected_documents
        }
        logger.info("🔍 Limiting retriever to doc_ids: %s", allowed_doc_ids)

    retriever = create_hybrid_retriever(
        vectorstore=vectordb,
        directory=store_directory,
        allowed_doc_ids=allowed_doc_ids,
    )
    logger.info("✅ Retriever created successfully.")
    return retriever


# --- Vector Store Bootstrap ---
def ensure_vector_store_initialized(
    api: Literal["google", "openai"], mode: ModeLiteral
) -> None:
    directory = get_vector_store_dir(mode, api)
    if _vector_store_exists(directory):
        logger.info(
            "🟢 Vector store already present for api=%s mode=%s at %s",
            api,
            mode,
            directory,
        )
        return

    logger.info(
        "⚪ Vector store missing for api=%s mode=%s. Initializing...", api, mode
    )
    try:
        load_or_create_retriever(api=api, mode=mode, force_reload=False)
        logger.info(
            "✅ Vector store initialization complete for api=%s mode=%s", api, mode
        )
    except ValueError as exc:
        logger.warning(
            "⚠️ Unable to initialize vector store for api=%s mode=%s: %s",
            api,
            mode,
            exc,
        )
    except Exception as exc:
        logger.error(
            "❌ Failed to initialize vector store for api=%s mode=%s: %s",
            api,
            mode,
            exc,
        )
        raise


# --- Querying ---
def query_llm(
    retriever,
    query: str,
    api: Literal["google", "openai"],
    chat_history: list[tuple[str, str]],
    mode: ModeLiteral,
    selected_documents: Optional[List[str]] = None,
):
    api_key = _get_api_key(api)
    start_time = time.perf_counter()

    def finalize(
        answer_text: str,
        docs: List[Document],
        chart_payload: Optional[Dict[str, object]],
        tool_used: str,
    ):
        elapsed = time.perf_counter() - start_time
        logger.info("⏱️ Query completed in %.2fs", elapsed)
        try:
            _log_conversation_entry(
                query=query,
                answer=answer_text,
                api=api,
                mode=mode,
                tool=tool_used,
                documents=docs,
                chart_payload=chart_payload,
                chat_history=chat_history,
                selected_documents=selected_documents,
                latency_seconds=elapsed,
            )
        except Exception as exc:
            logger.debug("Conversation logging failed: %s", exc)
        return answer_text, docs, chart_payload

    try:
        logger.info(
            "🧠 Executing query with api=%s mode=%s | chat history turns=%s",
            api,
            mode,
            len(chat_history),
        )
        router_llm = google_chatmodel(api_key) if api == "google" else openai_chatmodel(api_key)
        router = ToolOrchestrator(router_llm, faq_lookup=get_faq_entries)
        tool_choice = router.select_tool(query)
        logger.info("🛠️ Tool router selected: %s", tool_choice)

        if tool_choice == "faq_search":
            answer = faq_search_tool(query, get_faq_entries)
            if answer:
                logger.info("📄 FAQ tool handled the query.")
                return finalize(answer, [], None, "faq_search")
            logger.info("ℹ️ FAQ did not yield a result; falling back to hybrid search.")
            tool_choice = "hybrid_search_tool"

        if hasattr(retriever, "retrieve"):
            documents = retriever.retrieve(query)
        else:
            documents = retriever.get_relevant_documents(query)
        documents = list(documents)

        if tool_choice == "table_to_graph_tool":
            graph_result = table_to_graph_tool(query, documents)
            if graph_result:
                logger.info("📊 Table-to-graph tool handled the query.")
                sources = list(graph_result.sources)
                return finalize(graph_result.answer, sources, graph_result.chart, "table_to_graph_tool")
            logger.info("ℹ️ Graph tool could not produce a chart; falling back to hybrid search.")
            tool_choice = "hybrid_search_tool"

        if not documents:
            logger.info("No documents retrieved for the query.")
            return finalize(
                "I'm sorry, I could not find any relevant information in the provided documents.",
                [],
                None,
                tool_choice,
            )

        clarification = maybe_generate_clarification(
            llm=router_llm,
            query=query,
            documents=documents,
        )
        if clarification:
            logger.info("ℹ️ Clarification requested from the user.")
            return finalize(clarification, [], None, "clarification_prompt")

        retrieved_ids = [
            doc.metadata.get("chunk_id") for doc in documents if doc.metadata.get("chunk_id")
        ]
        logger.info("🗂️ Retrieved %s chunk(s): %s", len(retrieved_ids), retrieved_ids)

        qa_llm = google_chatmodel(api_key) if api == "google" else openai_chatmodel(api_key)
        answer = synthesize_answer(
            llm=qa_llm,
            query=query,
            documents=documents,
            chat_history=chat_history,
        )
        return finalize(answer, documents, None, tool_choice)
    except Exception as e:
        logger.error(f"❌ Error during query: {e}")
        raise e

