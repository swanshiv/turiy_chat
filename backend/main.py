import os
import shutil
import logging
from pathlib import Path
from typing import Literal, List, Tuple, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Body, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import engine

ModeLiteral = Literal["turiya-standard", "turiya-thinking"]

# --- 1. Define API Data Models ---
class QueryRequest(BaseModel):
    query: str
    api: Literal["google", "openai"]
    chat_history: List[Tuple[str, str]] = Field(default=[])
    selected_documents: List[str] = Field(default=[])  # List of selected document filenames
    mode: ModeLiteral = Field(default="turiya-standard")
    escalate_to_thinking: bool = Field(
        default=False,
        description="If true, forces the request to run in turiya-thinking mode regardless of current chat mode.",
    )

class PageMetadata(BaseModel):
    source: str
    page: int = Field(default=0)

class SourceDocument(BaseModel):
    page_content: str
    metadata: PageMetadata

class GeneratedChart(BaseModel):
    title: str
    data: str
    mime_type: str = "image/png"
    description: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
    mode_used: ModeLiteral
    chart: Optional[GeneratedChart] = None

# --- 2. Create FastAPI App & CORS ---
app = FastAPI(
    title="RAG Engine API",
    description="An API for querying documents using RAG.",
)

origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def bootstrap_vector_stores():
    logger = logging.getLogger(__name__)
    logger.info("🚀 FastAPI startup: ensuring vector stores are initialized")
    engine.ensure_faq_initialized()

    available_apis: list[Literal["google", "openai"]] = [
        candidate
        for candidate in ("google", "openai")
        if engine.api_key_available(candidate)
    ]

    missing_apis = {
        candidate
        for candidate in ("google", "openai")
        if candidate not in available_apis
    }
    for missing in missing_apis:
        logger.info("🔑 API key for %s not detected; skipping bootstrap", missing)

    for api in available_apis:
        engine.ensure_vector_store_initialized(api=api, mode="turiya-standard")

    if "google" in available_apis:
        engine.ensure_vector_store_initialized(api="google", mode="turiya-thinking")
    else:
        logger.warning(
            "⚠️ GOOGLE_API_KEY missing; unable to bootstrap turiya-thinking vector store at startup"
        )

# --- 3. API Endpoints ---
@app.get("/")
def read_root():
    return {"message": "RAG API is running!"}

@app.get("/documents", response_model=List[str])
async def get_document_list():
    """Returns a list of PDF filenames in the data directory."""
    try:
        full_paths = engine.list_pdf_files()
        return [os.path.basename(path) for path in full_paths]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not list documents: {e}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Receives a PDF file and saves it."""
    file_path = engine.DATA_DIR / file.filename
    print(f"Receiving file: {file.filename}")
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"filename": file.filename, "message": "File uploaded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not upload file: {e}")
    finally:
        file.file.close()

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest = Body(...)):
    """Handles a user query against the documents."""
    print(f"Received query: {request.query}")
    print(f"Selected documents: {request.selected_documents}")
    effective_mode: ModeLiteral = (
        "turiya-thinking" if request.escalate_to_thinking else request.mode
    )
    print(f"Mode requested: {request.mode} | escalate: {request.escalate_to_thinking} -> using {effective_mode}")
    try:
        retriever = engine.load_or_create_retriever(
            api=request.api,
            mode=effective_mode,
            selected_documents=request.selected_documents,
        )
        answer, source_docs, chart_payload = engine.query_llm(
            retriever=retriever,
            query=request.query,
            api=request.api,
            chat_history=request.chat_history,
            mode=effective_mode,
            selected_documents=request.selected_documents,
        )
        formatted_sources = [
            SourceDocument(
                page_content=doc.page_content,
                metadata=PageMetadata(
                    source=doc.metadata.get("source", "Unknown"),
                    page=doc.metadata.get("page", 0),
                ),
            )
            for doc in source_docs
        ]
        chart = GeneratedChart(**chart_payload) if chart_payload else None
        return QueryResponse(answer=answer, sources=formatted_sources, mode_used=effective_mode, chart=chart)
    except Exception as e:
        print(f"❌ Error during query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RebuildRequest(BaseModel):
    api: Literal["google", "openai"]
    mode: ModeLiteral = Field(default="turiya-standard")

@app.post("/rebuild-index")
async def rebuild_index(request: RebuildRequest = Body(...)):
    """Rebuilds the vector store index from all PDF documents."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Rebuild index endpoint called")
    logger.info(f"   API: {request.api}")
    try:
        print(f"🔄 Rebuilding index with {request.api} API...")
        logger.info("Calling load_or_create_retriever with force_reload=True...")
        retriever = engine.load_or_create_retriever(
            api=request.api,
            mode=request.mode,
            force_reload=True
        )
        logger.info("✅ Index rebuild completed successfully")
        print("✅ Index rebuild completed successfully")
        return {"message": "Index rebuilt successfully.", "status": "success"}
    except Exception as e:
        logger.error(f"❌ Error rebuilding index: {e}")
        print(f"❌ Error rebuilding index: {e}")
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")

# --- 4. Run Server ---
if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)
