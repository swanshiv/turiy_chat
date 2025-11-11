from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from langchain_core.documents import Document


@dataclass
class ChunkedDocument:
    page_content: str
    metadata: Dict[str, Any]

    def to_document(self) -> Document:
        return Document(page_content=self.page_content, metadata=self.metadata)

    @classmethod
    def from_document(cls, doc: Document) -> ChunkedDocument:
        return cls(page_content=doc.page_content, metadata=dict(doc.metadata))

