from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import matplotlib  # type: ignore

    matplotlib.use("Agg")  # type: ignore[attr-defined]
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    matplotlib = None
    plt = None

import pandas as pd
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

TOOL_DEFINITIONS: Sequence[Dict[str, str]] = [
    {
        "name": "faq_search",
        "description": "Use for metadata questions about the report itself (contact info, reporting period, what is GRI, etc.).",
    },
    {
        "name": "hybrid_search_tool",
        "description": "Use for questions that require contextual understanding, summaries, comparisons, or detailed answers sourced from the document body.",
    },
    {
        "name": "table_to_graph_tool",
        "description": "Use only when the user explicitly requests a chart, plot, graph, or visualization derived from tabular data.",
    },
]


@dataclass
class GraphResponse:
    answer: str
    chart: Dict[str, str]
    sources: List[Document]

def faq_search_tool(
    query: str,
    faq_lookup: Callable[[], Dict[str, str]],
) -> Optional[str]:
    entries = faq_lookup()
    normalized_query = ToolOrchestrator._normalize_query(query)
    normalized_entries = {
        ToolOrchestrator._normalize_query(key): value for key, value in entries.items()
    }
    match_key = ToolOrchestrator._match_faq_key(normalized_query, normalized_entries)
    if match_key:
        return normalized_entries[match_key]
    return None


def _parse_markdown_table(markdown: str) -> pd.DataFrame:
    lines = [line.strip() for line in markdown.strip().splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("Table markdown does not contain enough rows")

    header_line = lines[0]
    headers = [cell.strip() or f"Column {idx+1}" for idx, cell in enumerate(header_line.strip("|").split("|"))]

    def _is_separator(row: str) -> bool:
        content = row.strip().strip("|").replace("-", "").replace(":", "").replace(" ", "")
        return content == ""

    rows = []
    for row in lines[1:]:
        if _is_separator(row):
            continue
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        if not cells:
            continue
        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        rows.append(cells[: len(headers)])

    if not rows:
        raise ValueError("Table markdown has no data rows")

    df = pd.DataFrame(rows, columns=headers)
    return df


def _coerce_numeric(series: pd.Series) -> pd.Series:
    def _convert(value):
        if value is None:
            return None
        text = str(value).strip()
        if text == "" or text.lower() in {"n/a", "na", "none", "null"}:
            return None
        text = text.replace(",", "").replace("%", "")
        try:
            return float(text)
        except ValueError:
            return None

    return series.apply(_convert)


def _generate_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> Dict[str, str]:
    x_series = df[x_col]
    y_series = df[y_col]
    if isinstance(x_series, pd.DataFrame):
        x_series = x_series.iloc[:, 0]
    if isinstance(y_series, pd.DataFrame):
        y_series = y_series.iloc[:, 0]

    x_values = x_series.astype(str).tolist()
    y_values = pd.to_numeric(y_series, errors="coerce").tolist()
    if len(x_values) > 12:
        x_values = x_values[:12]
        y_values = y_values[:12]
    if len(y_values) < len(x_values):
        y_values.extend([0.0] * (len(x_values) - len(y_values)))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x_values, y_values, color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {
        "mime_type": "image/png",
        "data": encoded,
        "title": title,
        "x_label": x_col,
        "y_label": y_col,
    }


def table_to_graph_tool(query: str, documents: Iterable[Document]) -> Optional[GraphResponse]:
    if plt is None:
        logger.warning("Matplotlib is not available; unable to generate charts.")
        return None
    table_documents = [doc for doc in documents if doc.metadata.get("chunk_type") == "table"]
    if not table_documents:
        logger.info("No table chunks found in retrieved documents; cannot generate chart.")
        return None

    for doc in table_documents:
        try:
            df = _parse_markdown_table(doc.page_content)
        except Exception as exc:
            logger.debug("Skipping table due to parse failure: %s", exc)
            continue

        numeric_candidates = {}
        for column in df.columns:
            converted = _coerce_numeric(df[column])
            if converted.notnull().sum() >= 2:
                numeric_candidates[column] = converted

        if not numeric_candidates:
            continue

        x_candidates = [col for col in df.columns if col not in numeric_candidates]
        if x_candidates:
            x_col = x_candidates[0]
            df[x_col] = df[x_col].astype(str)
        else:
            x_col = df.columns[0]
            df[x_col] = df[x_col].astype(str)
            numeric_candidates.pop(x_col, None)
        if not numeric_candidates:
            continue

        y_col = next(iter(numeric_candidates.keys()))
        df[y_col] = numeric_candidates[y_col]
        df = df.dropna(subset=[y_col])
        if df.empty:
            continue

        file_name = doc.metadata.get("file_name", "the document")
        page = doc.metadata.get("page")
        title = f"{y_col} by {x_col}" if y_col != x_col else f"Chart for {file_name}"
        chart_df = df[[x_col, y_col]].copy()
        chart_payload = _generate_bar_chart(chart_df, x_col, y_col, title)
        description = f"Generated from {file_name}{f' (page {page})' if page else ''}."
        answer = (
            f"Here is a bar chart visualizing {y_col} by {x_col} using data from {file_name}"
            f"{f' (page {page})' if page else ''}."
        )
        chart_payload["description"] = description
        return GraphResponse(answer=answer, chart=chart_payload, sources=[doc])

    logger.info("Unable to derive numeric data from available tables; falling back to text answer.")
    return None


@dataclass
class ToolOrchestrator:
    llm: object
    tool_definitions: Sequence[Dict[str, str]] = field(
        default_factory=lambda: tuple(TOOL_DEFINITIONS)
    )
    faq_lookup: Optional[Callable[[], Dict[str, str]]] = None

    @staticmethod
    def _normalize_query(query: str) -> str:
        normalized = query.strip().lower()
        normalized = re.sub(r"[\s\u00a0]+", " ", normalized)
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    @staticmethod
    def _looks_like_graph_request(normalized_query: str) -> bool:
        if not normalized_query:
            return False
        query_with_bounds = f" {normalized_query} "
        graph_phrases = (
            " graph ",
            " chart ",
            " plot ",
            " visualize ",
            " visualise ",
            " visualization ",
            " visualisation ",
            " bar chart ",
            " line chart ",
            " scatter plot ",
            " draw chart ",
        )
        return any(phrase in query_with_bounds for phrase in graph_phrases)

    @staticmethod
    def _extract_tool_from_text(raw_text: str, valid_names: Sequence[str]) -> Optional[str]:
        if not raw_text:
            return None
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                return None
            candidate = match.group(0)
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                return None
        tool_name = payload.get("tool")
        if isinstance(tool_name, str) and tool_name in valid_names:
            return tool_name
        return None

    @staticmethod
    def _match_faq_key(
        normalized_query: str, normalized_entries: Dict[str, str]
    ) -> Optional[str]:
        if not normalized_query or not normalized_entries:
            return None
        if normalized_query in normalized_entries:
            return normalized_query

        for key in normalized_entries:
            if key in normalized_query or normalized_query in key:
                return key

        query_tokens = normalized_query.split()
        if query_tokens:
            query_token_set = set(query_tokens)
            for key in normalized_entries:
                key_tokens = key.split()
                if not key_tokens:
                    continue
                overlap = len(query_token_set & set(key_tokens)) / len(query_token_set)
                if overlap >= 0.75:
                    return key

        best_key = None
        best_score = 0.0
        for key in normalized_entries:
            score = SequenceMatcher(None, normalized_query, key).ratio()
            if score > best_score:
                best_score = score
                best_key = key
        if best_key and best_score >= 0.75:
            return best_key
        return None

    def select_tool(self, query: str) -> str:
        normalized = self._normalize_query(query)
        faq_entries: Dict[str, str] = {}
        if self.faq_lookup:
            try:
                raw_entries = self.faq_lookup()
                faq_entries = {
                    self._normalize_query(key): value for key, value in raw_entries.items()
                }
            except Exception as exc:
                logger.warning("Unable to load FAQ entries (%s); defaulting to router.", exc)
            else:
                if normalized in faq_entries:
                    logger.info("FAQ entry matched directly; routing to faq_search (key=%s).", normalized)
                    return "faq_search"
                match_key = self._match_faq_key(normalized, faq_entries)
                if match_key:
                    logger.info(
                        "FAQ fuzzy match succeeded (key=%s); routing to faq_search.", match_key
                    )
                    return "faq_search"

        if self._looks_like_graph_request(normalized):
            logger.info("Graph intent detected heuristically; routing to table_to_graph_tool.")
            return "table_to_graph_tool"

        descriptions = "\n".join(
            f"- {tool['name']}: {tool['description']}" for tool in self.tool_definitions
        )
        prompt = (
            "You are a tool router. Select the best tool for answering the user's question.\n"
            f"{descriptions}\n"
            'Respond with a JSON object like {"tool": "tool_name", "reason": "short reason"}.\n'
            f"Question: {query}"
        )
        valid_names = {tool["name"] for tool in self.tool_definitions}
        router_prompts = [
            prompt,
            (
                f"{prompt}\nRemember: respond with JSON only and include the 'tool' field."
            ),
        ]
        for attempt, router_prompt in enumerate(router_prompts, start=1):
            try:
                response = self.llm.invoke(router_prompt)
            except Exception as exc:
                logger.warning(
                    "Tool routing invocation failed on attempt %s (%s).", attempt, exc
                )
                continue

            content = getattr(response, "content", response)
            tool_name = self._extract_tool_from_text(content, valid_names)
            if tool_name:
                logger.debug("Tool router selected %s on attempt %s.", tool_name, attempt)
                return tool_name
            logger.warning(
                "Unable to parse tool router response on attempt %s: %s", attempt, content
            )

        logger.warning("Tool routing failed after retries. Falling back heuristics before hybrid search.")
        if faq_entries:
            match_key = self._match_faq_key(normalized, faq_entries)
            if match_key:
                logger.info(
                    "Fallback FAQ match succeeded after retries (key=%s); routing to faq_search.",
                    match_key,
                )
                return "faq_search"
        if self._looks_like_graph_request(normalized):
            logger.info("Fallback graph heuristic triggered; routing to table_to_graph_tool.")
            return "table_to_graph_tool"
        return "hybrid_search_tool"


def synthesize_answer(
    llm: object,
    query: str,
    documents: Iterable[Document],
    chat_history: Sequence[Tuple[str, str]],
) -> str:
    context_sections: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        doc.metadata["context_index"] = idx
        source = doc.metadata.get("file_name", doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_start") or doc.metadata.get("page")
        context_sections.append(
            f"[{idx}] Source: {source} (page {page})\n{doc.page_content}"
        )

    context_block = "\n\n".join(context_sections)
    history_text = "\n".join(f"User: {u}\nAssistant: {a}" for u, a in chat_history[-3:])

    synthesis_prompt = f"""
You are a meticulous analyst. Answer the user's question strictly using the provided context.

Rules:
- Include citations using [index] that reference the relevant context snippet.
- Do not use external knowledge. If the context lacks the answer, say you could not find it.
- Be concise and factual.

Context:
{context_block}

Recent conversation:
{history_text}

Question: {query}

Answer:
"""
    draft_response = llm.invoke(synthesis_prompt)
    draft_answer = getattr(draft_response, "content", draft_response)

    refinement_prompt = f"""
You are reviewing the draft answer for accuracy and citation fidelity.

Requirements:
- Ensure each statement is supported by the provided context and keeps the correct citation markers [index].
- Remove unsupported claims or clearly mark when information is unavailable.
- Keep the answer concise and factual.
- Respond with the refined answer text only. Do not add commentary, headers, or analysis outside the final answer.

Context:
{context_block}

Draft Answer:
{draft_answer}

Refined Answer:
"""
    refined_response = llm.invoke(refinement_prompt)
    refined_answer = getattr(refined_response, "content", refined_response)
    if isinstance(refined_answer, str):
        parts = re.split(r"(?i)refined answer\s*:\s*", refined_answer, maxsplit=1)
        if len(parts) == 2:
            refined_answer = parts[1]
        refined_answer = refined_answer.strip()
    return refined_answer


def maybe_generate_clarification(
    llm: object,
    query: str,
    documents: Sequence[Document],
    max_examples: int = 5,
) -> Optional[str]:
    snippet_summary = "\n".join(
        f"- {doc.metadata.get('file_name', 'unknown')} (page {doc.metadata.get('page_start') or doc.metadata.get('page')}): {doc.page_content[:200].replace(chr(10), ' ')}"
        for doc in documents[:max_examples]
    )

    clarification_prompt = f"""
Determine whether the user's question is ambiguous and requires clarification.

Question: {query}

Available snippets:
{snippet_summary}

Respond with JSON like {{"clarify": true/false, "question": "clarifying question or empty"}}.
Only set clarify=true when the question genuinely has multiple possible interpretations.
"""
    try:
        response = llm.invoke(clarification_prompt)
        content = getattr(response, "content", response)
        data = json.loads(content)
        if data.get("clarify") and data.get("question"):
            return data["question"]
    except Exception as exc:
        logger.debug("Clarification evaluation failed: %s", exc)
    return None

