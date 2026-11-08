"""Grounded Vietnamese legal answers over internal and web evidence."""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage

from .prompt import SYSTEM_PROMPT, REFUSAL_PHRASE, REFUSAL_TOKEN

logger = logging.getLogger(__name__)
_SOURCE_CITATION_RE = re.compile(
    r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)(?:\s*[—–-]\s*[^\]\r\n]+)?\s*\]",
    re.IGNORECASE,
)


def _source_numbers(text: str) -> list[int]:
    return [int(number) for group in _SOURCE_CITATION_RE.findall(text)
            for number in re.findall(r"S(\d+)", group, re.IGNORECASE)]


def _is_refusal_answer(answer: str) -> bool:
    return answer.strip() in {REFUSAL_TOKEN, REFUSAL_PHRASE}



def _web_hostname(url: str | None) -> str | None:
    """Return a safe display hostname only for HTTP(S) URLs."""
    try:
        parsed = urlsplit(str(url or "").strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return None
    host = parsed.hostname.lower().rstrip(".")
    return host[4:] if host.startswith("www.") else host


def _response_content_to_text(content) -> str:
    """Normalize LangChain text or multimodal content blocks to plain text.

    Newer chat-model integrations may return ``response.content`` as a list of
    typed blocks instead of a single string. The legal answer generator only
    consumes text, so retain text blocks and ignore non-text payloads.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for block in content:
            if isinstance(block, str):
                texts.append(block)
                continue
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    texts.append(text)
                continue
            text = getattr(block, "text", None)
            if isinstance(text, str):
                texts.append(text)
        return "\n".join(texts)
    return str(content)


def _format_chunk(i: int, chunk: dict) -> str:
    """Render a chunk with a stable source id the model can cite exactly."""
    meta: dict = chunk.get("metadata", {})
    doc_id = meta.get("doc_id", "?")
    ten_van_ban = meta.get("ten_van_ban", "")
    dieu = meta.get("dieu", "")
    dieu_title = meta.get("dieu_title", "")
    khoan = meta.get("khoan")
    diem = meta.get("diem")
    is_sibling = bool(meta.get("is_sibling"))
    source_type = str(meta.get("source_type") or "internal").lower()
    source_id = f"S{i}"

    loc_bits = [f"Điều {dieu}"] if dieu else []
    if khoan:
        loc_bits.append(f"Khoản {khoan}")
    if diem:
        loc_bits.append(f"Điểm {diem}")
    location = " · ".join(loc_bits) if loc_bits else ""

    if source_type == "web":
        url = meta.get("url") or doc_id
        header = f"[SOURCE_ID={source_id}] [WEB] {ten_van_ban or 'Nguồn web'}"
        if url:
            header += f" | URL: {url}"
    else:
        role = "SIBLING" if is_sibling else "PRIMARY"
        header = f"[SOURCE_ID={source_id}] [INTERNAL:{role}] {ten_van_ban} ({doc_id})"
    if location:
        header += f" · {location}"
    if dieu_title:
        header += f"\n{dieu_title}"

    content = chunk.get("content", "")
    return f"{header}\n{content}"


def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "(Không có ngữ cảnh nào được truy xuất.)"
    return "\n\n---\n\n".join(_format_chunk(i, c) for i, c in enumerate(chunks, 1))


class LegalAnswerGenerator:
    """Generate answers, expand retrieval queries, and expose source metadata."""

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
        temperature: float | None = 0.0,
        api_key: str | None = None,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ):
        self.provider = provider.lower()
        self.temperature = temperature
        self.max_tokens = max_tokens

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI provider requires OPENAI_API_KEY env var or api_key arg."
                )
            self.model_name = model or "gpt-4o-mini"
            options = {"model": self.model_name, "api_key": key, "max_tokens": max_tokens}
            if self.model_name.startswith("gpt-6-luna") or reasoning_effort is not None:
                options.update(use_responses_api=True, reasoning_effort=reasoning_effort or "none")
            else:
                options["temperature"] = temperature
            self.llm = ChatOpenAI(**options)
        elif self.provider == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "Google provider requires GOOGLE_API_KEY env var or api_key arg."
                )
            self.model_name = model or "gemini-1.5-flash"
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=temperature,
                google_api_key=key,
                max_output_tokens=max_tokens,
            )
        else:
            raise ValueError(
                f"Unsupported provider '{provider}'. Use 'openai' or 'google'."
            )

        logger.info(f"LegalAnswerGenerator: {self.provider} / {self.model_name}")

    def expand_query(self, query: str) -> str | None:
        """Rewrite a user question as one corpus-oriented retrieval query.

        The instruction is deliberately domain-general: it may translate
        everyday wording to formal legal language, but may not add an answer,
        an article number, a penalty value, or a narrower scope.
        """
        messages = [
            SystemMessage(
                content=(
                    "Bạn tối ưu truy vấn tìm kiếm trong kho văn bản pháp luật giao thông. "
                    "Hãy viết đúng MỘT truy vấn tiếng Việt ngắn gọn dùng thuật ngữ pháp lý "
                    "có khả năng xuất hiện nguyên văn trong tài liệu. Giữ nguyên mọi đối tượng, "
                    "điều kiện và phạm vi mà người dùng nêu; nếu câu hỏi tổng quát thì truy vấn "
                    "cũng phải tổng quát. Không trả lời câu hỏi, không thêm số Điều, tên văn bản, "
                    "mức tiền hoặc dữ kiện pháp lý. Chỉ xuất truy vấn, không giải thích."
                )
            ),
            HumanMessage(content=query),
        ]
        content = self._invoke(messages)
        content = re.sub(r"^```(?:text)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"^(?:truy vấn|query)\s*:\s*", "", content, flags=re.IGNORECASE).strip()
        content = " ".join(content.splitlines()).strip()
        if not content or len(content) > 600 or content.casefold() == query.casefold():
            return None
        return content

    def _invoke(self, messages) -> str:
        kwargs = {"automatic_function_calling": {"disable": True}} if self.provider == "google" else {}
        response = self.llm.invoke(messages, **kwargs)
        return _response_content_to_text(getattr(response, "content", response)).strip()

    def generate(self, query: str, chunks: list[dict]) -> dict:
        """Generate once; resolve cited source IDs without rewriting legal claims."""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"NGỮ CẢNH:\n{_build_context(chunks)}\n\nCÂU HỎI: {query}"),
        ]
        answer = self._invoke(messages)
        refused = _is_refusal_answer(answer)
        if refused:
            answer, sources, issues = REFUSAL_PHRASE, [], []
        else:
            answer = self._canonicalize_web_citations(answer, chunks)
            sources, issues = self._resolve_citations(answer, chunks)
        return {
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "answer_status": "INSUFFICIENT_EVIDENCE" if refused else "ANSWERED",
            "citation_valid": refused or not issues,
            "citation_issues": issues,
            "model_calls": 1,
            "model": f"{self.provider}/{self.model_name}",
        }

    @staticmethod
    def _canonicalize_web_citations(answer: str, chunks: list[dict]) -> str:
        """Render web citations from trusted metadata, never model-authored URLs."""
        def render(match):
            labels, has_web = [], False
            for number in _source_numbers(match.group()):
                meta = chunks[number - 1].get("metadata", {}) if 1 <= number <= len(chunks) else {}
                if str(meta.get("source_type") or "internal").lower() == "web":
                    has_web = True
                    url = str(meta.get("url") or meta.get("doc_id") or "").strip()
                    host = _web_hostname(url)
                    labels.append(f"[S{number} — {host}]({url})" if host else f"[S{number} — nguồn web]")
                else:
                    labels.append(f"[S{number}]")
            return " ".join(labels) if has_web else match.group()

        pattern = re.compile(_SOURCE_CITATION_RE.pattern + r"(?:\([^\)\r\n]+\))?", re.IGNORECASE)
        return pattern.sub(render, answer)

    @staticmethod
    def _source_record(index: int, chunk: dict) -> dict:
        meta = chunk.get("metadata", {})
        source_type = str(meta.get("source_type") or "internal").lower()
        title = meta.get("ten_van_ban", "") or meta.get("title", "") or ""
        url = meta.get("url")
        return {
            "source_id": f"S{index + 1}",
            "source_type": source_type,
            "doc_id": meta.get("doc_id", "") or "",
            "title": title,
            "ten_van_ban": title,
            "url": url,
            "domain": _web_hostname(url) if source_type == "web" else None,
            "dieu": meta.get("dieu"),
            "khoan": meta.get("khoan"),
            "diem": meta.get("diem"),
            "chunk_id": meta.get("chunk_id", ""),
        }

    @classmethod
    def _resolve_citations(cls, answer: str, chunks: list[dict]) -> tuple[list[dict], list[str]]:
        """Check source identity only; semantic grounding belongs to the judge."""
        sources, issues = [], []
        for number in dict.fromkeys(_source_numbers(answer)):
            if 1 <= number <= len(chunks):
                sources.append(cls._source_record(number - 1, chunks[number - 1]))
            else:
                issues.append(f"UNKNOWN_SOURCE_ID:S{number}")
        if not sources and not _is_refusal_answer(answer):
            issues.append("NO_RESOLVABLE_CITATIONS")
        return sources, issues
