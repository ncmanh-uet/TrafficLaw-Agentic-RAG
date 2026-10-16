# -*- coding: utf-8 -*-
"""
generator.py — Phase 3: Legal Answer Generator
==============================================
Grounded Vietnamese legal answer generation over retrieved chunks.

- Provider-agnostic via LangChain (OpenAI or Google Gemini).
- Strict prompt: Vietnamese only, cite Điều/Nghị định, refuse if context is
  insufficient, no outside knowledge.
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage, SystemMessage

from .prompt import SYSTEM_PROMPT, REFUSAL_PHRASE, REFUSAL_TOKEN

logger = logging.getLogger(__name__)


def _fold_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")


_SOURCE_CITATION_RE = re.compile(
    r"\[\s*S(\d+)(?:\s*[—–-]\s*[^\]\r\n]+)?\s*\]",
    flags=re.IGNORECASE,
)


def _source_numbers(text: str) -> list[int]:
    """Read both plain ``[S1]`` and canonical web ``[S1 — host]`` labels."""
    return [int(value) for value in _SOURCE_CITATION_RE.findall(text)]


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


def _is_refusal_answer(answer: str) -> bool:
    """Read the fixed refusal token, with legacy prose as a fallback."""
    if answer.strip() == REFUSAL_TOKEN:
        return True
    compact = re.sub(r"\s+", " ", _fold_text(answer)).strip().rstrip(".")
    expected = re.sub(r"\s+", " ", _fold_text(REFUSAL_PHRASE)).strip().rstrip(".")
    if compact == expected:
        return True
    terminal_patterns = (
        r"khong tim thay du (?:bang chung|thong tin|can cu).{0,240}$",
        r"khong co du (?:bang chung|thong tin|can cu).{0,240}(?:tra loi|xac dinh|ket luan).{0,120}$",
        r"khong co can cu de (?:xac dinh|tra loi|ket luan).{0,240}$",
    )
    return any(re.search(pattern, compact) for pattern in terminal_patterns)


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
    """LLM-backed generator for grounded Vietnamese legal answers."""

    def __init__(
        self,
        provider: str = "google",
        model: str | None = None,
        temperature: float = 0.0,
        api_key: str | None = None,
        max_tokens: int = 1024,
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
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=temperature,
                api_key=key,
                max_tokens=max_tokens,
            )
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
        invoke_kwargs = {}
        if self.provider == "google":
            invoke_kwargs["automatic_function_calling"] = {"disable": True}
        response = self.llm.invoke(messages, **invoke_kwargs)
        content = _response_content_to_text(getattr(response, "content", response)).strip()
        content = re.sub(r"^```(?:text)?\s*|\s*```$", "", content, flags=re.IGNORECASE).strip()
        content = re.sub(r"^(?:truy vấn|query)\s*:\s*", "", content, flags=re.IGNORECASE).strip()
        content = " ".join(content.splitlines()).strip()
        if not content or len(content) > 600 or _fold_text(content) == _fold_text(query):
            return None
        return content

    def generate(
        self,
        query: str,
        chunks: list[dict],
    ) -> dict:
        context = _build_context(chunks)

        frames = set()
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            frame = (
                meta.get("doc_id"),
                meta.get("dieu"),
                meta.get("khoan"),
                meta.get("diem"),
            )
            frames.add(frame)

        multi_frame = len(frames) >= 2
        multi_frame_hint = ""
        if multi_frame:
            multi_frame_hint = (
                "HỆ THỐNG ĐÃ PHÁT HIỆN NHIỀU LEGAL FRAME TRONG RETRIEVAL RESULT. Bắt buộc kiểm tra tất cả các Điều/Khoản phù hợp trước khi kết luận."
            )

        user_content = (
            f"NGỮ CẢNH:\n{context}\n\n"
            f"CÂU HỎI: {query}\n\n"
            f"{multi_frame_hint}\n\n"
            "Mỗi kết luận phải dẫn SOURCE_ID tồn tại trong ngữ cảnh theo dạng [S1]. "
            "Không được viết [Nguồn 1] hoặc tự tạo SOURCE_ID. Với nguồn nội bộ, "
            "ghi thêm Điều/Khoản/Điểm và đúng giá trị doc_id trong metadata của "
            "chính source đó; không dùng tên văn bản thay cho doc_id.\n\n"
            "Hãy trả lời dựa CHỈ trên ngữ cảnh trên, tuân thủ các quy tắc ở hệ thống. "
            f"Nếu thiếu dữ kiện cho bất kỳ ý nào được hỏi, chỉ trả về {REFUSAL_TOKEN}, "
            "không kèm giải thích hoặc citation."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]

        invoke_kwargs = {}
        if self.provider == "google":
            # This generator does not expose tools to Gemini. Explicitly
            # disabling SDK-side AFC avoids the generate_content AFC warning
            # and prevents an unnecessary remote-call loop.
            invoke_kwargs["automatic_function_calling"] = {"disable": True}
        response = self.llm.invoke(messages, **invoke_kwargs)
        response_content = getattr(response, "content", response)
        answer = self._normalize_generated_answer(
            _response_content_to_text(response_content).strip()
        )
        answer = self._canonicalize_internal_citation_doc_ids(answer, chunks)
        answer = self._canonicalize_web_citations(answer, chunks)

        refused = _is_refusal_answer(answer)
        sources, citation_issues = self._resolve_citations(answer, chunks)
        model_calls = 1

        # A formatting failure is not an evidence failure. Give the model one
        # bounded chance to rewrite the same grounded answer with resolvable
        # source IDs before the graph considers external search.
        if not refused and "NO_RESOLVABLE_CITATIONS" in citation_issues:
            repair_messages = [
                *messages,
                HumanMessage(
                    content=(
                        "Bản trả lời vừa tạo không có citation SOURCE_ID mà hệ thống có thể "
                        "đọc được. Hãy viết lại toàn bộ câu trả lời, giữ nguyên các kết luận "
                        "đã được ngữ cảnh hỗ trợ và gắn [S1], [S2]... ngay sau từng kết luận. "
                        "Với nguồn nội bộ, dùng riêng hai cặp ngoặc theo mẫu: "
                        "[S1] [Điều X, Khoản Y — doc_id]. Không gộp thành [S1, Điều X]. "
                        "Không thêm thông tin hoặc nguồn mới.\n\n"
                        "Nếu ngữ cảnh không đủ để trả lời các ý được hỏi, "
                        f"chỉ trả về {REFUSAL_TOKEN}, không kèm citation hay giải thích.\n\n"
                        f"BẢN CẦN SỬA:\n{answer}"
                    )
                ),
            ]
            repaired_response = self.llm.invoke(repair_messages, **invoke_kwargs)
            repaired_content = getattr(repaired_response, "content", repaired_response)
            repaired_answer = self._normalize_generated_answer(
                _response_content_to_text(repaired_content).strip()
            )
            repaired_answer = self._canonicalize_internal_citation_doc_ids(
                repaired_answer, chunks
            )
            repaired_answer = self._canonicalize_web_citations(repaired_answer, chunks)
            repaired_sources, repaired_issues = self._resolve_citations(repaired_answer, chunks)
            model_calls = 2
            if (
                _is_refusal_answer(repaired_answer)
                or repaired_sources
                or len(repaired_issues) < len(citation_issues)
            ):
                answer = repaired_answer
                sources = repaired_sources
                citation_issues = repaired_issues

        # A repair may legitimately conclude that the context is insufficient.
        # Recompute this structured state from the final selected answer.
        refused = _is_refusal_answer(answer)
        if refused:
            # Keep the control token internal; callers receive a Vietnamese
            # message plus the structured refusal used by the graph.
            answer = REFUSAL_PHRASE
            sources = []
            citation_issues = []

        return {
            "answer": answer,
            "sources": sources,
            "refused": refused,
            "answer_status": "INSUFFICIENT_EVIDENCE" if refused else "ANSWERED",
            "citation_valid": refused or not citation_issues,
            "citation_issues": citation_issues,
            "model_calls": model_calls,
            "model": f"{self.provider}/{self.model_name}",
        }

    @staticmethod
    def _normalize_generated_answer(answer: str) -> str:
        """Repair harmless citation/spacing variants without changing claims."""
        answer = re.sub(
            r"\[\s*Nguồn\s+(\d+)\s*\]",
            lambda match: f"[S{match.group(1)}]",
            answer,
            flags=re.IGNORECASE,
        )

        def split_source_group(match: re.Match) -> str:
            return " ".join(
                f"[S{number}]"
                for number in re.findall(r"S(\d+)", match.group(1), flags=re.IGNORECASE)
            )

        answer = re.sub(
            r"\[\s*(S\d+(?:\s*,\s*S\d+)+)\s*\]",
            split_source_group,
            answer,
            flags=re.IGNORECASE,
        )
        answer = re.sub(
            r"\b(Điều|Khoản|Điểm)(?=[0-9a-zA-ZđĐ])",
            r"\1 ",
            answer,
            flags=re.IGNORECASE,
        )

        def split_combined_citation(match: re.Match) -> str:
            content = match.group(1)
            segments = re.split(r";\s*(?=S\d+\s*,)", content, flags=re.IGNORECASE)
            normalized: list[str] = []
            for segment in segments:
                parsed = re.fullmatch(
                    r"\s*S(\d+)\s*,\s*((?:Điều|Khoản|Điểm)\b.+)\s*",
                    segment,
                    flags=re.IGNORECASE,
                )
                if not parsed:
                    return match.group(0)
                normalized.append(f"[S{parsed.group(1)}] [{parsed.group(2).strip()}]")
            return " ".join(normalized)

        answer = re.sub(
            r"\[\s*((?:S\d+\s*,\s*(?:Điều|Khoản|Điểm)\b[^\]]+))\]",
            split_combined_citation,
            answer,
            flags=re.IGNORECASE,
        )
        return answer

    @staticmethod
    def _canonicalize_internal_citation_doc_ids(
        answer: str,
        chunks: list[dict],
    ) -> str:
        """Attach the canonical metadata ``doc_id`` to every internal citation.

        Models sometimes omit the document suffix or emit a human-readable
        title instead.  SOURCE_ID is deterministic, so use it to resolve the
        cited chunk and render the exact doc_id without asking the model to
        reproduce identifiers.
        """
        pattern = re.compile(
            r"\[\s*S(\d+)\s*\](?P<gap>[ \t]*)(?P<legal>\[\s*Điều\b[^\]\r\n]*\])?",
            flags=re.IGNORECASE,
        )

        def replace(match: re.Match) -> str:
            source_number = int(match.group(1))
            index = source_number - 1
            if not 0 <= index < len(chunks):
                return match.group(0)

            metadata = chunks[index].get("metadata", {})
            source_type = str(metadata.get("source_type") or "internal").lower()
            doc_id = str(metadata.get("doc_id") or "").strip()
            if source_type == "web" or not doc_id:
                return match.group(0)

            legal = match.group("legal")
            if legal:
                location = legal[1:-1].strip()
                location = re.split(r"\s+[—–-]\s+", location, maxsplit=1)[0].strip()
                location = re.sub(
                    r"\s*,?\s*doc_id\s*:.*$",
                    "",
                    location,
                    flags=re.IGNORECASE,
                ).strip()
            else:
                dieu = metadata.get("dieu")
                if dieu in (None, ""):
                    return match.group(0)
                bits = [f"Điều {dieu}"]
                if metadata.get("khoan") not in (None, ""):
                    bits.append(f"Khoản {metadata['khoan']}")
                if metadata.get("diem") not in (None, ""):
                    bits.append(f"Điểm {metadata['diem']}")
                location = ", ".join(bits)

            return f"[S{source_number}] [{location} — {doc_id}]"

        return pattern.sub(replace, answer)

    @staticmethod
    def _canonicalize_web_citations(answer: str, chunks: list[dict]) -> str:
        """Render web citations from trusted metadata, never model-authored URLs."""
        rendered = answer
        for index, chunk in enumerate(chunks, 1):
            metadata = chunk.get("metadata", {})
            if str(metadata.get("source_type") or "internal").lower() != "web":
                continue

            url = str(metadata.get("url") or metadata.get("doc_id") or "").strip()
            hostname = _web_hostname(url)
            if hostname:
                replacement = f"[S{index} — {hostname}]({url})"
            else:
                replacement = f"[S{index} — nguồn web]"

            pattern = re.compile(
                rf"\[\s*S{index}(?:\s*[—–-]\s*[^\]\r\n]+)?\s*\]"
                r"(?:\([^\)\r\n]+\))?",
                flags=re.IGNORECASE,
            )
            rendered = pattern.sub(lambda _match, value=replacement: value, rendered)
        return rendered

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
    def _resolve_citations(
        cls,
        answer: str,
        chunks: list[dict],
    ) -> tuple[list[dict], list[str]]:
        """Resolve source IDs first, then legal citations as a compatibility fallback."""
        def _norm(v):
            if v is None:
                return None
            s = str(v).strip()
            return s.lower() or None

        sources: list[dict] = []
        issues: list[str] = []
        used_indexes: set[int] = set()

        # SOURCE_ID is the primary, unambiguous citation mechanism.
        referenced_ids = _source_numbers(answer)
        for source_number in referenced_ids:
            index = source_number - 1
            if not 0 <= index < len(chunks):
                issue = f"UNKNOWN_SOURCE_ID:S{source_number}"
                if issue not in issues:
                    issues.append(issue)
                continue
            if index not in used_indexes:
                used_indexes.add(index)
                sources.append(cls._source_record(index, chunks[index]))

        # Compatibility fallback: resolve explicit legal citations even when a
        # model omits SOURCE_ID. This also supports older stored answers.
        parsed: list[tuple[str, str | None, str | None, str | None]] = []
        for block in re.findall(r"\[([^\[\]]+)\]", answer):
            dieu_match = re.search(r"(?:Điều|Dieu)\s*([0-9]+)", block, re.IGNORECASE)
            if not dieu_match:
                continue
            khoan_match = re.search(r"(?:Khoản|Khoan)\s*([0-9]+[a-zA-ZđĐ]?)", block, re.IGNORECASE)
            diem_match = re.search(r"(?:Điểm|Diem)\s*([a-zA-ZđĐ0-9]+)", block, re.IGNORECASE)
            tail_parts = re.split(r"\s+[—–-]\s+", block, maxsplit=1)
            tail = tail_parts[1].strip().lower() if len(tail_parts) == 2 else ""
            parsed.append(
                (
                    tail,
                    _norm(dieu_match.group(1)),
                    _norm(khoan_match.group(1)) if khoan_match else None,
                    _norm(diem_match.group(1)) if diem_match else None,
                )
            )

        def _tail_matches_chunk(tail: str, chunk_did: str, chunk_name: str) -> bool:
            """Return True if the citation tail refers to this chunk's doc."""
            if not tail:
                # No tail parsed — accept any doc; Điều/Khoản/Điểm carry the
                # specificity. (Rare; the prompt asks for `— {doc_id}`.)
                return True
            tail_l = tail.lower()
            cdid = (chunk_did or "").lower()
            cname = (chunk_name or "").lower()
            if cdid and cdid in tail_l:
                return True
            if cdid and tail_l in cdid:
                return True
            # Number-slug match (e.g. "168/2024" inside both)
            slug_m = re.search(r"(\d+/\d{4})", tail_l)
            if slug_m and cdid and slug_m.group(1) in cdid:
                return True
            # Name match — require a reasonably long overlap to avoid spurious
            # hits like the word "Luật" alone.
            if cname:
                short = re.sub(r"\s*\([^)]*\)\s*$", "", cname).strip()
                if len(short) >= 8 and short in tail_l:
                    return True
            return False

        for index, c in enumerate(chunks):
            meta = c.get("metadata", {})
            if str(meta.get("source_type") or "internal").lower() == "web":
                continue
            cdid = meta.get("doc_id", "") or ""
            cname = meta.get("ten_van_ban", "") or ""
            c_dieu = _norm(meta.get("dieu"))
            c_khoan = _norm(meta.get("khoan"))
            c_diem = _norm(meta.get("diem"))

            for tail, p_dieu, p_khoan, p_diem in parsed:
                if p_dieu and c_dieu != p_dieu:
                    continue
                if p_khoan and c_khoan != p_khoan:
                    continue
                if p_diem and c_diem != p_diem:
                    continue
                if not _tail_matches_chunk(tail, cdid, cname):
                    continue

                if index not in used_indexes:
                    used_indexes.add(index)
                    sources.append(cls._source_record(index, c))
                break

        refused = _is_refusal_answer(answer)
        if not refused and not sources:
            issues.append("NO_RESOLVABLE_CITATIONS")
        if not refused:
            for issue in cls._citation_support_issues(answer, chunks):
                if issue not in issues:
                    issues.append(issue)
        return sources, issues

    @staticmethod
    def _citation_support_issues(answer: str, chunks: list[dict]) -> list[str]:
        """Validate that each source ID supports the claim unit citing it.

        This is deliberately conservative. It checks exact legal metadata and
        requires a direct action source when a claim names the red-light
        offence; a cross-reference table alone cannot establish the offence.
        """
        issues: list[str] = []

        def norm(value):
            if value is None:
                return None
            text = str(value).strip().lower()
            return text or None

        def legal_locations(unit: str) -> list[tuple[str, str | None, str | None, str]]:
            locations: list[tuple[str, str | None, str | None, str]] = []
            for block in re.findall(r"\[([^\[\]]+)\]", unit):
                dieu = re.search(r"(?:Điều|Dieu)\s*([0-9]+)", block, re.IGNORECASE)
                if not dieu:
                    continue
                khoan = re.search(r"(?:Khoản|Khoan)\s*([0-9]+[a-zA-ZđĐ]?)", block, re.IGNORECASE)
                diem = re.search(r"(?:Điểm|Diem)\s*([a-zA-ZđĐ0-9]+)", block, re.IGNORECASE)
                tail = re.split(r"\s+[—–-]\s+", block, maxsplit=1)
                locations.append(
                    (
                        norm(dieu.group(1)) or "",
                        norm(khoan.group(1)) if khoan else None,
                        norm(diem.group(1)) if diem else None,
                        tail[1].strip().lower() if len(tail) == 2 else "",
                    )
                )
            return locations

        def chunk_contains_location(
            chunk: dict,
            p_dieu: str,
            p_khoan: str | None,
            p_diem: str | None,
            tail: str,
        ) -> bool:
            """Match metadata exactly or verify a child location in parent text."""
            meta = chunk.get("metadata", {})
            c_dieu = norm(meta.get("dieu"))
            c_khoan = norm(meta.get("khoan"))
            c_diem = norm(meta.get("diem"))
            c_doc = str(meta.get("doc_id") or "").lower()
            c_name = str(meta.get("ten_van_ban") or meta.get("title") or "").lower()

            tail_matches = not tail
            if tail:
                if c_doc and (c_doc in tail or tail in c_doc):
                    tail_matches = True
                slug = re.search(r"\d+/\d{4}", tail)
                if slug and slug.group(0) in c_doc:
                    tail_matches = True
                short_name = re.sub(r"\s*\([^)]*\)\s*$", "", c_name).strip()
                if len(short_name) >= 8 and short_name in tail:
                    tail_matches = True

            if p_dieu != c_dieu or not tail_matches:
                return False
            if c_khoan is not None and p_khoan != c_khoan:
                return False
            if c_diem is not None and p_diem != c_diem:
                return False

            content = chunk.get("content", "")
            segment = content
            if p_khoan is not None and c_khoan is None:
                paragraph = re.search(
                    rf"(?m)^\s*{re.escape(p_khoan)}\.\s+",
                    content,
                )
                if not paragraph:
                    return False
                next_paragraph = re.search(
                    r"(?m)^\s*\d+[a-zA-ZđĐ]?\.\s+",
                    content[paragraph.end():],
                )
                end = (
                    paragraph.end() + next_paragraph.start()
                    if next_paragraph
                    else len(content)
                )
                segment = content[paragraph.start():end]
            elif p_khoan != c_khoan:
                return False

            if p_diem is not None and c_diem is None:
                if not re.search(
                    rf"(?:^|[\n;])\s*{re.escape(p_diem)}\)\s*",
                    segment,
                    re.IGNORECASE,
                ):
                    return False
            elif p_diem != c_diem:
                return False
            return True

        scope_indexes: list[int] = []

        def references_location(content: str, target: dict) -> bool:
            folded = _fold_text(content)
            khoan = _fold_text(str(target.get("khoan") or "")).strip()
            diem = _fold_text(str(target.get("diem") or "")).strip()
            if not khoan:
                return False
            for match in re.finditer(rf"\bkhoan\s+{re.escape(khoan)}\b", folded):
                prefix = folded[:match.start()].rsplit(";", 1)[-1]
                if not diem or re.search(rf"\bdiem\s+{re.escape(diem)}\b", prefix):
                    return True
                if not re.search(r"\bdiem\s+", prefix) or re.search(r"[,;]\s*$", prefix):
                    return True
            return False

        for line_number, unit in enumerate(answer.splitlines(), 1):
            if unit.strip().startswith("#"):
                scope_indexes = []
            source_numbers = _source_numbers(unit)
            if not source_numbers:
                continue
            locations = legal_locations(unit)
            valid_indexes = [number - 1 for number in source_numbers if 0 < number <= len(chunks)]

            for index in valid_indexes:
                chunk = chunks[index]
                meta = chunk.get("metadata", {})
                if str(meta.get("source_type") or "internal").lower() == "web":
                    continue
                matched = any(
                    chunk_contains_location(chunk, p_dieu, p_khoan, p_diem, tail)
                    for p_dieu, p_khoan, p_diem, tail in locations
                )
                if not matched:
                    issues.append(f"SOURCE_LEGAL_LOCATION_MISMATCH:S{index + 1}:L{line_number}")

            cross_reference_markers = (
                "vi pham quy dinh tai",
                "thuc hien hanh vi quy dinh tai",
            )
            def is_cross_reference(index: int) -> bool:
                content = _fold_text(chunks[index].get("content", ""))
                return any(marker in content for marker in cross_reference_markers)

            # Build a support chain inside the current Markdown section. A
            # later bullet may say "hành vi này" and cite only its consequence
            # table, provided that table really points back to a direct source
            # cited by an earlier bullet in the same section.
            available = list(dict.fromkeys([*scope_indexes, *valid_indexes]))
            reachable = {index for index in available if not is_cross_reference(index)}
            changed = True
            while changed:
                changed = False
                for index in available:
                    if index in reachable or not is_cross_reference(index):
                        continue
                    meta = chunks[index].get("metadata", {})
                    article = norm(meta.get("dieu"))
                    if any(
                        norm(chunks[parent].get("metadata", {}).get("dieu")) == article
                        and references_location(
                            chunks[index].get("content", ""),
                            chunks[parent].get("metadata", {}),
                        )
                        for parent in reachable
                    ):
                        reachable.add(index)
                        changed = True
            unsupported_cross_reference = any(
                is_cross_reference(index) and index not in reachable
                for index in valid_indexes
            )
            if unsupported_cross_reference:
                issues.append(f"CROSS_REFERENCE_WITHOUT_DIRECT_SOURCE:L{line_number}")

            scope_indexes.extend(index for index in valid_indexes if index not in scope_indexes)

        return issues

    @classmethod
    def _extract_cited_sources(cls, answer: str, chunks: list[dict]) -> list[dict]:
        """Backward-compatible wrapper used by existing evaluation code."""
        return cls._resolve_citations(answer, chunks)[0]
