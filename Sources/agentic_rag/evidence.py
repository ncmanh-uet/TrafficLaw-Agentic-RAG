"""Normalization, deduplication, and deterministic evidence verification."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit

from .config import AgentConfig
from .state import Evidence


_STOPWORDS = {
    "ai", "bao", "bi", "bị", "cac", "các", "cho", "co", "có", "cua",
    "của", "duoc", "được", "gi", "gì", "la", "là", "mot", "một", "nao",
    "nào", "nhung", "những", "o", "ở", "phai", "phải", "the", "thế",
    "thi", "thì", "theo", "toi", "tôi", "trong", "va", "và", "voi",
    "với", "xe",
}

def _tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFC", text.lower())
    return {
        token
        for token in re.findall(r"[\wđ]+", normalized, flags=re.UNICODE)
        if len(token) > 1 and token not in _STOPWORDS
    }


def _fold(text: str) -> str:
    """Lowercase Vietnamese text and remove accents for stable rule matching."""
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")


def _date_values(text: str) -> set[str]:
    """Extract valid, explicit calendar dates as ISO strings.

    Only full dates are used. A bare year or month is too ambiguous to reject
    otherwise useful legal evidence.
    """
    values: set[str] = set()
    patterns = (
        r"(?<!\d)(?P<day>[0-3]?\d)[/\.\-](?P<month>[01]?\d)[/\.\-](?P<year>(?:19|20)\d{2})(?!\d)",
        r"(?<!\d)(?P<year>(?:19|20)\d{2})-(?P<month>[01]?\d)-(?P<day>[0-3]?\d)(?!\d)",
        r"\bngày\s+(?P<day>[0-3]?\d)\s+tháng\s+(?P<month>[01]?\d)\s+năm\s+(?P<year>(?:19|20)\d{2})\b",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            try:
                parsed = date(
                    int(match.group("year")),
                    int(match.group("month")),
                    int(match.group("day")),
                )
            except ValueError:
                continue
            values.add(parsed.isoformat())
    return values


@dataclass(frozen=True)
class TemporalRequirement:
    mode: str | None
    dates: frozenset[str]


def temporal_requirement(query: str) -> TemporalRequirement:
    """Classify generic freshness and exact-event-time requirements."""
    folded = _fold(query)
    dates = frozenset(_date_values(query))
    freshness_markers = (
        "tinh den",
        "moi nhat",
        "hien nay",
        "hien tai",
        "hom nay",
        "den nay",
        "con hieu luc",
    )
    if any(marker in folded for marker in freshness_markers):
        return TemporalRequirement("as_of", dates)

    if dates:
        return TemporalRequirement("event", dates)
    return TemporalRequirement(None, dates)


def _evidence_date_values(item: Evidence) -> set[str]:
    metadata = item.get("metadata", {})
    fields = [item.get("title") or "", item.get("content", "")]
    for key in (
        "published_date",
        "date",
        "ngay_ban_hanh",
        "ngay_hieu_luc",
        "effective_date",
    ):
        value = metadata.get(key)
        if value not in (None, ""):
            fields.append(str(value))
    return _date_values("\n".join(fields))


def _references_location(content: str, khoan: object, diem: object | None) -> bool:
    """Detect a legal cross-reference to a selected Khoản/Điểm."""
    folded = _fold(content)
    khoan_text = _fold(str(khoan or "")).strip()
    diem_text = _fold(str(diem or "")).strip()
    if not khoan_text:
        return False
    for match in re.finditer(rf"\bkhoan\s+{re.escape(khoan_text)}\b", folded):
        if not diem_text:
            return True
        # A semicolon separates reference groups. Looking only inside the
        # current group avoids borrowing a point letter from the prior Khoản.
        prefix = folded[:match.start()].rsplit(";", 1)[-1]
        if re.search(rf"\bdiem\s+{re.escape(diem_text)}\b", prefix):
            return True
        # A reference to the whole paragraph ("khoản 10") covers every point
        # in that paragraph and therefore supports a second-hop consequence.
        if not re.search(r"\bdiem\s+", prefix) or re.search(r"[,;]\s*$", prefix):
            return True
    return False


def evidence_relevance(query: str, item: Evidence) -> float:
    """Return a provider-independent relevance score in the range 0..1.

    The score deliberately uses only the user query, evidence text, retrieval
    score and structural metadata.  It contains no rule for a particular
    offence, vehicle, article, or document.
    """
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0

    content_tokens = _tokens(item.get("content", ""))
    title_tokens = _tokens(item.get("title") or "")
    lexical = len(query_tokens & content_tokens) / len(query_tokens)
    title_overlap = len(query_tokens & title_tokens) / len(query_tokens)

    folded_query = _fold(query).strip()
    folded_text = _fold(f"{item.get('title') or ''} {item.get('content') or ''}")
    phrase_bonus = 0.12 if len(folded_query) >= 12 and folded_query in folded_text else 0.0

    raw_score = item.get("score")
    retrieval_signal = 0.0 if raw_score is None else min(max(float(raw_score), 0.0), 1.0)
    primary_bonus = 0.04 if not item.get("metadata", {}).get("is_sibling") else 0.0
    return min(
        1.0,
        0.68 * lexical
        + 0.16 * title_overlap
        + 0.12 * retrieval_signal
        + primary_bonus
        + phrase_bonus,
    )


def build_feedback_query(query: str, evidence: list[Evidence], max_chars: int = 600) -> str | None:
    """Build one generic pseudo-relevance-feedback query from retrieved text.

    Legal questions often use everyday wording while the corpus uses a formal
    statutory phrase.  This function extracts the most query-relevant lines
    from the first retrieval and feeds those corpus terms back to the same
    retriever.  Nothing is generated and no domain-specific synonym table is
    used, so the mechanism also works for procedures, definitions, fees, and
    conditions.
    """
    query_tokens = _tokens(query)
    if not query_tokens or not evidence:
        return None

    ranked_lines: list[tuple[float, str]] = []
    for item in evidence:
        base = evidence_relevance(query, item)
        for raw_line in re.split(r"[\r\n]+|(?<=[.;:])\s+", item.get("content", "")):
            line = re.sub(r"\s+", " ", raw_line).strip(" -\t")
            line_tokens = _tokens(line)
            overlap = len(query_tokens & line_tokens)
            if overlap < 2 or len(line) < 20:
                continue
            focus = overlap / max(1.0, len(line_tokens) ** 0.5)
            ranked_lines.append((focus + base, line))

    if not ranked_lines:
        return None

    selected: list[str] = []
    seen: set[str] = set()
    current_length = len(query) + 1
    for _, line in sorted(ranked_lines, key=lambda pair: pair[0], reverse=True):
        normalized = _fold(line)
        if normalized in seen or normalized in _fold(query):
            continue
        if current_length + len(line) + 2 > max_chars:
            continue
        seen.add(normalized)
        selected.append(line)
        current_length += len(line) + 2
        if len(selected) >= 3:
            break

    if not selected:
        return None
    return f"{query}\n" + "\n".join(selected)


def normalize_internal_chunk(chunk: Any) -> tuple[dict[str, Any], Evidence]:
    if hasattr(chunk, "to_dict"):
        raw = dict(chunk.to_dict())
    elif isinstance(chunk, dict):
        raw = dict(chunk)
    else:
        raw = {
            "id": getattr(chunk, "id", None),
            "score": getattr(chunk, "score", None),
            "content": getattr(chunk, "content", ""),
            "metadata": getattr(chunk, "metadata", {}),
        }

    metadata = dict(raw.get("metadata") or {})
    chunk_id = metadata.get("chunk_id") or raw.get("id")
    source = str(metadata.get("doc_id") or metadata.get("source_file") or chunk_id or "internal")
    score = raw.get("score")
    evidence: Evidence = {
        "content": str(raw.get("content") or ""),
        "source_type": "internal",
        "source": source,
        "url": None,
        "title": metadata.get("ten_van_ban") or metadata.get("dieu_title"),
        "score": float(score) if score is not None else None,
        "metadata": metadata,
    }
    return raw, evidence


def normalize_web_result(result: dict[str, Any]) -> tuple[dict[str, Any], Evidence]:
    raw = dict(result)
    url = str(raw.get("url") or "").strip() or None
    title = str(raw.get("title") or "").strip() or None
    content = str(raw.get("content") or raw.get("raw_content") or "").strip()
    score = raw.get("score")
    metadata = {
        "source_type": "web",
        "url": url,
        "title": title,
        "published_date": raw.get("published_date"),
    }
    evidence: Evidence = {
        "content": content,
        "source_type": "web",
        "source": url or title or "web",
        "url": url,
        "title": title,
        "score": float(score) if score is not None else None,
        "metadata": metadata,
    }
    return raw, evidence


def evidence_key(item: Evidence) -> str:
    if item["source_type"] == "web" and item.get("url"):
        parts = urlsplit(item["url"] or "")
        return "web:" + urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))
    meta = item.get("metadata", {})
    chunk_id = meta.get("chunk_id")
    if chunk_id not in (None, ""):
        return f"internal:{chunk_id}"
    location = (meta.get("doc_id"), meta.get("dieu"), meta.get("khoan"), meta.get("diem"))
    if any(value not in (None, "") for value in location):
        return "internal:" + "|".join(str(value or "") for value in location)
    return f"{item['source_type']}:{item['source']}:{hash(item['content'])}"


def merge_evidence(existing: Iterable[Evidence], additions: Iterable[Evidence]) -> tuple[list[Evidence], list[str]]:
    merged: list[Evidence] = []
    positions: dict[str, int] = {}
    for item in [*existing, *additions]:
        key = evidence_key(item)
        if not item.get("content", "").strip():
            continue
        if key not in positions:
            positions[key] = len(merged)
            merged.append(item)
            continue

        # A feedback retrieval may return a chunk that was previously attached
        # only as a sibling.  Preserve one copy, but promote the direct hit and
        # retain the strongest score observed across retrieval passes.
        index = positions[key]
        current = merged[index]
        current_sibling = bool(current.get("metadata", {}).get("is_sibling"))
        new_sibling = bool(item.get("metadata", {}).get("is_sibling"))
        if current_sibling and not new_sibling:
            replacement = item
        else:
            replacement = current
        scores = [score for score in (current.get("score"), item.get("score")) if score is not None]
        if scores:
            replacement = dict(replacement)
            replacement["score"] = max(scores)
        merged[index] = replacement
    return merged, sorted(positions)


def select_generation_evidence(query: str, evidence: list[Evidence], limit: int) -> list[Evidence]:
    """Bound context using relevance, source diversity, and legal references.

    Selection is intentionally agnostic to the subject of the question.  It
    first chooses relevant chunks from distinct document/article frames, then
    follows explicit Khoản/Điểm references for up to two hops, and finally
    fills the remaining context by score.
    """
    if limit <= 0 or not evidence:
        return []

    indexed = list(enumerate(evidence))
    ranked = sorted(
        indexed,
        key=lambda pair: (
            evidence_relevance(query, pair[1]),
            not bool(pair[1].get("metadata", {}).get("is_sibling")),
            -pair[0],
        ),
        reverse=True,
    )
    top_score = evidence_relevance(query, ranked[0][1])
    threshold = max(0.12, top_score * 0.55)
    eligible = [item for _, item in ranked if evidence_relevance(query, item) >= threshold]
    if len(eligible) < min(2, len(evidence)):
        eligible = [item for _, item in ranked[: min(2, len(ranked))]]

    seed_limit = min(limit, 12)
    seeds: list[Evidence] = []
    deferred: list[Evidence] = []
    seen_frames: set[tuple[object, object]] = set()
    for item in eligible:
        meta = item.get("metadata", {})
        frame = (meta.get("doc_id") or item.get("source"), meta.get("dieu"))
        if frame in seen_frames:
            deferred.append(item)
            continue
        seen_frames.add(frame)
        seeds.append(item)
        if len(seeds) >= seed_limit:
            break
    if len(seeds) < seed_limit:
        for item in deferred:
            seeds.append(item)
            if len(seeds) >= seed_limit:
                break

    priority = list(seeds)
    locations: list[tuple[object, object, object, object | None]] = []
    for item in seeds:
        meta = item.get("metadata", {})
        if item["source_type"] == "internal" and meta.get("khoan") not in (None, ""):
            locations.append(
                (meta.get("doc_id"), meta.get("dieu"), meta.get("khoan"), meta.get("diem"))
            )

    for _ in range(2):
        additions: list[Evidence] = []
        for item in evidence:
            if item["source_type"] != "internal":
                continue
            meta = item.get("metadata", {})
            if meta.get("diem") in (None, ""):
                continue
            if any(
                meta.get("doc_id") == doc_id
                and meta.get("dieu") == dieu
                and _references_location(item.get("content", ""), khoan, diem)
                for doc_id, dieu, khoan, diem in locations
            ):
                additions.append(item)
        priority.extend(additions)
        for item in additions:
            meta = item.get("metadata", {})
            location = (
                meta.get("doc_id"),
                meta.get("dieu"),
                meta.get("khoan"),
                meta.get("diem"),
            )
            if location not in locations:
                locations.append(location)

    # Do not pad the model context with unrelated chunks merely to reach the
    # configured maximum.  ``limit`` is a ceiling, not a target size.
    candidates = priority
    selected: list[Evidence] = []
    seen: set[str] = set()
    for item in candidates:
        key = evidence_key(item)
        if key in seen:
            continue
        seen.add(key)
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


@dataclass(frozen=True)
class EvidenceVerification:
    sufficient: bool
    issues: list[str]
    relevant_count: int
    query_coverage: float
    conflict_count: int


class DeterministicEvidenceVerifier:
    """Transparent baseline verifier, replaceable without changing the graph."""

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    def verify(self, query: str, evidence: list[Evidence]) -> EvidenceVerification:
        if not evidence:
            return EvidenceVerification(False, ["NO_EVIDENCE"], 0, 0.0, 0)

        query_tokens = _tokens(query)
        temporal = temporal_requirement(query)
        covered: set[str] = set()
        relevant_count = 0
        has_quality_source = False
        relevant_items: list[Evidence] = []

        for item in evidence:
            content_tokens = _tokens(item.get("content", "")) | _tokens(item.get("title") or "")
            overlap = query_tokens & content_tokens
            covered.update(overlap)
            lexical_relevance = len(overlap) / max(1, len(query_tokens))
            score = item.get("score")
            score_relevant = score is not None and score >= self.config.min_relevance_score
            if lexical_relevance >= self.config.min_query_coverage or score_relevant:
                relevant_count += 1
                relevant_items.append(item)
                if item["source_type"] == "internal" or self._is_preferred_web_source(item):
                    has_quality_source = True

        coverage = len(covered) / max(1, len(query_tokens))
        conflicts = self._find_conflicts(evidence)
        issues: list[str] = []
        if relevant_count < self.config.min_relevant_evidence:
            issues.append("LOW_RELEVANCE")
        if coverage < self.config.min_query_coverage:
            issues.append("LOW_QUERY_COVERAGE")
        if not has_quality_source:
            issues.append("NO_AUTHORITATIVE_SOURCE")
        if temporal.mode == "as_of" and not any(
            item["source_type"] == "web" for item in relevant_items
        ):
            # A static local corpus can establish what a document says, but it
            # cannot by itself prove that no newer external fact or instrument
            # exists at the user's requested cutoff.
            issues.append("FRESHNESS_UNVERIFIED")
        if temporal.mode == "event" and temporal.dates and not any(
            temporal.dates & _evidence_date_values(item) for item in relevant_items
        ):
            # Topical overlap is insufficient for a dated operational event.
            # Evidence about the same place or event type on another day must
            # not be used to answer the requested day.
            issues.append("TEMPORAL_MISMATCH")
        if conflicts:
            issues.append("CONFLICTING_EVIDENCE")

        return EvidenceVerification(
            sufficient=not issues,
            issues=issues,
            relevant_count=relevant_count,
            query_coverage=coverage,
            conflict_count=len(conflicts),
        )

    def _is_preferred_web_source(self, item: Evidence) -> bool:
        host = urlsplit(item.get("url") or "").netloc.lower().split(":", 1)[0]
        return any(host == domain or host.endswith("." + domain) for domain in self.config.preferred_web_domains)

    @staticmethod
    def _find_conflicts(evidence: list[Evidence]) -> list[str]:
        """Detect explicit conflict markers or incompatible values for one legal key."""
        grouped: dict[str, set[str]] = {}
        for item in evidence:
            meta = item.get("metadata", {})
            if meta.get("conflicting") is True:
                grouped.setdefault("explicit", set()).add(item.get("content", ""))
            key = meta.get("conflict_key")
            value = meta.get("conflict_value")
            if key is not None and value is not None:
                grouped.setdefault(str(key), set()).add(str(value))

        return [key for key, values in grouped.items() if key == "explicit" or len(values) > 1]
