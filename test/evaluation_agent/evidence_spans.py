"""Deterministic, source-bound evidence addresses for the Faithfulness judge.

Every published text is an unchanged slice of one supplied context. Legal
locations are structural addresses, not an assertion that a claim is entailed.
Ambiguous quoted amendments and unrecognised text remain usable evidence, but
are deliberately not assigned a guessed legal child address.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import json
import re
import unicodedata

from .legal_references import LegalLocation


@dataclass(frozen=True)
class EvidenceSpan:
    evidence_id: str
    source_id: str
    context_index: int
    start: int
    end: int
    text: str
    location: LegalLocation | None
    kind: str
    parent_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceIndex:
    spans: dict[str, EvidenceSpan]
    prompt_context: str


_WRAPPER = re.compile(r"^\[SOURCE_ID=([^\]\r\n]+)\]")
_DOC_ID = re.compile(r"\b(\d+/\d{4}/[\wĐđ-]+)\b")
_CLAUSE = re.compile(r"(?m)^[ \t]*(\d+[a-zđ]?)\.[ \t]+", re.IGNORECASE)
_POINT = re.compile(r"(?<![\w(])([a-zđ])\)(?=\s)", re.IGNORECASE)
_TABLE_LINE = re.compile(r"^[ \t]*\|.*\|[ \t]*(?:\r?\n)?$")


def _fold(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFD", text.casefold())
                   if unicodedata.category(char) != "Mn").replace("đ", "d")


def _location(source: dict, context: str) -> LegalLocation | None:
    if source.get("source_type") == "web":
        return None
    first_line = context.split("\n", 1)[0]
    doc_id = source.get("doc_id")
    if not doc_id:
        match = _DOC_ID.search(first_line)
        doc_id = match.group(1) if match else None
    article = source.get("dieu")
    paragraph = source.get("khoan")
    point = source.get("diem")
    # Only infer from the outer source header, never an inline reference or a
    # provision quoted inside an amendment.
    if article is None:
        match = re.search(r"\bĐiều\s+(\d+[a-zđ]?)\b", first_line, re.IGNORECASE)
        article = match.group(1) if match else None
    if paragraph is None:
        match = re.search(r"(?:·|\|)\s*Khoản\s+(\d+[a-zđ]?)\b", first_line,
                          re.IGNORECASE)
        paragraph = match.group(1) if match else None
    if point is None:
        match = re.search(r"(?:·|\|)\s*Điểm\s+([a-zđ])\b", first_line,
                          re.IGNORECASE)
        point = match.group(1) if match else None
    if not doc_id or article is None:
        return None
    loc = LegalLocation(str(doc_id), str(article),
                        str(paragraph) if paragraph is not None else None,
                        str(point) if point is not None else None)
    return loc if loc.valid else None


def _child(location: LegalLocation | None, paragraph: str | None = None,
           point: str | None = None) -> LegalLocation | None:
    if location is None:
        return None
    return LegalLocation(location.doc_id, location.article, paragraph, point)


def _point_definitions(text: str, start: int, end: int) -> list[re.Match]:
    definitions = []
    for match in _POINT.finditer(text, start, end):
        prefix = text[start:match.start()]
        # Mentions and reference lists are not point definitions. In
        # particular, "các điểm a), b)" must not create two child provisions.
        if re.search(r"\b(?:diem|muc|khoan)\s+$", _fold(prefix)):
            continue
        if prefix.rstrip().endswith((",", "(")):
            continue
        definitions.append(match)
    return definitions


def _quoted_amendment(text: str) -> bool:
    folded = _fold(text)
    return ("sua doi" in folded or "bo sung" in folded) and (
        "“" in text or bool(re.search(r'(?m)^\s*"', text)))


def build_evidence_index(contexts: Sequence[str],
                         sources: Sequence[dict] | None = None) -> EvidenceIndex:
    """Address exact source passages without accepting model-generated quotes.

    ``sources`` is the corresponding saved generation source mapping. Where it
    is absent, the outer ``SOURCE_ID`` header may supply a legal address. A
    source with no trustworthy address remains ordinary text, not a fabricated
    provision. IDs are stable for the same ordered input contexts and mapping.
    """
    if sources is not None and len(sources) != len(contexts):
        raise ValueError("evidence_source_context_length_mismatch")
    spans: dict[str, EvidenceSpan] = {}
    counts: dict[str, int] = {}

    for context_index, context in enumerate(contexts):
        source = sources[context_index] if sources is not None else {}
        wrapper = _WRAPPER.match(context)
        wrapper_id = wrapper.group(1) if wrapper else None
        mapped_id = source.get("source_id")
        if mapped_id and wrapper_id and str(mapped_id) != wrapper_id:
            raise ValueError("evidence_source_context_id_mismatch")
        source_id = str(mapped_id or wrapper_id or f"S{context_index + 1}")
        location = _location(source, context)

        def add(start: int, end: int, loc: LegalLocation | None, kind: str,
                parents: tuple[str, ...] = ()) -> str | None:
            while start < end and context[start].isspace():
                start += 1
            while end > start and context[end - 1].isspace():
                end -= 1
            if start == end:
                return None
            counts[source_id] = counts.get(source_id, 0) + 1
            evidence_id = f"{source_id}:E{counts[source_id]}"
            spans[evidence_id] = EvidenceSpan(
                evidence_id, source_id, context_index, start, end,
                context[start:end], loc, kind, parents)
            return evidence_id

        def generic(start: int, end: int, loc: LegalLocation | None,
                    parents: tuple[str, ...]) -> None:
            # A logical Markdown table can continue across blank lines and
            # repeat a separator. Keep its original header and every column
            # together rather than fabricating row/header quotations.
            lines = list(re.finditer(r"[^\n]*(?:\n|$)", context[start:end]))
            table_ranges: list[tuple[int, int]] = []
            cursor = 0
            while cursor < len(lines):
                line = lines[cursor]
                if not _TABLE_LINE.fullmatch(line.group()):
                    cursor += 1
                    continue
                first = cursor
                last = cursor
                cursor += 1
                while cursor < len(lines):
                    value = lines[cursor].group()
                    if _TABLE_LINE.fullmatch(value):
                        last = cursor
                        cursor += 1
                    elif not value.strip():
                        after = cursor + 1
                        while after < len(lines) and not lines[after].group().strip():
                            after += 1
                        if after < len(lines) and _TABLE_LINE.fullmatch(lines[after].group()):
                            cursor = after
                        else:
                            break
                    else:
                        break
                table_ranges.append((start + lines[first].start(),
                                     start + lines[last].end()))
            position = start
            last_id: str | None = None

            def paragraphs(a: int, b: int) -> None:
                nonlocal last_id
                part_start = a
                for blank in re.finditer(r"\n[ \t\r]*\n+", context[a:b]):
                    item_id = add(part_start, a + blank.start(), loc, "text", parents)
                    if item_id:
                        last_id = item_id
                    part_start = a + blank.end()
                item_id = add(part_start, b, loc, "text", parents)
                if item_id:
                    last_id = item_id

            for table_start, table_end in table_ranges:
                paragraphs(position, table_start)
                table_parents = parents + ((last_id,) if last_id else ())
                last_id = add(table_start, table_end, loc, "table", table_parents)
                position = table_end
            paragraphs(position, end)

        # Keep the source's subject and article heading as context parents, but
        # never publish the SOURCE_ID marker itself as proof.
        body_start = 0
        root_parents: tuple[str, ...] = ()
        if wrapper:
            newline = context.find("\n")
            body_start = len(context) if newline < 0 else newline + 1
            document_header = re.search(r"(?m)^Văn bản:[^\n]*(?:\n|$)", context)
            if document_header:
                body_start = document_header.end()
            header = add(wrapper.end(), body_start, _child(location), "header")
            if header:
                root_parents = (header,)
        if body_start >= len(context):
            continue
        body = context[body_start:]
        if _quoted_amendment(body):
            generic(body_start, len(context), None, root_parents)
            continue
        clause_matches = list(_CLAUSE.finditer(context, body_start))
        if location is None:
            generic(body_start, len(context), None, root_parents)
            continue
        if location.paragraph is not None and any(
            str(int(match.group(1))) != location.paragraph
            if match.group(1).isdigit() else match.group(1).casefold() != location.paragraph
            for match in clause_matches
        ):
            # A child source unexpectedly containing other numbered clauses
            # may be a quotation, a nested list, or malformed extraction.
            generic(body_start, len(context), None, root_parents)
            continue
        sections = [(match.start(),
                     clause_matches[index + 1].start()
                     if index + 1 < len(clause_matches) else len(context),
                     match.group(1))
                    for index, match in enumerate(clause_matches)]
        if not sections:
            sections = [(body_start, len(context), location.paragraph)]
        elif sections[0][0] > body_start:
            header = add(body_start, sections[0][0], _child(location), "header", root_parents)
            if header:
                root_parents += (header,)
        for start, end, paragraph in sections:
            paragraph_location = _child(location, paragraph)
            # An unnumbered article-level block cannot supply guessed points.
            points = _point_definitions(context, start, end) if paragraph else []
            if location.point and any(match.group(1).casefold() != location.point
                                      for match in points):
                generic(start, end, None, root_parents)
                continue
            if not points:
                loc = _child(location, paragraph, location.point)
                generic(start, end, loc, root_parents)
                continue
            parents = root_parents
            lead = add(start, points[0].start(), paragraph_location, "clause_lead", parents)
            if lead:
                parents += (lead,)
            for index, point_match in enumerate(points):
                stop = points[index + 1].start() if index + 1 < len(points) else end
                add(point_match.start(), stop,
                    _child(location, paragraph, point_match.group(1)), "point", parents)

    catalog = []
    for span in spans.values():
        entry = {"evidence_id": span.evidence_id, "source_id": span.source_id,
                 "kind": span.kind, "parent_ids": list(span.parent_ids),
                 "text": span.text}
        if span.location:
            entry.update(doc_id=span.location.doc_id, article=span.location.article,
                         paragraph=span.location.paragraph, point=span.location.point)
        catalog.append(entry)
    return EvidenceIndex(spans, json.dumps({"evidence_catalog": catalog}, ensure_ascii=False))
