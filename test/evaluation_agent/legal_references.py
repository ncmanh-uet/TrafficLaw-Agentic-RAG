"""Conservative structural parsing of Vietnamese legal cross-references.

This module checks identities and links, not whether a provision entails a
claim. Point letters are deliberately not accent-folded: ``d`` and ``đ`` are
different legal locations. Unresolved document names and unsupported reference
forms do not establish a link.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


def _text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFC", str(value or "")).casefold().split())


def _document(value: object) -> str:
    result = _text(value).translate(str.maketrans({"–": "-", "—": "-", "−": "-"}))
    return re.sub(r"\s*([/-])\s*", r"\1", result)


def _number(value: object, label: str) -> str | None:
    result = re.sub(r"^(?:" + label + r")\s+", "", _text(value))
    match = re.fullmatch(r"(\d+)([a-zđ]?)", result)
    return str(int(match[1])) + match[2] if match else None


@dataclass(frozen=True)
class LegalLocation:
    doc_id: str
    article: str
    paragraph: str | None = None
    point: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "doc_id", _document(self.doc_id))
        object.__setattr__(self, "article", _number(self.article, "điều") or "")
        if self.paragraph is not None:
            object.__setattr__(self, "paragraph", _number(self.paragraph, "khoản") or "")
        if self.point is not None:
            point = re.sub(r"^điểm\s+", "", _text(self.point))
            object.__setattr__(self, "point", point if re.fullmatch(r"[a-zđ]", point) else "")

    @property
    def valid(self) -> bool:
        return bool(self.doc_id and self.article and self.paragraph != "" and self.point != ""
                    and (self.point is None or self.paragraph is not None))


_JOIN = r"(?:\s*,\s*|\s+(?:và|hoặc)\s+)"
_POINTS = r"[a-zđ](?:" + _JOIN + r"(?:điểm\s+)?[a-zđ])*"
_NUMBER = r"\d+[a-zđ]?"
_TOKEN = re.compile(
    r"(?P<point>\b(?:các\s+)?điểm\s+(?P<points>" + _POINTS + r")\s*,?\s*"
    r"(?:các\s+)?khoản\s+(?P<point_paragraph>" + _NUMBER + r")\b)"
    r"|(?P<paragraph>\b(?:các\s+)?khoản\s+(?P<paragraphs>" + _NUMBER
    + r"(?:" + _JOIN + r"(?:khoản\s+)?" + _NUMBER + r")*)\b)"
    r"|(?P<article>\bđiều\s+(?P<article_value>này|" + _NUMBER + r")\b)"
)
_GAP = re.compile(r"(?:\s|[,;]|\b(?:và|hoặc|tại|của)\b)*\Z")
_DOC_NAME = re.compile(r"\s*(?:của\s+)?(?:bộ\s+luật|luật|nghị\s+định|thông\s+tư|"
                       r"nghị\s+quyết|quyết\s+định|pháp\s+lệnh)\b")
_DOC_ID = re.compile(r"\b\d{1,6}\s*/\s*\d{4}\s*/\s*[a-zđ0-9]+"
                     r"(?:\s*-\s*[a-zđ0-9]+)*\b")
_REFERENCE_CUE = re.compile(r"\b(?:điểm\s+(?:[a-zđ]\b|này\b)|"
                            r"khoản\s+(?:\d|này\b)|điều\s+(?:\d|này\b))")
_RANGE = re.compile(r"\b(?:điểm|khoản|điều)\s+(?:\d+[a-zđ]?|[a-zđ])\s*"
                    r"(?:đến|tới|[-–—])\s*(?:(?:điểm|khoản|điều)\s+)?(?:\d+[a-zđ]?|[a-zđ])\b")


def _document_suffix(text: str, end: int, default: str) -> str | None:
    """Resolve only an immediately attached document designation.

    A fully numbered identifier is accepted. ``Nghị định này`` keeps the
    current document. A named foreign law without its identifier is unresolved,
    rather than silently being interpreted as the source document.
    """
    suffix = text[end:]
    name = _DOC_NAME.match(suffix)
    if not name:
        return default
    body = suffix[name.end():]
    if re.match(r"\s+này\b", body):
        return default
    identifier = _DOC_ID.match(re.sub(r"^\s*(?:số\s*)?", "", body))
    if identifier:
        return _document(identifier[0])
    return None


def _dangling_point(text: str, start: int, previous_end: int = 0) -> bool:
    """Do not widen an unparsed point reference to its whole paragraph."""
    prefix = re.split(r"[;.\n]", text[previous_end:start])[-1]
    # The ordinary penalty word in "bị trừ điểm" is not a legal point
    # reference. A dangling reference needs a point letter (or "này"), not
    # merely an earlier occurrence of the word "điểm" in the provision.
    return bool(re.search(r"\bđiểm\s+(?:[a-zđ]\b|này\b)[^;\.\n]*$", prefix))


def _is_heading(text: str, token: re.Match[str]) -> bool:
    line_start = text.rfind("\n", 0, token.start()) + 1
    line_end = text.find("\n", token.end())
    line = text[line_start:line_end if line_end >= 0 else len(text)]
    if re.match(r"\s*(?:\[source_id=|văn bản\s*:)", line):
        return True
    if not token["article"] and not token["paragraph"]:
        return False
    return bool(re.fullmatch(r"[ \t#*_]*", text[line_start:token.start()])
                and re.match(r"\s*[.:]", text[token.end():]))


def _is_list_introduction(text: str, token: re.Match[str]) -> bool:
    # "... một trong các điểm, khoản sau của Điều này ...: điểm a khoản 1; ..."
    # qualifies the following enumerated list. It is not a blanket reference
    # to every provision in the article.
    return bool(token["article"] and re.search(
        r"\b(?:các\s+)?(?:điểm\s*(?:,|và)\s*)?khoản\s+sau\s+của\s*$",
        text[:token.start()]))


def _parse_references(text: str, *, doc_id: str, article: str) -> tuple[list[LegalLocation], bool]:
    normalized = unicodedata.normalize("NFC", str(text or "")).casefold()
    current = LegalLocation(doc_id, article)
    if not current.valid:
        return [], False
    ranges = list(_RANGE.finditer(normalized))
    raw_matches = list(_TOKEN.finditer(normalized))
    ignored = [token for token in raw_matches if _is_heading(normalized, token)
               or _is_list_introduction(normalized, token)]
    unsupported = [token for token in raw_matches if any(
        token.start() < scope.end() and scope.start() < token.end()
        or bool(token["article"] and scope.end() <= token.start()
                and _GAP.fullmatch(normalized[scope.end():token.start()]))
        for scope in ranges)]
    matches = [token for token in raw_matches if token not in ignored and token not in unsupported]
    covered = [(token.start(), token.end()) for token in ignored]
    covered.extend((line.start(), line.end()) for line in re.finditer(
        r"(?m)^[ \t]*(?:\[source_id=|văn bản\s*:)[^\n]*", normalized))
    complete = not bool(ranges)
    runs: list[list[re.Match[str]]] = []
    for match in matches:
        if runs and _GAP.fullmatch(normalized[runs[-1][-1].end():match.start()]):
            runs[-1].append(match)
        else:
            runs.append([match])
    result: list[LegalLocation] = []
    for run in runs:
        qualifiers = [index for index, token in enumerate(run) if token["article"]]
        previous_article = current.article
        previous_doc: str | None = current.doc_id
        for index, token in enumerate(run):
            previous_end = run[index - 1].end() if index else 0
            following = next((i for i in qualifiers if i >= index), None)
            if following is not None:
                qualifier = run[following]
                target_article = (current.article if qualifier["article_value"] == "này"
                                  else _number(qualifier["article_value"], "điều"))
                target_doc = _document_suffix(normalized, qualifier.end(), current.doc_id)
            else:
                target_article, target_doc = previous_article, previous_doc
                if index == len(run) - 1:
                    target_doc = _document_suffix(normalized, token.end(), target_doc or "")
            if token["article"]:
                previous_article, previous_doc = target_article or "", target_doc
                # The article token qualifies preceding point/clause tokens;
                # it is not also a reference to the entire article.
                if index and (run[index - 1]["point"] or run[index - 1]["paragraph"]):
                    covered.append((token.start(), token.end()))
                    continue
                if _dangling_point(normalized, token.start(), previous_end):
                    complete = False
                    continue
                locations = [(None, None)]
            elif token["point"]:
                # Unsupported range/point syntax before this token must not
                # turn its last point into a confidently parsed location.
                if _dangling_point(normalized, token.start(), previous_end):
                    complete = False
                    continue
                points = re.findall(r"(?<![\wđ])[a-zđ](?![\wđ])", token["points"])
                locations = [(token["point_paragraph"], point) for point in points]
            else:
                if _dangling_point(normalized, token.start(), previous_end):
                    complete = False
                    continue
                locations = [(value, None) for value in re.findall(_NUMBER, token["paragraphs"])]
            if not target_doc or not target_article:
                complete = False
                continue
            covered.append((token.start(), token.end()))
            for paragraph, point in locations:
                location = LegalLocation(target_doc, target_article, paragraph, point)
                if location.valid and location not in result:
                    result.append(location)
                elif not location.valid:
                    complete = False
    # A parser that recognises part of a list cannot certify that an unparsed
    # target is absent. In particular, missing clause-this/range/point syntax
    # must not validate a negative judge verdict by turning unknown into False.
    for cue in _REFERENCE_CUE.finditer(normalized):
        if not any(start <= cue.start() and cue.end() <= end for start, end in covered):
            complete = False
    return result, complete


def parse_references(text: str, *, doc_id: str, article: str) -> list[LegalLocation]:
    """Return confidently parsed reference targets, preserving reference scope.

    Supported forms are point lists followed by their paragraph, paragraph
    lists, and explicit/current-article suffixes. A trailing article qualifier
    applies backwards across the adjacent comma/semicolon reference list, not
    across intervening prose. Headings and introductions to enumerated lists
    are not blanket article references. Unresolved forms are omitted, so use
    :func:`reference_relation`, not absence from this list, to certify a
    negative relation.
    """
    return _parse_references(text, doc_id=doc_id, article=article)[0]


def reference_relation(text: str, source_location: LegalLocation,
                       target_location: LegalLocation) -> bool | None:
    """Return a structurally certain positive/negative relation, or unknown.

    A reference to a whole paragraph can cover a point within it, but a point
    reference cannot establish support for a different point or the entire
    paragraph. A negative relation is certified only if the reference list was
    completely parsed. Unsupported syntax never establishes a negative verdict.
    Explicit foreign references require a resolvable document ID.
    """
    if not source_location.valid or not target_location.valid:
        return None
    references, complete = _parse_references(text, doc_id=source_location.doc_id,
                                           article=source_location.article)
    for reference in references:
        if reference.doc_id != target_location.doc_id or reference.article != target_location.article:
            continue
        if reference.paragraph is not None and reference.paragraph != target_location.paragraph:
            continue
        if reference.point is not None and reference.point != target_location.point:
            continue
        return True
    return False if complete and references else None


def refers_to(text: str, source_location: LegalLocation, target_location: LegalLocation) -> bool:
    """Compatibility convenience: only a certain positive returns True.

    Do not use False from this helper as proof of a negative relation; it also
    covers unresolved syntax. Judge validation must use reference_relation.
    """
    return reference_relation(text, source_location, target_location) is True
