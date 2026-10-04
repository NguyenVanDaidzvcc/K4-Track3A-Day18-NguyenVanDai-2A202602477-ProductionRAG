"""Keep explicitly superseded policy versions out of current-policy retrieval."""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import date

_VERSION = r"\d+(?:\.\d+)*"
_VERSION_LABEL = rf"(?:phiên bản|versions?|bản|v)\s*({_VERSION})"
_REPLACEMENT = re.compile(r"\b(?:thay thế|replaces?|supersedes?)\b", re.IGNORECASE)


def _version_key(value) -> tuple[int, ...]:
    match = re.fullmatch(rf"\s*[vV]?({_VERSION})\s*", str(value or ""))
    if not match:
        return ()
    parts = [int(part) for part in match.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _version(value) -> str | None:
    match = re.fullmatch(rf"\s*[vV]?({_VERSION})\s*", str(value or ""))
    return match.group(1) if match else None


def _date(value) -> str | None:
    text = str(value or "").strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
            return date.fromisoformat(text).isoformat()
        match = re.fullmatch(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", text)
        if match:
            day, month, year = map(int, match.groups())
            return date(year, month, day).isoformat()
    except ValueError:
        return None
    return None


def _source_stem(source: str) -> str:
    # Sources may be Windows paths even when a consumer runs on another OS.
    stem = os.path.splitext(re.split(r"[/\\]", source)[-1])[0]
    return re.sub(r"(?:[_-]v\d+(?:[._-]\d+)*)$", "", stem, flags=re.IGNORECASE)


def _family(title: str, source: str) -> str:
    normalized = unicodedata.normalize("NFC", title or _source_stem(source)).casefold()
    normalized = re.sub(
        r"[\[(][^\])]*(?:phiên bản|version|hiện hành|cũ|current|old|previous|"
        r"effective|hiệu lực|\b(?:19|20)\d{2}\b|\bv\d)[^\])]*[\])]",
        "", normalized,
    )
    normalized = re.sub(
        rf"(?:\s*[-–—:|]\s*|\s+)(?:phiên bản|version|v)\s*{_VERSION}\s*$",
        "", normalized,
    )
    normalized = re.sub(r"(?:\s*[-–—:|]\s*|\s+)\b(?:19|20)\d{2}\b\s*$", "", normalized)
    normalized = normalized.replace("_", " ")
    return re.sub(r"\s+", " ", normalized).strip(" #\t-–—:|")


def _document_metadata(document: dict) -> dict:
    text = document.get("text", "")
    metadata = dict(document.get("metadata") or {})
    source = str(metadata.get("source", ""))
    heading = re.search(r"^ {0,3}#[ \t]+(.+)$", text, flags=re.MULTILINE)
    title = str(metadata.get("document_title") or metadata.get("title")
                or (heading.group(1).strip(" #\t\r") if heading else _source_stem(source)))
    header = "\n".join(line for line in text.splitlines()[:8] if line.lstrip().startswith(">"))
    version_match = re.search(rf"\b(?:phiên bản|version)\s*:\s*[vV]?({_VERSION})", header, re.IGNORECASE)
    source_version = re.search(rf"(?:[_-]v)({_VERSION})(?:\.\w+)?$", source, re.IGNORECASE)
    title_version = re.search(rf"\b{_VERSION_LABEL}\b", title, re.IGNORECASE)
    parsed_version = next((match.group(1) for match in (version_match, title_version, source_version) if match), None)
    date_match = re.search(
        r"(?:ngày hiệu lực|effective(?:\s+date)?)\s*:\s*"
        r"(\d{4}-\d{2}-\d{2}|\d{1,2}[/.-]\d{1,2}[/.-]\d{4})",
        header, re.IGNORECASE,
    )
    metadata.update({
        "document_title": title,
        "policy_family": metadata.get("policy_family") or _family(title, source),
        "version": _version(metadata.get("version")) or _version(parsed_version),
        "effective_date": _date(metadata.get("effective_date")) or _date(date_match.group(1) if date_match else None),
        "is_superseded": False,
        "is_current": True,
    })
    metadata.pop("superseded_by", None)
    return metadata


def _replacement_versions(text: str, family: str, families: set[str]) -> tuple[set[tuple[int, ...]], bool]:
    versions = set()
    replaces_all = False
    for statement in re.split(r"(?<=[.!?])\s+|\n+", text):
        replacement = _REPLACEMENT.search(statement)
        if replacement is None:
            continue
        prefix = statement[:replacement.start()]
        if re.search(r"(?:không|chưa|not|never|doesn't|isn't)\s*$", prefix, re.IGNORECASE):
            continue
        tail = statement[replacement.end():]
        # A statement saying a document "is replaced by" names its successor,
        # not an older version replaced by the document itself.
        if re.match(r"\s+(?:bởi|by)\b", tail, re.IGNORECASE):
            continue
        reference = unicodedata.normalize("NFC", tail).casefold()
        if family not in reference and any(other != family and other in reference for other in families):
            continue
        if re.search(
            r"(?:các|mọi|tất cả)(?:\s+\w+){0,3}\s+phiên bản(?:\s+\w+){0,2}\s+trước|"
            r"(?:all\s+)?(?:previous|prior|earlier)\s+versions?",
            tail, re.IGNORECASE,
        ):
            replaces_all = True
        for match in re.finditer(rf"\b{_VERSION_LABEL}\b", tail, re.IGNORECASE):
            versions.add(_version_key(match.group(1)))
        # "Replaces versions 1.2 and 1.3" also explicitly names 1.3.
        if versions:
            for match in re.finditer(r"\b\d+(?:\.\d+)+\b", tail):
                versions.add(_version_key(match.group()))
    return versions, replaces_all


def annotate_documents(documents: list[dict]) -> list[dict]:
    """Copy documents and mark replacements proven by the latest family document.

    A newer timestamp alone is insufficient evidence that an older document was
    superseded. Replacement statements must name the old version or explicitly
    replace all previous versions in the same policy family.
    """
    annotated = [{**document, "metadata": _document_metadata(document)} for document in documents]
    families: dict[str, list[dict]] = {}
    for document in annotated:
        family = document["metadata"]["policy_family"]
        if family:
            families.setdefault(family, []).append(document)
    for family_documents in families.values():
        latest = max(family_documents, key=lambda document: (
            document["metadata"]["effective_date"] or "",
            _version_key(document["metadata"]["version"]),
        ))
        replaced_versions, replaces_all = _replacement_versions(
            latest.get("text", ""), latest["metadata"]["policy_family"], set(families),
        )
        for document in family_documents:
            if document is latest:
                continue
            metadata = document["metadata"]
            key = _version_key(metadata["version"])
            if key and (key in replaced_versions or replaces_all):
                metadata.update({
                    "is_superseded": True,
                    "is_current": False,
                    "superseded_by": latest["metadata"].get("source") or latest["metadata"]["document_title"],
                })
    return annotated


def prefer_current_candidates(query: str, candidates: list[dict]) -> list[dict]:
    """Filter superseded candidates while preserving explicit historical intent.

    For a specified version/year, matching candidates come first. Current
    documents from other families stay available for questions needing several
    policies; conflicting versions in the requested family are omitted.
    Comparison questions retain both old and current documents.
    """
    query = unicodedata.normalize("NFC", query).casefold()
    if re.search(r"\b(?:so sánh|khác nhau|thay đổi|compare|comparison|versus|difference)\b", query):
        return list(candidates)
    versions = {_version_key(match.group(1)) for match in re.finditer(rf"\b{_VERSION_LABEL}\b", query)}
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", query))
    historical = bool(re.search(r"\b(?:cũ|trước đây|trước đó|lịch sử|old|previous|historical|formerly)\b", query))

    def current(candidate: dict) -> bool:
        metadata = candidate.get("metadata") or {}
        return not metadata.get("is_superseded", False) and metadata.get("is_current", True) is not False

    if versions or years or historical:
        matching = []
        for candidate in candidates:
            metadata = candidate.get("metadata") or {}
            document_years = set(re.findall(
                r"(?<!\d)(?:19|20)\d{2}(?!\d)",
                " ".join(str(metadata.get(key) or "").replace("_", " ")
                         for key in ("effective_date", "source", "document_title")),
            ))
            if ((versions and _version_key(metadata.get("version")) in versions)
                    or (years and document_years & years)
                    or (historical and metadata.get("is_superseded", False))):
                matching.append(candidate)
        if matching:
            requested_families = {(candidate.get("metadata") or {}).get("policy_family") for candidate in matching}
            requested_families.discard(None)
            requested_families.discard("")
            other_current = [candidate for candidate in candidates
                             if candidate not in matching and current(candidate)
                             and (candidate.get("metadata") or {}).get("policy_family") not in requested_families]
            return matching + other_current
        # Do not discard potentially necessary history when its metadata lacks
        # the date/version explicitly requested by the user.
        return list(candidates)
    return [candidate for candidate in candidates if current(candidate)]
