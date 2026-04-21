"""Lightweight query parsing for single-view semantic retrieval."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Mapping, Sequence

LOGGER = logging.getLogger(__name__)

ATTRIBUTE_WORDS = {
    "red",
    "green",
    "blue",
    "yellow",
    "orange",
    "purple",
    "pink",
    "black",
    "white",
    "gray",
    "grey",
    "brown",
    "small",
    "large",
    "big",
    "tiny",
    "medium",
    "round",
    "square",
    "metal",
    "metallic",
    "wooden",
    "plastic",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "at",
    "find",
    "get",
    "grab",
    "of",
    "pick",
    "please",
    "select",
    "target",
    "the",
    "to",
}

RELATION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("left_of", r"\bleft\s+of\b"),
    ("right_of", r"\bright\s+of\b"),
    ("in_front_of", r"\bin\s+front\s+of\b"),
    ("behind", r"\bbehind\b"),
    ("next_to", r"\bnext\s+to\b"),
    ("near", r"\bnear\b"),
    ("on", r"\bon\b"),
    ("under", r"\bunder\b"),
)

SYNONYMS: dict[str, list[str]] = {
    "block": ["block", "cube"],
    "cube": ["cube", "block"],
    "cup": ["cup", "mug"],
    "mug": ["mug", "cup"],
    "can": ["can", "tin"],
    "bottle": ["bottle"],
    "banana": ["banana"],
    "apple": ["apple"],
    "object": ["object"],
}


def parse_query_rules(query: str) -> dict[str, Any]:
    """Parse a natural-language object query using deterministic rules."""

    raw_query = query.strip()
    if not raw_query:
        raise ValueError("Query must be a non-empty string.")

    normalized_text = _normalize_text(raw_query)
    target_phrase, relations = _split_relations(normalized_text)
    tokens = [token for token in target_phrase.split() if token and token not in STOP_WORDS]
    attributes = _dedupe([token for token in tokens if token in ATTRIBUTE_WORDS])
    target_candidates = [token for token in tokens if token not in ATTRIBUTE_WORDS]
    target_name = _choose_target_name(target_candidates, normalized_text)
    synonyms = SYNONYMS.get(target_name, [target_name])
    normalized_prompt = " ".join(_dedupe([*attributes, target_name])).strip()

    return {
        "raw_query": raw_query,
        "target_name": target_name,
        "attributes": attributes,
        "relations": relations,
        "synonyms": synonyms,
        "normalized_prompt": normalized_prompt or target_name,
    }


def parse_query_llm(
    query: str,
    llm_callable: Callable[[str], str | Mapping[str, Any]] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Parse a query with an externally supplied LLM callable.

    No model provider is imported here. Pass a callable that returns either a
    parsed mapping or a JSON string with the expected query schema.
    """

    if llm_callable is None:
        raise RuntimeError(
            "parse_query_llm requires an llm_callable. For Phase 2A, use "
            "parse_query_rules() or parse_query(..., prefer_llm=False) when no "
            "LLM client is configured."
        )

    response = llm_callable(query)
    if isinstance(response, str):
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM parser returned invalid JSON: {exc}") from exc
    else:
        parsed = dict(response)

    return _normalize_parsed_query(query, parsed)


def parse_query(
    query: str,
    prefer_llm: bool = False,
    llm_callable: Callable[[str], str | Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Parse a query, optionally trying an LLM parser before rules fallback."""

    if prefer_llm:
        try:
            return parse_query_llm(query, llm_callable=llm_callable)
        except Exception as exc:
            LOGGER.warning("LLM query parsing failed; falling back to rules: %s", exc)
    return parse_query_rules(query)


def _normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _split_relations(text: str) -> tuple[str, list[dict[str, str]]]:
    earliest_match: tuple[int, int, str, str] | None = None
    for relation_type, pattern in RELATION_PATTERNS:
        match = re.search(pattern, text)
        if match and (earliest_match is None or match.start() < earliest_match[0]):
            earliest_match = (match.start(), match.end(), relation_type, match.group(0))

    if earliest_match is None:
        return text, []

    start, end, relation_type, _ = earliest_match
    target_phrase = text[:start].strip()
    relation_object_phrase = text[end:].strip()
    relation_object = _clean_relation_object(relation_object_phrase)
    relations = [{"type": relation_type, "object": relation_object}] if relation_object else []
    return target_phrase, relations


def _clean_relation_object(text: str) -> str:
    tokens = [token for token in text.split() if token and token not in STOP_WORDS]
    tokens = [token for token in tokens if token not in ATTRIBUTE_WORDS]
    return tokens[-1] if tokens else ""


def _choose_target_name(candidates: Sequence[str], normalized_text: str) -> str:
    if candidates:
        return candidates[-1]

    tokens = [token for token in normalized_text.split() if token not in STOP_WORDS]
    if tokens:
        return tokens[-1]

    raise ValueError(f"Could not identify target object from query: {normalized_text!r}")


def _normalize_parsed_query(raw_query: str, parsed: Mapping[str, Any]) -> dict[str, Any]:
    rule_fallback = parse_query_rules(raw_query)
    target_name = str(parsed.get("target_name") or rule_fallback["target_name"]).strip().lower()
    attributes = _dedupe([str(item).strip().lower() for item in parsed.get("attributes", []) if str(item).strip()])
    relations = parsed.get("relations", rule_fallback["relations"])
    synonyms = _dedupe([str(item).strip().lower() for item in parsed.get("synonyms", []) if str(item).strip()])
    if not synonyms:
        synonyms = SYNONYMS.get(target_name, [target_name])

    normalized_prompt = str(parsed.get("normalized_prompt") or " ".join([*attributes, target_name])).strip().lower()
    return {
        "raw_query": raw_query,
        "target_name": target_name,
        "attributes": attributes,
        "relations": list(relations) if isinstance(relations, Sequence) and not isinstance(relations, str) else [],
        "synonyms": synonyms,
        "normalized_prompt": normalized_prompt or target_name,
    }


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            deduped.append(item)
    return deduped
