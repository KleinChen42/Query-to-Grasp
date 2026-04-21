from __future__ import annotations

import pytest

from src.perception.query_parser import parse_query, parse_query_llm, parse_query_rules


def test_parse_query_rules_returns_expected_structure() -> None:
    parsed = parse_query_rules("pick the small red cube near the mug")

    assert parsed["raw_query"] == "pick the small red cube near the mug"
    assert parsed["target_name"] == "cube"
    assert parsed["attributes"] == ["small", "red"]
    assert parsed["relations"] == [{"type": "near", "object": "mug"}]
    assert parsed["synonyms"] == ["cube", "block"]
    assert parsed["normalized_prompt"] == "small red cube"


def test_parse_query_falls_back_to_rules_without_llm() -> None:
    parsed = parse_query("blue mug", prefer_llm=True)

    assert parsed["target_name"] == "mug"
    assert parsed["attributes"] == ["blue"]
    assert parsed["normalized_prompt"] == "blue mug"


def test_parse_query_rules_accepts_generic_object_query() -> None:
    parsed = parse_query_rules("object")

    assert parsed["target_name"] == "object"
    assert parsed["synonyms"] == ["object"]
    assert parsed["normalized_prompt"] == "object"


def test_parse_query_llm_accepts_json_callable() -> None:
    parsed = parse_query_llm(
        "red block",
        llm_callable=lambda _: '{"target_name": "block", "attributes": ["red"], "relations": [], "synonyms": ["block", "cube"], "normalized_prompt": "red block"}',
    )

    assert parsed["target_name"] == "block"
    assert parsed["synonyms"] == ["block", "cube"]


def test_parse_query_rules_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        parse_query_rules("  ")
