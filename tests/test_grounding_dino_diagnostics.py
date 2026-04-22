from __future__ import annotations

import pytest

from src.perception.grounding_dino import (
    classify_hf_groundingdino_exception,
    format_hf_groundingdino_error,
    load_hf_component_with_cache_fallback,
    raise_hf_groundingdino_error,
)


def test_classifies_torchvision_cuda_mismatch() -> None:
    exc = RuntimeError("operator torchvision::nms does not exist")

    diagnosis = classify_hf_groundingdino_exception(exc)

    assert "torch/torchvision" in diagnosis.probable_cause
    assert "CUDA" in diagnosis.probable_cause
    assert "matching torch and torchvision" in diagnosis.suggested_action


def test_classifies_torchvision_import_failure() -> None:
    exc = ImportError("No module named 'torchvision'")

    diagnosis = classify_hf_groundingdino_exception(exc)

    assert diagnosis.probable_cause == "torchvision import failure"
    assert "torchvision" in diagnosis.suggested_action


def test_classifies_auto_processor_import_failure() -> None:
    exc = ImportError("cannot import name 'AutoProcessor' from 'transformers'")

    diagnosis = classify_hf_groundingdino_exception(exc)

    assert diagnosis.probable_cause == "Transformers AutoProcessor import failure"
    assert "transformers" in diagnosis.suggested_action.lower()


def test_formatted_error_includes_original_exception_text() -> None:
    exc = ImportError("cannot import name 'AutoProcessor' from 'transformers'")

    message = format_hf_groundingdino_error("importing Transformers GroundingDINO classes", exc)

    assert "cannot import name 'AutoProcessor' from 'transformers'" in message
    assert "Probable cause:" in message
    assert "Suggested next action:" in message
    assert "importing Transformers GroundingDINO classes" in message


def test_classifies_hf_network_or_cache_failure() -> None:
    exc = RuntimeError("Cannot send a request, as the client has been closed.")

    diagnosis = classify_hf_groundingdino_exception(exc)

    assert diagnosis.probable_cause == "HF model download/cache access failed"
    assert "cached" in diagnosis.suggested_action


def test_hf_component_loader_retries_local_cache() -> None:
    class FakeComponent:
        calls: list[dict[str, object]] = []

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs):
            cls.calls.append({"model_id": model_id, **kwargs})
            if not kwargs.get("local_files_only"):
                raise RuntimeError("Network is unreachable")
            return {"model_id": model_id, "local_files_only": True}

    loaded = load_hf_component_with_cache_fallback(FakeComponent, "fake/model")

    assert loaded == {"model_id": "fake/model", "local_files_only": True}
    assert FakeComponent.calls == [
        {"model_id": "fake/model"},
        {"model_id": "fake/model", "local_files_only": True},
    ]


def test_raise_helper_preserves_exception_chain() -> None:
    exc = RuntimeError("operator torchvision::nms does not exist")

    with pytest.raises(RuntimeError) as error:
        raise_hf_groundingdino_error("importing torchvision", exc)

    assert error.value.__cause__ is exc
    assert "operator torchvision::nms does not exist" in str(error.value)
