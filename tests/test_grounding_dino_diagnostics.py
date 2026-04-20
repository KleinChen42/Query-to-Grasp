from __future__ import annotations

import pytest

from src.perception.grounding_dino import (
    classify_hf_groundingdino_exception,
    format_hf_groundingdino_error,
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


def test_raise_helper_preserves_exception_chain() -> None:
    exc = RuntimeError("operator torchvision::nms does not exist")

    with pytest.raises(RuntimeError) as error:
        raise_hf_groundingdino_error("importing torchvision", exc)

    assert error.value.__cause__ is exc
    assert "operator torchvision::nms does not exist" in str(error.value)
