"""Check whether the HF GroundingDINO dependency path is runnable."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.grounding_dino import (  # noqa: E402
    classify_hf_groundingdino_exception,
    load_hf_component_with_cache_fallback,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True
    probable_cause: str | None = None
    suggested_action: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose the HF GroundingDINO Python environment.")
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument("--try-model-load", action="store_true", help="Also call from_pretrained for the HF model.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(levelname)s:%(name)s:%(message)s")

    print("HF GroundingDINO environment check")
    print(f"Python: {sys.version.replace(chr(10), ' ')}")
    print(f"Model id: {args.model_id}")
    print("")

    results: list[CheckResult] = []
    torch_result, torch_module = _run_check("torch", _import_torch)
    results.append(torch_result)
    results.append(_run_check("torchvision", _import_torchvision)[0])
    transformers_result, transformers_module = _run_check("transformers", _import_transformers)
    results.append(transformers_result)
    auto_processor_result, auto_processor = _run_check("transformers.AutoProcessor", _import_auto_processor)
    auto_model_result, auto_model = _run_check(
        "transformers.AutoModelForZeroShotObjectDetection",
        _import_auto_model_for_zero_shot_object_detection,
    )
    results.extend([auto_processor_result, auto_model_result])

    if args.try_model_load:
        if auto_processor_result.ok and auto_model_result.ok:
            model_result, _ = _run_check(
                f"model load: {args.model_id}",
                lambda: _load_model(args.model_id, auto_processor, auto_model),
            )
            results.append(model_result)
        else:
            results.append(
                CheckResult(
                    name=f"model load: {args.model_id}",
                    ok=False,
                    detail="skipped because required Transformers classes did not import",
                    probable_cause="Transformers GroundingDINO import path is broken",
                    suggested_action="Fix the failed import checks above before trying model loading.",
                )
            )

    _print_version_details(torch_module=torch_module, transformers_module=transformers_module)
    _print_results(results)

    failed_required = [result for result in results if result.required and not result.ok]
    print("")
    print(f"Summary: {len(results) - len(failed_required)} passed, {len(failed_required)} failed")
    if failed_required:
        print("HF GroundingDINO is not runnable in this environment yet.")
        return 1
    print("HF GroundingDINO dependency path looks runnable.")
    return 0


def _run_check(name: str, fn: Callable[[], Any]) -> tuple[CheckResult, Any | None]:
    try:
        value = fn()
        return CheckResult(name=name, ok=True, detail=_describe_value(value)), value
    except Exception as exc:
        LOGGER.debug("Check failed: %s", name, exc_info=True)
        diagnosis = classify_hf_groundingdino_exception(exc)
        return (
            CheckResult(
                name=name,
                ok=False,
                detail=f"{type(exc).__name__}: {exc}",
                probable_cause=diagnosis.probable_cause,
                suggested_action=diagnosis.suggested_action,
            ),
            None,
        )


def _print_version_details(torch_module: Any | None, transformers_module: Any | None) -> None:
    print("Versions:")
    if torch_module is None:
        print("- torch: unavailable")
    else:
        cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
        print(f"- torch: {getattr(torch_module, '__version__', 'unknown')} (torch.version.cuda={cuda_version})")

    try:
        import torchvision

        print(f"- torchvision: {getattr(torchvision, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"- torchvision: unavailable ({type(exc).__name__}: {exc})")

    if transformers_module is None:
        print("- transformers: unavailable")
    else:
        print(f"- transformers: {getattr(transformers_module, '__version__', 'unknown')}")
    print("")


def _print_results(results: list[CheckResult]) -> None:
    print("Checks:")
    for result in results:
        status = "OK" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.detail}")
        if not result.ok:
            print(f"      probable cause: {result.probable_cause}")
            print(f"      suggested next action: {result.suggested_action}")


def _describe_value(value: Any) -> str:
    version = getattr(value, "__version__", None)
    if version is not None:
        return str(version)
    name = getattr(value, "__name__", None)
    if name is not None:
        return str(name)
    return type(value).__name__


def _import_torch() -> Any:
    import torch

    return torch


def _import_torchvision() -> Any:
    import torchvision

    return torchvision


def _import_transformers() -> Any:
    import transformers

    return transformers


def _import_auto_processor() -> Any:
    from transformers import AutoProcessor

    return AutoProcessor


def _import_auto_model_for_zero_shot_object_detection() -> Any:
    from transformers import AutoModelForZeroShotObjectDetection

    return AutoModelForZeroShotObjectDetection


def _load_model(model_id: str, auto_processor: Any, auto_model: Any) -> dict[str, str]:
    processor = load_hf_component_with_cache_fallback(auto_processor, model_id)
    model = load_hf_component_with_cache_fallback(auto_model, model_id)
    return {
        "processor": type(processor).__name__,
        "model": type(model).__name__,
    }


if __name__ == "__main__":
    raise SystemExit(main())
