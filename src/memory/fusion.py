"""Confidence-aware score fusion for 3D object hypotheses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class FusionWeights:
    """Configurable weights for ablation-friendly confidence fusion."""

    det_score: float = 0.30
    clip_score: float = 0.30
    view_score: float = 0.15
    consistency_score: float = 0.15
    geometry_score: float = 0.10

    def to_json_dict(self) -> dict[str, float]:
        """Return JSON-serializable weights."""

        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class FusionScoreTerms:
    """Normalized score terms for one object hypothesis."""

    det_score: float = 0.0
    clip_score: float = 0.0
    view_score: float = 0.0
    consistency_score: float = 0.0
    geometry_score: float = 0.0

    def to_json_dict(self) -> dict[str, float]:
        """Return JSON-serializable score terms."""

        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class FusionResult:
    """Fused confidence and the terms that produced it."""

    overall_confidence: float
    terms: FusionScoreTerms
    weights: FusionWeights
    contributions: dict[str, float]
    normalizer: float

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable fusion trace."""

        return {
            "overall_confidence": float(self.overall_confidence),
            "terms": self.terms.to_json_dict(),
            "weights": self.weights.to_json_dict(),
            "contributions": {key: float(value) for key, value in self.contributions.items()},
            "normalizer": float(self.normalizer),
        }


def compute_fusion_score(
    terms: FusionScoreTerms | Mapping[str, Any],
    weights: FusionWeights | None = None,
) -> FusionResult:
    """Fuse normalized confidence terms into one bounded score.

    The implementation intentionally keeps the formula transparent for paper
    ablations: each term is clipped to ``[0, 1]``, multiplied by its configured
    weight, and divided by the sum of active weights. Set any weight to ``0`` to
    disable that term without changing call sites.
    """

    weights = weights or FusionWeights()
    normalized_terms = _coerce_terms(terms)
    _validate_weights(weights)

    term_values = normalized_terms.to_json_dict()
    weight_values = weights.to_json_dict()
    contributions = {
        name: term_values[name] * weight_values[name]
        for name in term_values
    }
    normalizer = sum(weight for weight in weight_values.values() if weight > 0.0)
    if normalizer <= 0.0:
        raise ValueError("At least one fusion weight must be positive.")

    overall_confidence = clip01(sum(contributions.values()) / normalizer)
    return FusionResult(
        overall_confidence=overall_confidence,
        terms=normalized_terms,
        weights=weights,
        contributions=contributions,
        normalizer=float(normalizer),
    )


def clip01(value: Any) -> float:
    """Convert a value to float and clip it into ``[0, 1]``."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


def _coerce_terms(terms: FusionScoreTerms | Mapping[str, Any]) -> FusionScoreTerms:
    if isinstance(terms, FusionScoreTerms):
        values = terms.to_json_dict()
    else:
        values = dict(terms)
    return FusionScoreTerms(
        det_score=clip01(values.get("det_score", 0.0)),
        clip_score=clip01(values.get("clip_score", 0.0)),
        view_score=clip01(values.get("view_score", 0.0)),
        consistency_score=clip01(values.get("consistency_score", 0.0)),
        geometry_score=clip01(values.get("geometry_score", 0.0)),
    )


def _validate_weights(weights: FusionWeights) -> None:
    for name, value in weights.to_json_dict().items():
        if value < 0.0:
            raise ValueError(f"Fusion weight {name!r} must be non-negative, got {value}.")
