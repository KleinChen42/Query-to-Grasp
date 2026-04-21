from __future__ import annotations

import pytest

from src.memory.fusion import FusionScoreTerms, FusionWeights, compute_fusion_score


def test_compute_fusion_score_uses_configured_weights() -> None:
    result = compute_fusion_score(
        FusionScoreTerms(
            det_score=0.8,
            clip_score=0.2,
            view_score=1.0,
            consistency_score=0.5,
            geometry_score=0.0,
        ),
        FusionWeights(
            det_score=0.5,
            clip_score=0.5,
            view_score=0.0,
            consistency_score=0.0,
            geometry_score=0.0,
        ),
    )

    assert result.overall_confidence == pytest.approx(0.5)
    assert result.contributions["det_score"] == pytest.approx(0.4)
    assert result.contributions["clip_score"] == pytest.approx(0.1)
    assert result.normalizer == pytest.approx(1.0)


def test_compute_fusion_score_clips_terms() -> None:
    result = compute_fusion_score(
        {
            "det_score": 2.0,
            "clip_score": -1.0,
            "view_score": 0.5,
            "consistency_score": 0.5,
            "geometry_score": 0.5,
        },
        FusionWeights(
            det_score=1.0,
            clip_score=1.0,
            view_score=0.0,
            consistency_score=0.0,
            geometry_score=0.0,
        ),
    )

    assert result.terms.det_score == 1.0
    assert result.terms.clip_score == 0.0
    assert result.overall_confidence == pytest.approx(0.5)


def test_compute_fusion_score_rejects_invalid_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        compute_fusion_score(
            FusionScoreTerms(det_score=1.0),
            FusionWeights(
                det_score=-1.0,
                clip_score=0.0,
                view_score=0.0,
                consistency_score=0.0,
                geometry_score=0.0,
            ),
        )

    with pytest.raises(ValueError, match="positive"):
        compute_fusion_score(
            FusionScoreTerms(det_score=1.0),
            FusionWeights(
                det_score=0.0,
                clip_score=0.0,
                view_score=0.0,
                consistency_score=0.0,
                geometry_score=0.0,
            ),
        )
