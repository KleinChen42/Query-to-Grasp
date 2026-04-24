"""Persistent 3D object hypotheses built from multi-view observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from src.memory.fusion import FusionResult, FusionScoreTerms, FusionWeights, clip01, compute_fusion_score


@dataclass(frozen=True)
class ObjectMemoryConfig:
    """Configuration for lightweight 3D object memory."""

    merge_distance: float = 0.08
    min_points_for_full_geometry_confidence: int = 1000
    max_views_for_full_view_confidence: int = 3
    fusion_weights: FusionWeights = field(default_factory=FusionWeights)


@dataclass
class ObjectObservation3D:
    """One semantic 3D observation from a camera view."""

    world_xyz: np.ndarray
    label: str
    det_score: float = 0.0
    clip_score: float = 0.0
    fused_2d_score: float | None = None
    view_id: str | None = None
    num_points: int = 0
    depth_valid_ratio: float = 0.0
    point_cloud_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_xyz", _validate_xyz(self.world_xyz))
        object.__setattr__(self, "label", self.label.strip())
        if not self.label:
            raise ValueError("ObjectObservation3D.label must be non-empty.")

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable observation."""

        return {
            "world_xyz": np.asarray(self.world_xyz, dtype=float).tolist(),
            "label": self.label,
            "det_score": float(self.det_score),
            "clip_score": float(self.clip_score),
            "fused_2d_score": None if self.fused_2d_score is None else float(self.fused_2d_score),
            "view_id": self.view_id,
            "num_points": int(self.num_points),
            "depth_valid_ratio": float(self.depth_valid_ratio),
            "point_cloud_path": self.point_cloud_path,
            "metadata": self.metadata,
        }


@dataclass
class MemoryObject3D:
    """A persistent object hypothesis in world coordinates."""

    object_id: str
    world_xyz: np.ndarray
    label_votes: dict[str, float] = field(default_factory=dict)
    det_scores: list[float] = field(default_factory=list)
    clip_scores: list[float] = field(default_factory=list)
    fused_2d_scores: list[float] = field(default_factory=list)
    view_ids: list[str] = field(default_factory=list)
    num_observations: int = 0
    geometry_confidence: float = 0.0
    semantic_confidence: float = 0.0
    overall_confidence: float = 0.0
    point_cloud_path: str | None = None
    score_terms: FusionScoreTerms = field(default_factory=FusionScoreTerms)
    fusion_trace: dict[str, Any] = field(default_factory=dict)
    observation_xyzs: list[np.ndarray] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_label(self) -> str:
        """Return the highest-vote label with deterministic tie-breaking."""

        if not self.label_votes:
            return ""
        return sorted(self.label_votes.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable object hypothesis."""

        return {
            "object_id": self.object_id,
            "world_xyz": np.asarray(self.world_xyz, dtype=float).tolist(),
            "label_votes": {label: float(score) for label, score in sorted(self.label_votes.items())},
            "top_label": self.top_label,
            "det_scores": [float(score) for score in self.det_scores],
            "clip_scores": [float(score) for score in self.clip_scores],
            "fused_2d_scores": [float(score) for score in self.fused_2d_scores],
            "view_ids": list(self.view_ids),
            "num_observations": int(self.num_observations),
            "geometry_confidence": float(self.geometry_confidence),
            "semantic_confidence": float(self.semantic_confidence),
            "overall_confidence": float(self.overall_confidence),
            "point_cloud_path": self.point_cloud_path,
            "score_terms": self.score_terms.to_json_dict(),
            "fusion_trace": self.fusion_trace,
            "observation_xyzs": [np.asarray(xyz, dtype=float).tolist() for xyz in self.observation_xyzs],
            "metadata": self.metadata,
        }


class ObjectMemory3D:
    """Merge 3D semantic observations into persistent object hypotheses."""

    def __init__(self, config: ObjectMemoryConfig | None = None) -> None:
        self.config = config or ObjectMemoryConfig()
        if self.config.merge_distance <= 0.0:
            raise ValueError("merge_distance must be positive.")
        self._objects: list[MemoryObject3D] = []
        self._next_object_index = 0

    @property
    def objects(self) -> list[MemoryObject3D]:
        """Return current object hypotheses in deterministic creation order."""

        return list(self._objects)

    def add_observation(self, observation: ObjectObservation3D) -> MemoryObject3D:
        """Add one observation, merging it with the nearest compatible object."""

        match, _ = self.add_observation_with_preferred_object(observation)
        return match

    def add_observation_with_preferred_object(
        self,
        observation: ObjectObservation3D,
        preferred_object_id: str | None = None,
        preferred_merge_distance: float | None = None,
    ) -> tuple[MemoryObject3D, dict[str, Any]]:
        """Add one observation with an optional preferred compatible object."""

        preferred_object = self.get_object_by_id(preferred_object_id)
        max_distance = self.config.merge_distance if preferred_merge_distance is None else float(preferred_merge_distance)
        preferred_distance = (
            None
            if preferred_object is None
            else float(np.linalg.norm(preferred_object.world_xyz - observation.world_xyz))
        )
        preferred_object_compatible = (
            preferred_object is not None
            and np.isfinite(max_distance)
            and max_distance >= 0.0
            and preferred_distance is not None
            and preferred_distance <= max_distance
        )
        existing_object_ids = {obj.object_id for obj in self._objects}
        confidence_state_before: dict[str, float] | None = None
        if preferred_object_compatible:
            match = preferred_object
            confidence_state_before = self._confidence_update_state(match)
            self._merge_into(match, observation)
        else:
            match = self._find_match(observation)
            if match is None:
                match = self._create_object(observation)
                self._objects.append(match)
            else:
                confidence_state_before = self._confidence_update_state(match)
                self._merge_into(match, observation)
        self._refresh_confidences(match, confidence_state_before=confidence_state_before)
        assignment = {
            "object_id": match.object_id,
            "preferred_object_id": preferred_object_id,
            "preferred_distance": preferred_distance,
            "preferred_object_compatible": preferred_object_compatible,
            "used_preferred_object": preferred_object_compatible and match.object_id == preferred_object_id,
            "created_new_object": match.object_id not in existing_object_ids,
        }
        return match, assignment

    def extend(self, observations: Iterable[ObjectObservation3D]) -> list[MemoryObject3D]:
        """Add several observations and return the updated hypotheses."""

        for observation in observations:
            self.add_observation(observation)
        return self.objects

    def select_best(self, label: str | None = None) -> MemoryObject3D | None:
        """Select the best object with deterministic tie-breaking."""

        candidates = self._objects
        if label is not None:
            label = label.strip()
            candidates = [obj for obj in candidates if label in obj.label_votes]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda obj: (
                -obj.overall_confidence,
                -len(obj.view_ids),
                -obj.geometry_confidence,
                obj.object_id,
            ),
        )[0]

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable memory snapshot."""

        return {
            "config": {
                "merge_distance": float(self.config.merge_distance),
                "min_points_for_full_geometry_confidence": int(
                    self.config.min_points_for_full_geometry_confidence
                ),
                "max_views_for_full_view_confidence": int(self.config.max_views_for_full_view_confidence),
                "fusion_weights": self.config.fusion_weights.to_json_dict(),
            },
            "num_objects": len(self._objects),
            "objects": [obj.to_json_dict() for obj in self._objects],
        }

    def get_object_by_id(self, object_id: str | None) -> MemoryObject3D | None:
        """Return one object by id."""

        if object_id is None:
            return None
        for obj in self._objects:
            if obj.object_id == object_id:
                return obj
        return None

    def _find_match(self, observation: ObjectObservation3D) -> MemoryObject3D | None:
        matches: list[tuple[float, str, MemoryObject3D]] = []
        for obj in self._objects:
            distance = float(np.linalg.norm(obj.world_xyz - observation.world_xyz))
            if distance <= self.config.merge_distance:
                matches.append((distance, obj.object_id, obj))
        if not matches:
            return None
        return sorted(matches, key=lambda item: (item[0], item[1]))[0][2]

    def _create_object(self, observation: ObjectObservation3D) -> MemoryObject3D:
        object_id = f"obj_{self._next_object_index:04d}"
        self._next_object_index += 1
        obj = MemoryObject3D(object_id=object_id, world_xyz=observation.world_xyz.copy())
        self._merge_into(obj, observation)
        return obj

    def _merge_into(self, obj: MemoryObject3D, observation: ObjectObservation3D) -> None:
        next_count = obj.num_observations + 1
        obj.world_xyz = ((obj.world_xyz * obj.num_observations) + observation.world_xyz) / next_count
        obj.observation_xyzs.append(observation.world_xyz.copy())
        obj.num_observations = next_count

        vote_weight = _semantic_vote_weight(observation)
        obj.label_votes[observation.label] = obj.label_votes.get(observation.label, 0.0) + vote_weight
        obj.det_scores.append(clip01(observation.det_score))
        obj.clip_scores.append(clip01(observation.clip_score))
        if observation.fused_2d_score is not None:
            obj.fused_2d_scores.append(clip01(observation.fused_2d_score))
        if observation.view_id and observation.view_id not in obj.view_ids:
            obj.view_ids.append(observation.view_id)
        if observation.point_cloud_path is not None:
            obj.point_cloud_path = observation.point_cloud_path
        _append_observation_metadata(obj, observation)

    def _refresh_confidences(
        self,
        obj: MemoryObject3D,
        confidence_state_before: dict[str, float] | None = None,
    ) -> None:
        obj.semantic_confidence = _semantic_confidence(obj.label_votes)
        obj.geometry_confidence = _geometry_confidence(
            num_points=_mean(obj.metadata.get("num_points_history", [])),
            depth_valid_ratio=_mean(obj.metadata.get("depth_valid_ratio_history", [])),
            min_points=self.config.min_points_for_full_geometry_confidence,
        )
        if not obj.metadata.get("num_points_history"):
            obj.geometry_confidence = 0.0

        view_score = self._view_score(obj)
        consistency_score = _spatial_consistency(obj.observation_xyzs, self.config.merge_distance)
        terms = FusionScoreTerms(
            det_score=_mean(obj.det_scores),
            clip_score=_mean(obj.clip_scores),
            view_score=view_score,
            consistency_score=consistency_score,
            geometry_score=obj.geometry_confidence,
        )
        result: FusionResult = compute_fusion_score(terms, self.config.fusion_weights)
        obj.score_terms = result.terms
        obj.fusion_trace = result.to_json_dict()
        obj.overall_confidence = result.overall_confidence
        self._apply_new_view_support_floor(obj, confidence_state_before)

    def _confidence_update_state(self, obj: MemoryObject3D) -> dict[str, float]:
        """Return confidence terms needed for monotonic support updates."""

        return {
            "overall_confidence": float(obj.overall_confidence),
            "view_score": self._view_score(obj),
        }

    def _view_score(self, obj: MemoryObject3D) -> float:
        return clip01(len(obj.view_ids) / max(1, self.config.max_views_for_full_view_confidence))

    def _apply_new_view_support_floor(
        self,
        obj: MemoryObject3D,
        confidence_state_before: dict[str, float] | None,
    ) -> None:
        raw_overall_confidence = float(obj.overall_confidence)
        previous_floor = clip01(obj.metadata.get("new_view_support_confidence_floor", 0.0))
        if confidence_state_before is None:
            if previous_floor > raw_overall_confidence:
                obj.overall_confidence = previous_floor
                obj.fusion_trace["overall_confidence"] = float(obj.overall_confidence)
                obj.fusion_trace["new_view_support_floor"] = {
                    "applied": True,
                    "raw_overall_confidence": raw_overall_confidence,
                    "confidence_floor": previous_floor,
                    "support_gain": 0.0,
                    "before_view_score": self._view_score(obj),
                    "after_view_score": self._view_score(obj),
                }
            return

        before_view_score = float(confidence_state_before.get("view_score", 0.0))
        after_view_score = self._view_score(obj)
        view_score_gain = max(0.0, after_view_score - before_view_score)
        if view_score_gain <= 0.0 and previous_floor <= raw_overall_confidence:
            return

        weights = self.config.fusion_weights.to_json_dict()
        normalizer = sum(weight for weight in weights.values() if weight > 0.0)
        if normalizer <= 0.0:
            return

        support_gain = view_score_gain * max(0.0, weights["view_score"]) / normalizer
        confidence_floor = max(
            previous_floor,
            clip01(float(confidence_state_before.get("overall_confidence", 0.0)) + support_gain),
        )
        applied = confidence_floor > raw_overall_confidence
        obj.metadata["new_view_support_confidence_floor"] = confidence_floor
        if applied:
            obj.overall_confidence = confidence_floor
            obj.fusion_trace["overall_confidence"] = float(obj.overall_confidence)
        obj.fusion_trace["new_view_support_floor"] = {
            "applied": bool(applied),
            "raw_overall_confidence": raw_overall_confidence,
            "confidence_floor": confidence_floor,
            "support_gain": support_gain,
            "before_view_score": before_view_score,
            "after_view_score": after_view_score,
        }


def observation_from_candidate(
    candidate_3d: Any,
    ranked_candidate: Any,
    view_id: str | None = None,
    label: str | None = None,
) -> ObjectObservation3D:
    """Build a memory observation from existing pipeline candidate objects."""

    world_xyz = getattr(candidate_3d, "world_xyz", None)
    if world_xyz is None:
        raise ValueError("candidate_3d.world_xyz is required for object memory.")
    phrase = label or str(getattr(ranked_candidate, "phrase", "")).strip()
    return ObjectObservation3D(
        world_xyz=np.asarray(world_xyz, dtype=np.float32),
        label=phrase,
        det_score=float(getattr(ranked_candidate, "det_score", 0.0)),
        clip_score=float(getattr(ranked_candidate, "clip_score", 0.0)),
        fused_2d_score=float(getattr(ranked_candidate, "fused_2d_score", 0.0)),
        view_id=view_id,
        num_points=int(getattr(candidate_3d, "num_points", 0)),
        depth_valid_ratio=float(getattr(candidate_3d, "depth_valid_ratio", 0.0)),
        point_cloud_path=getattr(candidate_3d, "point_cloud_path", None),
    )


def _append_observation_metadata(obj: MemoryObject3D, observation: ObjectObservation3D) -> None:
    obj.metadata.setdefault("num_points_history", []).append(int(observation.num_points))
    obj.metadata.setdefault("depth_valid_ratio_history", []).append(float(observation.depth_valid_ratio))


def _validate_xyz(value: Any) -> np.ndarray:
    xyz = np.asarray(value, dtype=np.float32)
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError(f"world_xyz must be a finite shape-(3,) vector, got shape {xyz.shape}.")
    return xyz


def _semantic_vote_weight(observation: ObjectObservation3D) -> float:
    if observation.fused_2d_score is not None:
        return max(clip01(observation.fused_2d_score), 1e-6)
    return max(0.5 * clip01(observation.det_score) + 0.5 * clip01(observation.clip_score), 1e-6)


def _semantic_confidence(label_votes: dict[str, float]) -> float:
    if not label_votes:
        return 0.0
    total = sum(max(score, 0.0) for score in label_votes.values())
    if total <= 0.0:
        return 0.0
    top_vote = max(label_votes.values())
    return clip01(top_vote / total)


def _geometry_confidence(num_points: float, depth_valid_ratio: float, min_points: int) -> float:
    point_score = clip01(num_points / max(1, min_points))
    return 0.5 * point_score + 0.5 * clip01(depth_valid_ratio)


def _spatial_consistency(observation_xyzs: Sequence[np.ndarray], merge_distance: float) -> float:
    if len(observation_xyzs) <= 1:
        return 1.0
    points = np.asarray(observation_xyzs, dtype=np.float32)
    center = np.mean(points, axis=0)
    mean_distance = float(np.mean(np.linalg.norm(points - center, axis=1)))
    return clip01(1.0 - mean_distance / max(merge_distance, 1e-6))


def _mean(values: Sequence[float] | Any) -> float:
    values_list = list(values)
    if not values_list:
        return 0.0
    return float(sum(float(value) for value in values_list) / len(values_list))
