"""Classification-specific validation checks."""

from __future__ import annotations

import numpy as np

from model2mobile.config import RunConfig
from model2mobile.models import ValidationCheck, ValidationStatus


def _softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax along the last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def _check_top1_accuracy(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Compare argmax of outputs -- top-1 predictions should match."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 2 or cm_arr.ndim < 2:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        pt_top1 = np.argmax(pt_arr, axis=-1).flatten()
        cm_top1 = np.argmax(cm_arr, axis=-1).flatten()

        if pt_top1.size == 0:
            continue

        match_ratio = float(np.mean(pt_top1 == cm_top1))

        if match_ratio >= 1.0:
            return ValidationCheck(
                name="top1_accuracy",
                status=ValidationStatus.PASS,
                detail=f"Top-1 predictions match perfectly ({match_ratio*100:.0f}%).",
            )
        if match_ratio >= 0.9:
            return ValidationCheck(
                name="top1_accuracy",
                status=ValidationStatus.WARNING,
                detail=f"Top-1 match ratio: {match_ratio*100:.1f}%.",
            )
        return ValidationCheck(
            name="top1_accuracy",
            status=ValidationStatus.FAIL,
            detail=f"Top-1 match ratio: {match_ratio*100:.1f}%.",
        )

    return ValidationCheck(
        name="top1_accuracy",
        status=ValidationStatus.WARNING,
        detail="No suitable output arrays found for top-1 comparison.",
    )


def _check_top5_overlap(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Compare top-5 class indices -- should overlap significantly."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 2 or cm_arr.ndim < 2:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        num_classes = pt_arr.shape[-1]
        k = min(5, num_classes)

        # For each sample, compute top-k overlap
        overlaps = []
        for i in range(pt_arr.shape[0]):
            pt_topk = set(np.argsort(pt_arr[i].flatten())[-k:])
            cm_topk = set(np.argsort(cm_arr[i].flatten())[-k:])
            overlap = len(pt_topk & cm_topk) / k
            overlaps.append(overlap)

        if not overlaps:
            continue

        mean_overlap = float(np.mean(overlaps))

        if mean_overlap >= 0.8:
            return ValidationCheck(
                name="top5_overlap",
                status=ValidationStatus.PASS,
                detail=f"Top-{k} overlap: {mean_overlap*100:.1f}%.",
            )
        if mean_overlap >= 0.6:
            return ValidationCheck(
                name="top5_overlap",
                status=ValidationStatus.WARNING,
                detail=f"Top-{k} overlap: {mean_overlap*100:.1f}%.",
            )
        return ValidationCheck(
            name="top5_overlap",
            status=ValidationStatus.FAIL,
            detail=f"Top-{k} overlap: {mean_overlap*100:.1f}%.",
        )

    return ValidationCheck(
        name="top5_overlap",
        status=ValidationStatus.WARNING,
        detail="No suitable output arrays found for top-5 comparison.",
    )


def _check_probability_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Compare softmax probability distributions using cosine similarity."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 2 or cm_arr.ndim < 2:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        pt_probs = _softmax(pt_arr.astype(np.float64))
        cm_probs = _softmax(cm_arr.astype(np.float64))

        # Compute cosine similarity per sample, then average
        similarities = []
        for i in range(pt_probs.shape[0]):
            pt_flat = pt_probs[i].flatten()
            cm_flat = cm_probs[i].flatten()
            norm_pt = np.linalg.norm(pt_flat)
            norm_cm = np.linalg.norm(cm_flat)
            if norm_pt > 0 and norm_cm > 0:
                cos_sim = float(np.dot(pt_flat, cm_flat) / (norm_pt * norm_cm))
                similarities.append(cos_sim)

        if not similarities:
            continue

        mean_sim = float(np.mean(similarities))

        if mean_sim >= 0.99:
            return ValidationCheck(
                name="probability_consistency",
                status=ValidationStatus.PASS,
                detail=f"Probability distribution cosine similarity: {mean_sim:.6f}.",
            )
        if mean_sim >= 0.95:
            return ValidationCheck(
                name="probability_consistency",
                status=ValidationStatus.WARNING,
                detail=f"Probability distribution cosine similarity: {mean_sim:.6f}.",
            )
        return ValidationCheck(
            name="probability_consistency",
            status=ValidationStatus.FAIL,
            detail=f"Probability distribution cosine similarity: {mean_sim:.6f}.",
        )

    return ValidationCheck(
        name="probability_consistency",
        status=ValidationStatus.WARNING,
        detail="No suitable output arrays found for probability comparison.",
    )


def _check_confidence_delta(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Max absolute difference in class probabilities."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 2 or cm_arr.ndim < 2:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        pt_probs = _softmax(pt_arr.astype(np.float64))
        cm_probs = _softmax(cm_arr.astype(np.float64))

        max_diff = float(np.max(np.abs(pt_probs - cm_probs)))
        mean_diff = float(np.mean(np.abs(pt_probs - cm_probs)))

        if max_diff <= 0.01:
            return ValidationCheck(
                name="confidence_delta",
                status=ValidationStatus.PASS,
                detail=f"Max probability diff: {max_diff:.6f} (mean: {mean_diff:.6f}).",
            )
        if max_diff <= 0.05:
            return ValidationCheck(
                name="confidence_delta",
                status=ValidationStatus.WARNING,
                detail=f"Max probability diff: {max_diff:.6f} (mean: {mean_diff:.6f}).",
            )
        return ValidationCheck(
            name="confidence_delta",
            status=ValidationStatus.FAIL,
            detail=f"Max probability diff: {max_diff:.6f} (mean: {mean_diff:.6f}).",
        )

    return ValidationCheck(
        name="confidence_delta",
        status=ValidationStatus.WARNING,
        detail="No suitable output arrays found for confidence delta comparison.",
    )


def validate_classification(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    config: RunConfig,
) -> list[ValidationCheck]:
    """Run all classification-specific validation checks."""
    return [
        _check_top1_accuracy(pairs),
        _check_top5_overlap(pairs),
        _check_probability_consistency(pairs),
        _check_confidence_delta(pairs),
    ]
