"""Detection-specific validation checks."""

from __future__ import annotations

import numpy as np

from model2mobile.config import RunConfig
from model2mobile.models import ValidationCheck, ValidationStatus


def _check_detection_count(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
    tolerance_ratio: float = 0.3,
) -> ValidationCheck:
    """Compare detection counts (heuristic: look for variable-length dims)."""

    def _det_count(outputs: dict[str, np.ndarray]) -> int:
        total = 0
        for v in outputs.values():
            if v.ndim >= 2:
                total += int(np.prod(v.shape[1:]))
            else:
                total += v.size
        return total

    pt_count = _det_count(pt_outs)
    cm_count = _det_count(cm_outs)

    if pt_count == 0 and cm_count == 0:
        return ValidationCheck(
            name="detection_count",
            status=ValidationStatus.PASS,
            detail="Both models produced zero detections (or no detection-like output).",
            expected=pt_count,
            actual=cm_count,
        )

    denom = max(pt_count, cm_count, 1)
    ratio = abs(pt_count - cm_count) / denom

    if ratio <= tolerance_ratio:
        return ValidationCheck(
            name="detection_count",
            status=ValidationStatus.PASS,
            detail=f"Detection element counts within {tolerance_ratio*100:.0f}% tolerance.",
            expected=pt_count,
            actual=cm_count,
        )
    return ValidationCheck(
        name="detection_count",
        status=ValidationStatus.WARNING,
        detail=f"Detection element counts differ by {ratio*100:.1f}%.",
        expected=pt_count,
        actual=cm_count,
    )


def _check_confidence_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    tolerance: float,
) -> ValidationCheck:
    """Check that confidence-like values (0-1 range) are consistent."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.size == 0 or cm_arr.size == 0:
            continue

        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)

        pt_in_range = np.all((pt_flat >= -0.01) & (pt_flat <= 1.01))
        cm_in_range = np.all((cm_flat >= -0.01) & (cm_flat <= 1.01))

        if not (pt_in_range and cm_in_range):
            continue

        # Compare element-wise if shapes match
        if pt_flat.shape == cm_flat.shape:
            max_diff = float(np.max(np.abs(pt_flat - cm_flat)))
            mean_diff = float(np.mean(np.abs(pt_flat - cm_flat)))

            if max_diff <= tolerance:
                return ValidationCheck(
                    name="confidence_consistency",
                    status=ValidationStatus.PASS,
                    detail=f"Max confidence diff: {max_diff:.6f} (mean: {mean_diff:.6f}).",
                    tolerance=tolerance,
                )
            if max_diff <= tolerance * 5:
                return ValidationCheck(
                    name="confidence_consistency",
                    status=ValidationStatus.WARNING,
                    detail=f"Max confidence diff: {max_diff:.6f} exceeds tolerance {tolerance}.",
                    tolerance=tolerance,
                )
            return ValidationCheck(
                name="confidence_consistency",
                status=ValidationStatus.FAIL,
                detail=f"Max confidence diff: {max_diff:.6f} far exceeds tolerance {tolerance}.",
                tolerance=tolerance,
            )

        # Shapes differ -- compare distributions
        if pt_flat.size > 0 and cm_flat.size > 0:
            pt_mean, cm_mean = float(np.mean(pt_flat)), float(np.mean(cm_flat))
            diff = abs(pt_mean - cm_mean)
            if diff <= tolerance:
                return ValidationCheck(
                    name="confidence_consistency",
                    status=ValidationStatus.PASS,
                    detail=f"Mean confidence diff: {diff:.6f} (shapes differ).",
                    tolerance=tolerance,
                )
            return ValidationCheck(
                name="confidence_consistency",
                status=ValidationStatus.WARNING,
                detail=f"Mean confidence diff: {diff:.6f}, shapes differ.",
                tolerance=tolerance,
            )

    return ValidationCheck(
        name="confidence_consistency",
        status=ValidationStatus.WARNING,
        detail="No confidence-like output arrays found for comparison.",
        tolerance=tolerance,
    )


def _check_bbox_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    tolerance_px: float,
) -> ValidationCheck:
    """Check bounding-box consistency (arrays not confined to [0,1])."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.size == 0 or cm_arr.size == 0:
            continue

        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)

        # Heuristic: bbox values are typically > 1 (pixel coordinates)
        has_large = np.any(np.abs(pt_flat) > 1.5) or np.any(np.abs(cm_flat) > 1.5)
        if not has_large:
            continue

        if pt_flat.shape == cm_flat.shape:
            max_diff = float(np.max(np.abs(pt_flat - cm_flat)))
            mean_diff = float(np.mean(np.abs(pt_flat - cm_flat)))

            if max_diff <= tolerance_px:
                return ValidationCheck(
                    name="bbox_consistency",
                    status=ValidationStatus.PASS,
                    detail=f"Max bbox diff: {max_diff:.2f}px (mean: {mean_diff:.2f}px).",
                    tolerance=tolerance_px,
                )
            if max_diff <= tolerance_px * 5:
                return ValidationCheck(
                    name="bbox_consistency",
                    status=ValidationStatus.WARNING,
                    detail=f"Max bbox diff: {max_diff:.2f}px exceeds {tolerance_px}px.",
                    tolerance=tolerance_px,
                )
            return ValidationCheck(
                name="bbox_consistency",
                status=ValidationStatus.FAIL,
                detail=f"Max bbox diff: {max_diff:.2f}px far exceeds {tolerance_px}px.",
                tolerance=tolerance_px,
            )

    return ValidationCheck(
        name="bbox_consistency",
        status=ValidationStatus.WARNING,
        detail="No bbox-like output arrays found for element-wise comparison.",
        tolerance=tolerance_px,
    )


def _check_class_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Check class prediction consistency by comparing argmax indices."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 2 or cm_arr.ndim < 2:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        pt_classes = np.argmax(pt_arr, axis=-1).flatten()
        cm_classes = np.argmax(cm_arr, axis=-1).flatten()

        if pt_classes.shape != cm_classes.shape:
            continue

        if pt_classes.size == 0:
            continue

        match_ratio = float(np.mean(pt_classes == cm_classes))

        if match_ratio >= 0.95:
            return ValidationCheck(
                name="class_consistency",
                status=ValidationStatus.PASS,
                detail=f"Class match ratio: {match_ratio*100:.1f}%.",
            )
        if match_ratio >= 0.8:
            return ValidationCheck(
                name="class_consistency",
                status=ValidationStatus.WARNING,
                detail=f"Class match ratio: {match_ratio*100:.1f}%.",
            )
        return ValidationCheck(
            name="class_consistency",
            status=ValidationStatus.FAIL,
            detail=f"Class match ratio: {match_ratio*100:.1f}%.",
        )

    return ValidationCheck(
        name="class_consistency",
        status=ValidationStatus.WARNING,
        detail="No suitable output arrays found for class comparison.",
    )


def validate_detection(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    config: RunConfig,
) -> list[ValidationCheck]:
    """Run all detection-specific validation checks."""
    return [
        _check_detection_count(pt_outs, cm_outs),
        _check_confidence_consistency(pairs, config.confidence_tolerance),
        _check_bbox_consistency(pairs, config.bbox_tolerance),
        _check_class_consistency(pairs),
    ]
