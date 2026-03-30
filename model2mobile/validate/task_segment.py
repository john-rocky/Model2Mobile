"""Segmentation-specific validation checks."""

from __future__ import annotations

import numpy as np

from model2mobile.config import RunConfig
from model2mobile.models import ValidationCheck, ValidationStatus


def _check_mask_shape(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
) -> ValidationCheck:
    """Output spatial dimensions should match between models."""
    pt_shapes = {k: v.shape for k, v in pt_outs.items()}
    cm_shapes = {k: v.shape for k, v in cm_outs.items()}

    # Find spatial outputs (4-D tensors: N, C, H, W)
    pt_spatial = {k: s for k, s in pt_shapes.items() if len(s) == 4}
    cm_spatial = {k: s for k, s in cm_shapes.items() if len(s) == 4}

    if not pt_spatial and not cm_spatial:
        return ValidationCheck(
            name="mask_shape",
            status=ValidationStatus.WARNING,
            detail="No 4-D spatial outputs found in either model.",
        )

    if not pt_spatial or not cm_spatial:
        missing = "PyTorch" if not pt_spatial else "CoreML"
        return ValidationCheck(
            name="mask_shape",
            status=ValidationStatus.FAIL,
            detail=f"{missing} has no 4-D spatial output for segmentation mask.",
            expected=str(pt_spatial),
            actual=str(cm_spatial),
        )

    # Compare spatial dims (H, W) of the largest output
    pt_main = max(pt_spatial.values(), key=lambda s: s[2] * s[3])
    cm_main = max(cm_spatial.values(), key=lambda s: s[2] * s[3])

    if pt_main[2:] == cm_main[2:]:
        return ValidationCheck(
            name="mask_shape",
            status=ValidationStatus.PASS,
            detail=f"Spatial dimensions match: {pt_main[2:]}.",
            expected=str(pt_main),
            actual=str(cm_main),
        )

    return ValidationCheck(
        name="mask_shape",
        status=ValidationStatus.WARNING,
        detail=f"Spatial dimensions differ: PT={pt_main[2:]} vs CM={cm_main[2:]}.",
        expected=str(pt_main),
        actual=str(cm_main),
    )


def _check_class_iou(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Mean IoU between PyTorch and CoreML segmentation masks."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim != 4 or cm_arr.ndim != 4:
            continue
        if pt_arr.shape != cm_arr.shape:
            # Try to compare if spatial dims match despite channel differences
            if pt_arr.shape[2:] != cm_arr.shape[2:]:
                continue

        # Compute per-pixel class labels via argmax over channels
        pt_mask = np.argmax(pt_arr, axis=1)  # (N, H, W)
        cm_mask = np.argmax(cm_arr, axis=1)

        if pt_mask.shape != cm_mask.shape:
            continue

        num_classes = max(int(pt_arr.shape[1]), int(cm_arr.shape[1]))
        ious: list[float] = []

        for c in range(num_classes):
            pt_c = (pt_mask == c)
            cm_c = (cm_mask == c)
            intersection = np.sum(pt_c & cm_c)
            union = np.sum(pt_c | cm_c)
            if union > 0:
                ious.append(float(intersection) / float(union))

        if not ious:
            continue

        mean_iou = float(np.mean(ious))

        if mean_iou >= 0.9:
            return ValidationCheck(
                name="class_iou",
                status=ValidationStatus.PASS,
                detail=f"Mean IoU: {mean_iou:.4f} across {len(ious)} classes.",
            )
        if mean_iou >= 0.7:
            return ValidationCheck(
                name="class_iou",
                status=ValidationStatus.WARNING,
                detail=f"Mean IoU: {mean_iou:.4f} across {len(ious)} classes.",
            )
        return ValidationCheck(
            name="class_iou",
            status=ValidationStatus.FAIL,
            detail=f"Mean IoU: {mean_iou:.4f} across {len(ious)} classes.",
        )

    return ValidationCheck(
        name="class_iou",
        status=ValidationStatus.WARNING,
        detail="No suitable 4-D output arrays found for IoU comparison.",
    )


def _check_pixel_accuracy(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Percentage of pixels with matching class labels."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim != 4 or cm_arr.ndim != 4:
            continue
        if pt_arr.shape[2:] != cm_arr.shape[2:]:
            continue

        pt_mask = np.argmax(pt_arr, axis=1)  # (N, H, W)
        cm_mask = np.argmax(cm_arr, axis=1)

        if pt_mask.shape != cm_mask.shape:
            continue

        total_pixels = pt_mask.size
        if total_pixels == 0:
            continue

        correct = np.sum(pt_mask == cm_mask)
        accuracy = float(correct) / float(total_pixels)

        if accuracy >= 0.95:
            return ValidationCheck(
                name="pixel_accuracy",
                status=ValidationStatus.PASS,
                detail=f"Pixel accuracy: {accuracy*100:.2f}%.",
            )
        if accuracy >= 0.85:
            return ValidationCheck(
                name="pixel_accuracy",
                status=ValidationStatus.WARNING,
                detail=f"Pixel accuracy: {accuracy*100:.2f}%.",
            )
        return ValidationCheck(
            name="pixel_accuracy",
            status=ValidationStatus.FAIL,
            detail=f"Pixel accuracy: {accuracy*100:.2f}%.",
        )

    return ValidationCheck(
        name="pixel_accuracy",
        status=ValidationStatus.WARNING,
        detail="No suitable 4-D output arrays found for pixel accuracy comparison.",
    )


def _check_boundary_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    boundary_tolerance_px: int = 3,
) -> ValidationCheck:
    """Check that edges of segmentation are consistent (within a few pixels)."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim != 4 or cm_arr.ndim != 4:
            continue
        if pt_arr.shape[2:] != cm_arr.shape[2:]:
            continue

        pt_mask = np.argmax(pt_arr, axis=1)  # (N, H, W)
        cm_mask = np.argmax(cm_arr, axis=1)

        if pt_mask.shape != cm_mask.shape:
            continue

        # Detect boundary pixels using simple gradient (Sobel-like)
        def _boundary(mask: np.ndarray) -> np.ndarray:
            """Return boolean mask of boundary pixels."""
            padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode="edge")
            dx = padded[:, 1:-1, 2:] != padded[:, 1:-1, :-2]
            dy = padded[:, 2:, 1:-1] != padded[:, :-2, 1:-1]
            return dx | dy

        pt_boundary = _boundary(pt_mask)
        cm_boundary = _boundary(cm_mask)

        # Dilate boundaries by tolerance to allow small shifts
        from scipy.ndimage import binary_dilation  # type: ignore[import-untyped]

        struct = np.ones((1, 2 * boundary_tolerance_px + 1, 2 * boundary_tolerance_px + 1), dtype=bool)

        pt_dilated = binary_dilation(pt_boundary, structure=struct)
        cm_dilated = binary_dilation(cm_boundary, structure=struct)

        # Check how many CoreML boundary pixels fall within dilated PyTorch boundaries
        if np.sum(cm_boundary) == 0 and np.sum(pt_boundary) == 0:
            return ValidationCheck(
                name="boundary_consistency",
                status=ValidationStatus.PASS,
                detail="No boundaries detected in either output (uniform masks).",
            )

        if np.sum(cm_boundary) == 0 or np.sum(pt_boundary) == 0:
            return ValidationCheck(
                name="boundary_consistency",
                status=ValidationStatus.WARNING,
                detail="Boundaries detected in one model but not the other.",
            )

        cm_in_pt = float(np.sum(cm_boundary & pt_dilated)) / float(np.sum(cm_boundary))
        pt_in_cm = float(np.sum(pt_boundary & cm_dilated)) / float(np.sum(pt_boundary))
        consistency = min(cm_in_pt, pt_in_cm)

        if consistency >= 0.9:
            return ValidationCheck(
                name="boundary_consistency",
                status=ValidationStatus.PASS,
                detail=f"Boundary consistency: {consistency*100:.1f}% (tolerance: {boundary_tolerance_px}px).",
            )
        if consistency >= 0.7:
            return ValidationCheck(
                name="boundary_consistency",
                status=ValidationStatus.WARNING,
                detail=f"Boundary consistency: {consistency*100:.1f}% (tolerance: {boundary_tolerance_px}px).",
            )
        return ValidationCheck(
            name="boundary_consistency",
            status=ValidationStatus.FAIL,
            detail=f"Boundary consistency: {consistency*100:.1f}% (tolerance: {boundary_tolerance_px}px).",
        )

    return ValidationCheck(
        name="boundary_consistency",
        status=ValidationStatus.WARNING,
        detail="No suitable 4-D output arrays found for boundary comparison.",
    )


def validate_segmentation(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    config: RunConfig,
) -> list[ValidationCheck]:
    """Run all segmentation-specific validation checks."""
    checks = [
        _check_mask_shape(pt_outs, cm_outs),
        _check_class_iou(pairs),
        _check_pixel_accuracy(pairs),
    ]

    # Boundary check requires scipy; skip gracefully if unavailable
    try:
        checks.append(_check_boundary_consistency(pairs))
    except ImportError:
        checks.append(ValidationCheck(
            name="boundary_consistency",
            status=ValidationStatus.WARNING,
            detail="scipy not available; boundary consistency check skipped.",
        ))

    return checks
