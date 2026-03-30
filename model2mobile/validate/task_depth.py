"""Depth estimation-specific validation checks."""

from __future__ import annotations

import numpy as np

from model2mobile.config import RunConfig
from model2mobile.models import ValidationCheck, ValidationStatus


def _check_depth_shape(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
) -> ValidationCheck:
    """Output spatial dimensions should match between models."""
    pt_shapes = {k: v.shape for k, v in pt_outs.items()}
    cm_shapes = {k: v.shape for k, v in cm_outs.items()}

    # Find spatial outputs (4-D tensors with 1 channel: N, 1, H, W)
    pt_depth = {k: s for k, s in pt_shapes.items() if len(s) >= 3}
    cm_depth = {k: s for k, s in cm_shapes.items() if len(s) >= 3}

    if not pt_depth and not cm_depth:
        return ValidationCheck(
            name="depth_shape",
            status=ValidationStatus.WARNING,
            detail="No spatial outputs found in either model.",
        )

    if not pt_depth or not cm_depth:
        missing = "PyTorch" if not pt_depth else "CoreML"
        return ValidationCheck(
            name="depth_shape",
            status=ValidationStatus.FAIL,
            detail=f"{missing} has no spatial output for depth map.",
            expected=str(pt_depth),
            actual=str(cm_depth),
        )

    # Compare spatial dims of the largest output
    pt_main = max(pt_depth.values(), key=lambda s: int(np.prod(s[-2:])))
    cm_main = max(cm_depth.values(), key=lambda s: int(np.prod(s[-2:])))

    if pt_main[-2:] == cm_main[-2:]:
        return ValidationCheck(
            name="depth_shape",
            status=ValidationStatus.PASS,
            detail=f"Depth map spatial dimensions match: {pt_main[-2:]}.",
            expected=str(pt_main),
            actual=str(cm_main),
        )

    return ValidationCheck(
        name="depth_shape",
        status=ValidationStatus.WARNING,
        detail=f"Depth spatial dimensions differ: PT={pt_main[-2:]} vs CM={cm_main[-2:]}.",
        expected=str(pt_main),
        actual=str(cm_main),
    )


def _check_relative_error(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Mean relative error between depth maps."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 3 or cm_arr.ndim < 3:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)

        # Avoid division by zero: only compare pixels with non-trivial depth
        mask = np.abs(pt_flat) > 1e-6
        if np.sum(mask) == 0:
            continue

        rel_error = np.abs(pt_flat[mask] - cm_flat[mask]) / np.abs(pt_flat[mask])
        mean_rel = float(np.mean(rel_error))

        if mean_rel <= 0.05:
            return ValidationCheck(
                name="relative_error",
                status=ValidationStatus.PASS,
                detail=f"Mean relative error: {mean_rel:.6f} ({mean_rel*100:.2f}%).",
            )
        if mean_rel <= 0.15:
            return ValidationCheck(
                name="relative_error",
                status=ValidationStatus.WARNING,
                detail=f"Mean relative error: {mean_rel:.6f} ({mean_rel*100:.2f}%).",
            )
        return ValidationCheck(
            name="relative_error",
            status=ValidationStatus.FAIL,
            detail=f"Mean relative error: {mean_rel:.6f} ({mean_rel*100:.2f}%).",
        )

    return ValidationCheck(
        name="relative_error",
        status=ValidationStatus.WARNING,
        detail="No suitable spatial output arrays found for relative error comparison.",
    )


def _check_structural_similarity(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Simplified structural similarity: normalized correlation between depth maps."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 3 or cm_arr.ndim < 3:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)

        # Zero-mean normalized cross-correlation
        pt_centered = pt_flat - np.mean(pt_flat)
        cm_centered = cm_flat - np.mean(cm_flat)

        norm_pt = np.linalg.norm(pt_centered)
        norm_cm = np.linalg.norm(cm_centered)

        if norm_pt < 1e-10 or norm_cm < 1e-10:
            # Both are nearly constant -- consider them matching
            if norm_pt < 1e-10 and norm_cm < 1e-10:
                return ValidationCheck(
                    name="structural_similarity",
                    status=ValidationStatus.PASS,
                    detail="Both depth maps are near-constant; treated as matching.",
                )
            return ValidationCheck(
                name="structural_similarity",
                status=ValidationStatus.WARNING,
                detail="One depth map is near-constant while the other is not.",
            )

        ncc = float(np.dot(pt_centered, cm_centered) / (norm_pt * norm_cm))

        if ncc >= 0.99:
            return ValidationCheck(
                name="structural_similarity",
                status=ValidationStatus.PASS,
                detail=f"Normalized cross-correlation: {ncc:.6f}.",
            )
        if ncc >= 0.95:
            return ValidationCheck(
                name="structural_similarity",
                status=ValidationStatus.WARNING,
                detail=f"Normalized cross-correlation: {ncc:.6f}.",
            )
        return ValidationCheck(
            name="structural_similarity",
            status=ValidationStatus.FAIL,
            detail=f"Normalized cross-correlation: {ncc:.6f}.",
        )

    return ValidationCheck(
        name="structural_similarity",
        status=ValidationStatus.WARNING,
        detail="No suitable spatial output arrays found for structural similarity.",
    )


def _check_scale_invariant(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Check if depth maps are consistent up to a global scale factor."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 3 or cm_arr.ndim < 3:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)

        # Only use pixels with significant depth values
        mask = (np.abs(pt_flat) > 1e-6) & (np.abs(cm_flat) > 1e-6)
        if np.sum(mask) < 10:
            continue

        pt_valid = pt_flat[mask]
        cm_valid = cm_flat[mask]

        # Compute optimal scale factor: s = median(cm / pt)
        ratios = cm_valid / pt_valid
        scale = float(np.median(ratios))

        if scale <= 0:
            return ValidationCheck(
                name="scale_invariant",
                status=ValidationStatus.FAIL,
                detail="Negative scale factor detected; depth maps may be inverted.",
            )

        # After scaling, compute residual error
        scaled_pt = pt_valid * scale
        residual = np.abs(scaled_pt - cm_valid) / np.abs(cm_valid)
        mean_residual = float(np.mean(residual))

        scale_info = f"scale={scale:.4f}"
        if abs(scale - 1.0) < 0.01:
            scale_info = "scale~1.0 (no rescaling needed)"

        if mean_residual <= 0.05:
            return ValidationCheck(
                name="scale_invariant",
                status=ValidationStatus.PASS,
                detail=f"Scale-invariant residual error: {mean_residual:.6f} ({scale_info}).",
            )
        if mean_residual <= 0.15:
            return ValidationCheck(
                name="scale_invariant",
                status=ValidationStatus.WARNING,
                detail=f"Scale-invariant residual error: {mean_residual:.6f} ({scale_info}).",
            )
        return ValidationCheck(
            name="scale_invariant",
            status=ValidationStatus.FAIL,
            detail=f"Scale-invariant residual error: {mean_residual:.6f} ({scale_info}).",
        )

    return ValidationCheck(
        name="scale_invariant",
        status=ValidationStatus.WARNING,
        detail="No suitable spatial output arrays found for scale-invariant comparison.",
    )


def validate_depth(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    config: RunConfig,
) -> list[ValidationCheck]:
    """Run all depth estimation-specific validation checks."""
    return [
        _check_depth_shape(pt_outs, cm_outs),
        _check_relative_error(pairs),
        _check_structural_similarity(pairs),
        _check_scale_invariant(pairs),
    ]
