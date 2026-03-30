"""Optimization module: tries multiple quantization strategies and benchmarks each."""

from __future__ import annotations

import logging
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path

import coremltools as ct
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from model2mobile.benchmark.runner import run_benchmark
from model2mobile.config import RunConfig
from model2mobile.models import OptimizationResult, OptimizationVariant

logger = logging.getLogger(__name__)
console = Console()


def _get_model_size_mb(model_path: str) -> float:
    """Compute the total file size of an .mlpackage in MB."""
    p = Path(model_path)
    if p.is_dir():
        total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    else:
        total = p.stat().st_size
    return total / (1024 * 1024)


def _save_variant(model: ct.models.MLModel, tmp_dir: Path, name: str) -> str:
    """Save a Core ML model variant and return its path."""
    out_path = tmp_dir / f"{name}.mlpackage"
    model.save(str(out_path))
    return str(out_path)


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

_STRATEGIES: list[dict] = [
    {
        "name": "float16",
        "description": "Float16 precision (default Apple recommendation)",
        "apply": lambda model: _apply_float16(model),
    },
    {
        "name": "int8_linear",
        "description": "Linear quantization to 8-bit integer weights",
        "apply": lambda model: _apply_linear_quantize(model, nbits=8),
    },
    {
        "name": "int4_linear",
        "description": "Linear quantization to 4-bit integer weights",
        "apply": lambda model: _apply_linear_quantize(model, nbits=4),
    },
    {
        "name": "palettize_6bit",
        "description": "Palette-based weight compression with 6-bit lookup",
        "apply": lambda model: _apply_palettize(model, nbits=6),
    },
    {
        "name": "palettize_4bit",
        "description": "Palette-based weight compression with 4-bit lookup",
        "apply": lambda model: _apply_palettize(model, nbits=4),
    },
]


def _apply_float16(model: ct.models.MLModel) -> ct.models.MLModel:
    """Ensure model weights are in float16 precision.

    ML Program models (.mlpackage) are typically already float16 after
    ct.convert (the default compute_precision).  This strategy returns the
    model as-is so it serves as a baseline for comparing quantized variants.
    """
    # Models converted with ct.convert default to float16 for mlprogram.
    # Nothing extra to apply -- just return the model for benchmarking.
    return model


def _apply_linear_quantize(
    model: ct.models.MLModel, nbits: int
) -> ct.models.MLModel:
    """Apply post-training linear weight quantization.

    Args:
        model: The Core ML model to quantize.
        nbits: Bit width for quantization (4 or 8).
    """
    from coremltools.optimize.coreml import (
        OpLinearQuantizerConfig,
        OptimizationConfig,
        linear_quantize_weights,
    )

    dtype_str = f"int{nbits}"
    op_config = OpLinearQuantizerConfig(
        mode="linear_symmetric",
        dtype=dtype_str,
        weight_threshold=512,
    )
    config = OptimizationConfig(global_config=op_config)
    return linear_quantize_weights(model, config)


def _apply_palettize(
    model: ct.models.MLModel, nbits: int
) -> ct.models.MLModel:
    """Apply post-training palette-based weight compression.

    Args:
        model: The Core ML model to palettize.
        nbits: Number of bits for the palette lookup table.
    """
    from coremltools.optimize.coreml import (
        OpPalettizerConfig,
        OptimizationConfig,
        palettize_weights,
    )

    op_config = OpPalettizerConfig(nbits=nbits, weight_threshold=512)
    config = OptimizationConfig(global_config=op_config)
    return palettize_weights(model, config)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_optimization(
    coreml_path: str,
    config: RunConfig,
    target_fps: float | None = None,
) -> OptimizationResult:
    """Try multiple optimization strategies, benchmark each, and recommend the best.

    Args:
        coreml_path: Path to the original .mlpackage.
        config: Run configuration (used for benchmark settings).
        target_fps: Optional target FPS; influences recommendation.

    Returns:
        OptimizationResult with all variant results and a recommendation.
    """
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    original_size_mb = _get_model_size_mb(coreml_path)

    # Benchmark the original model
    console.print("\n[bold cyan]Benchmarking original model...[/bold cyan]")
    original_bench = run_benchmark(coreml_path, config)
    if not original_bench.success:
        console.print(
            f"[red]Original model benchmark failed:[/red] {original_bench.error_message}"
        )
        return OptimizationResult(
            original_size_mb=round(original_size_mb, 2),
            original_inference_ms=0.0,
            recommended="none",
            recommendation_reason="Benchmark of original model failed.",
        )

    original_inference_ms = original_bench.inference.mean_ms
    original_fps = original_bench.estimated_fps

    console.print(
        f"  Original: {original_size_mb:.1f} MB | "
        f"inference {original_inference_ms:.2f}ms | "
        f"FPS {original_fps:.1f}"
    )

    variants: list[OptimizationVariant] = []
    tmp_root = Path(tempfile.mkdtemp(prefix="m2m_optimize_"))

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            for strategy in _STRATEGIES:
                name = strategy["name"]
                desc = strategy["description"]
                task = progress.add_task(f"Trying {name}...", total=None)

                try:
                    # Load the original model fresh each time
                    model = ct.models.MLModel(coreml_path, compute_units=ct.ComputeUnit.ALL)

                    # Apply the optimization strategy
                    optimized = strategy["apply"](model)

                    # Save the variant
                    variant_path = _save_variant(optimized, tmp_root, name)
                    variant_size = _get_model_size_mb(variant_path)

                    # Benchmark the variant
                    bench = run_benchmark(variant_path, config)

                    if bench.success:
                        size_reduction = (
                            (original_size_mb - variant_size) / original_size_mb * 100
                            if original_size_mb > 0
                            else 0.0
                        )
                        speedup = (
                            (original_inference_ms - bench.inference.mean_ms)
                            / original_inference_ms
                            * 100
                            if original_inference_ms > 0
                            else 0.0
                        )

                        variant = OptimizationVariant(
                            name=name,
                            strategy=desc,
                            model_size_mb=round(variant_size, 2),
                            inference_mean_ms=round(bench.inference.mean_ms, 2),
                            inference_p95_ms=round(bench.inference.p95_ms, 2),
                            estimated_fps=round(bench.estimated_fps, 1),
                            size_reduction_pct=round(size_reduction, 1),
                            speedup_pct=round(speedup, 1),
                        )
                    else:
                        variant = OptimizationVariant(
                            name=name,
                            strategy=desc,
                            model_size_mb=round(variant_size, 2),
                            inference_mean_ms=0.0,
                            inference_p95_ms=0.0,
                            estimated_fps=0.0,
                            size_reduction_pct=0.0,
                            speedup_pct=0.0,
                            error=bench.error_message,
                        )

                    variants.append(variant)

                except Exception as exc:
                    logger.debug("Strategy %s failed: %s", name, traceback.format_exc())
                    variants.append(
                        OptimizationVariant(
                            name=name,
                            strategy=desc,
                            model_size_mb=0.0,
                            inference_mean_ms=0.0,
                            inference_p95_ms=0.0,
                            estimated_fps=0.0,
                            size_reduction_pct=0.0,
                            speedup_pct=0.0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                finally:
                    progress.update(task, completed=True)

    finally:
        # Clean up temporary directory
        shutil.rmtree(tmp_root, ignore_errors=True)

    # Determine the best variant
    recommended, reason = _pick_recommended(
        variants, original_inference_ms, original_size_mb, target_fps
    )

    return OptimizationResult(
        original_size_mb=round(original_size_mb, 2),
        original_inference_ms=round(original_inference_ms, 2),
        variants=variants,
        recommended=recommended,
        recommendation_reason=reason,
    )


def _pick_recommended(
    variants: list[OptimizationVariant],
    original_ms: float,
    original_size_mb: float,
    target_fps: float | None,
) -> tuple[str, str]:
    """Select the best variant balancing size, speed, and reliability."""
    successful = [v for v in variants if v.error is None and v.estimated_fps > 0]

    if not successful:
        return "none", "All optimization strategies failed or could not be benchmarked."

    # If target FPS is specified, filter to variants that meet it
    if target_fps is not None:
        meeting_target = [v for v in successful if v.estimated_fps >= target_fps]
        if meeting_target:
            # Among those meeting the target, pick the smallest
            best = min(meeting_target, key=lambda v: v.model_size_mb)
            return best.name, (
                f"Meets target FPS ({target_fps:.0f}) with "
                f"{best.size_reduction_pct:.0f}% size reduction "
                f"({best.model_size_mb:.1f} MB, {best.estimated_fps:.1f} FPS)."
            )

    # Score each variant: weighted combination of size reduction and speed
    # Higher score = better
    def _score(v: OptimizationVariant) -> float:
        # Normalize: size reduction 0-100%, speedup can be negative
        size_score = max(v.size_reduction_pct, 0.0)
        speed_score = max(v.speedup_pct, -50.0)  # cap penalty
        # Weight size reduction slightly more (deployment-focused)
        return size_score * 0.6 + speed_score * 0.4

    best = max(successful, key=_score)

    parts: list[str] = []
    if best.size_reduction_pct > 0:
        parts.append(f"{best.size_reduction_pct:.0f}% smaller ({best.model_size_mb:.1f} MB)")
    if best.speedup_pct > 0:
        parts.append(f"{best.speedup_pct:.0f}% faster ({best.inference_mean_ms:.1f}ms)")
    elif best.speedup_pct < -5:
        parts.append(
            f"{abs(best.speedup_pct):.0f}% slower but acceptable "
            f"({best.inference_mean_ms:.1f}ms)"
        )

    reason = "Best overall balance. " + ", ".join(parts) + "." if parts else "Best overall balance."
    return best.name, reason
