"""Optimization comparison report generator."""

from __future__ import annotations

from pathlib import Path

from model2mobile.models import OptimizationResult


def generate_optimization_report(result: OptimizationResult) -> str:
    """Generate a Markdown report comparing all optimization variants.

    Args:
        result: The aggregated optimization result.

    Returns:
        A Markdown-formatted string.
    """
    sections: list[str] = []

    # Title
    sections.append("# Optimization Report\n")

    # Original baseline
    sections.append("## Baseline\n")
    sections.append("| Property | Value |")
    sections.append("|----------|-------|")
    sections.append(f"| Model Size | {result.original_size_mb:.1f} MB |")
    sections.append(
        f"| Inference Latency | {result.original_inference_ms:.2f} ms |"
    )
    orig_fps = (
        1000.0 / result.original_inference_ms
        if result.original_inference_ms > 0
        else 0.0
    )
    sections.append(f"| Estimated FPS | {orig_fps:.1f} |")

    # Comparison table
    sections.append("\n## Variant Comparison\n")
    sections.append(
        "| Variant | Strategy | Size (MB) | Size Reduction | "
        "Inference (ms) | P95 (ms) | FPS | Speedup | Status |"
    )
    sections.append(
        "|---------|----------|-----------|----------------|"
        "----------------|----------|-----|---------|--------|"
    )

    for v in result.variants:
        if v.error:
            status = f"FAILED: {v.error[:40]}"
            sections.append(
                f"| {v.name} | {v.strategy} | - | - | - | - | - | - | {status} |"
            )
        else:
            rec_marker = " **[recommended]**" if v.name == result.recommended else ""
            sections.append(
                f"| {v.name}{rec_marker} | {v.strategy} | "
                f"{v.model_size_mb:.1f} | {v.size_reduction_pct:+.1f}% | "
                f"{v.inference_mean_ms:.2f} | {v.inference_p95_ms:.2f} | "
                f"{v.estimated_fps:.1f} | {v.speedup_pct:+.1f}% | OK |"
            )

    # Recommendation
    sections.append("\n## Recommendation\n")
    if result.recommended and result.recommended != "none":
        rec_variant = next(
            (v for v in result.variants if v.name == result.recommended), None
        )
        sections.append(f"**Recommended variant:** `{result.recommended}`\n")
        sections.append(f"{result.recommendation_reason}\n")
        if rec_variant and rec_variant.error is None:
            sections.append("### Recommended Variant Details\n")
            sections.append("| Property | Value |")
            sections.append("|----------|-------|")
            sections.append(f"| Strategy | {rec_variant.strategy} |")
            sections.append(f"| Model Size | {rec_variant.model_size_mb:.1f} MB |")
            sections.append(
                f"| Size Reduction | {rec_variant.size_reduction_pct:+.1f}% |"
            )
            sections.append(
                f"| Inference Latency | {rec_variant.inference_mean_ms:.2f} ms |"
            )
            sections.append(f"| P95 Latency | {rec_variant.inference_p95_ms:.2f} ms |")
            sections.append(f"| Estimated FPS | {rec_variant.estimated_fps:.1f} |")
            sections.append(f"| Speedup | {rec_variant.speedup_pct:+.1f}% |")
    else:
        sections.append(
            "No variant could be recommended. "
            f"{result.recommendation_reason}"
        )

    # Usage hint
    sections.append("\n---\n")
    sections.append(
        "> To apply the recommended optimization to your model, use the "
        "corresponding coremltools API in your deployment pipeline."
    )

    return "\n".join(sections)


def save_optimization_report(
    result: OptimizationResult, output_dir: Path
) -> Path:
    """Save the optimization report as a Markdown file.

    Args:
        result: The optimization result.
        output_dir: Directory to write the report to.

    Returns:
        Path to the saved report file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "optimization_report.md"
    path.write_text(generate_optimization_report(result), encoding="utf-8")
    return path
