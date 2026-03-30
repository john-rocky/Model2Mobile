"""Markdown report generator for Model2Mobile pipeline results."""

from __future__ import annotations

from pathlib import Path

from model2mobile.models import (
    BenchmarkResult,
    LatencyStats,
    ReadinessState,
    RunResult,
    ValidationResult,
    ValidationStatus,
)


def _readiness_label(state: ReadinessState) -> str:
    labels = {
        ReadinessState.READY: "READY",
        ReadinessState.PARTIAL: "PARTIAL",
        ReadinessState.NOT_READY: "NOT READY",
    }
    return labels.get(state, state.value)


def _status_icon(status: ValidationStatus) -> str:
    icons = {
        ValidationStatus.PASS: "PASS",
        ValidationStatus.WARNING: "WARN",
        ValidationStatus.FAIL: "FAIL",
    }
    return icons.get(status, status.value)


def _fmt_latency(stats: LatencyStats) -> str:
    if stats.samples == 0:
        return "N/A"
    return (
        f"mean={stats.mean_ms:.2f}ms, "
        f"median={stats.median_ms:.2f}ms, "
        f"p95={stats.p95_ms:.2f}ms, "
        f"min={stats.min_ms:.2f}ms, "
        f"max={stats.max_ms:.2f}ms "
        f"(n={stats.samples})"
    )


def _primary_bottleneck(result: RunResult) -> str:
    if result.diagnosis.has_issues:
        return result.diagnosis.primary_category.value
    if result.benchmark and not result.benchmark.success:
        return "runtime_failure"
    if not result.conversion.success:
        return "conversion_failure"
    return "none"


def _section_decision(result: RunResult) -> str:
    lines: list[str] = []
    lines.append("## Decision Summary\n")

    lines.append(f"**Readiness:** `{_readiness_label(result.readiness)}`\n")

    conv_status = "Success" if result.conversion.success else "Failed"
    lines.append(f"- **Conversion:** {conv_status}")

    if result.benchmark:
        rt_status = "Success" if result.benchmark.success else "Failed"
        lines.append(f"- **Runtime:** {rt_status}")
    else:
        lines.append("- **Runtime:** Not evaluated")

    if result.validation:
        lines.append(f"- **Validation:** {result.validation.status.value}")
    else:
        lines.append("- **Validation:** Not evaluated")

    lines.append(f"- **Main Bottleneck:** `{_primary_bottleneck(result)}`")

    if result.suggestions:
        lines.append("\n### Suggested Next Actions\n")
        for i, s in enumerate(result.suggestions, 1):
            lines.append(f"{i}. **{s.title}** (priority {s.priority})")
            lines.append(f"   {s.description}")

    return "\n".join(lines)


def _section_model_info(result: RunResult) -> str:
    info = result.model_info
    lines: list[str] = [
        "## Model Info\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Path | `{info.path}` |",
        f"| Architecture | {info.architecture} |",
        f"| Parameters | {info.parameter_count:,} |",
        f"| Input Shape | {info.input_shape} |",
        f"| Estimated Size | {info.estimated_size_mb:.1f} MB |",
        f"| Dynamic Shapes | {info.has_dynamic_shapes} |",
    ]
    if info.op_summary:
        lines.append("\n### Op Summary\n")
        lines.append("| Op | Count |")
        lines.append("|-----|-------|")
        for op, count in sorted(info.op_summary.items(), key=lambda x: -x[1]):
            lines.append(f"| {op} | {count} |")
    return "\n".join(lines)


def _section_conversion(result: RunResult) -> str:
    conv = result.conversion
    lines: list[str] = [
        "## Conversion Details\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Success | {conv.success} |",
        f"| Compute Unit | {conv.compute_unit} |",
        f"| Conversion Time | {conv.conversion_time_s:.2f}s |",
        f"| CoreML Size | {conv.coreml_size_mb:.1f} MB |",
    ]
    if conv.coreml_path:
        lines.append(f"| CoreML Path | `{conv.coreml_path}` |")
    if conv.error_message:
        lines.append(f"\n**Error:** {conv.error_message}")
    if conv.warnings:
        lines.append("\n### Warnings\n")
        for w in conv.warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


def _section_benchmark(benchmark: BenchmarkResult) -> str:
    lines: list[str] = [
        "## Benchmark Breakdown\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Device | {benchmark.device_name} |",
        f"| Compute Unit | {benchmark.compute_unit} |",
        f"| Warmup Iterations | {benchmark.warmup_iterations} |",
        f"| Measurement Iterations | {benchmark.measurement_iterations} |",
        f"| Estimated FPS | {benchmark.estimated_fps:.1f} |",
    ]
    if benchmark.peak_memory_mb is not None:
        lines.append(f"| Peak Memory | {benchmark.peak_memory_mb:.1f} MB |")

    lines.append("\n### Latency Stages\n")
    lines.append("| Stage | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | Max (ms) |")
    lines.append("|-------|-----------|-------------|----------|----------|----------|")
    for name, stats in [
        ("Preprocess", benchmark.preprocess),
        ("Inference", benchmark.inference),
        ("Postprocess", benchmark.postprocess),
        ("End-to-End", benchmark.end_to_end),
    ]:
        if stats.samples > 0:
            lines.append(
                f"| {name} | {stats.mean_ms:.2f} | {stats.median_ms:.2f} | "
                f"{stats.p95_ms:.2f} | {stats.min_ms:.2f} | {stats.max_ms:.2f} |"
            )

    if benchmark.compute_unit_comparison:
        lines.append("\n### Compute Unit Comparison\n")
        units = list(benchmark.compute_unit_comparison.keys())
        header_cols = " | ".join(units)
        lines.append(f"| Metric | {header_cols} |")
        sep_cols = " | ".join(["-------"] * len(units))
        lines.append(f"|--------|{sep_cols}|")
        # Gather all metric keys
        all_metrics: set[str] = set()
        for data in benchmark.compute_unit_comparison.values():
            all_metrics.update(data.keys())
        for metric in sorted(all_metrics):
            vals = []
            for unit in units:
                v = benchmark.compute_unit_comparison[unit].get(metric)
                vals.append(f"{v:.2f}" if v is not None else "N/A")
            lines.append(f"| {metric} | {' | '.join(vals)} |")

    if benchmark.error_message:
        lines.append(f"\n**Error:** {benchmark.error_message}")

    return "\n".join(lines)


def _section_validation(validation: ValidationResult) -> str:
    lines: list[str] = [
        "## Validation Details\n",
        f"**Overall Status:** `{validation.status.value}` "
        f"({validation.pass_count} pass, "
        f"{validation.warning_count} warning, "
        f"{validation.fail_count} fail)\n",
    ]
    if validation.checks:
        lines.append("| Check | Status | Detail |")
        lines.append("|-------|--------|--------|")
        for c in validation.checks:
            status_label = _status_icon(c.status)
            detail = c.detail.replace("|", "\\|")
            lines.append(f"| {c.name} | {status_label} | {detail} |")
    if validation.error_message:
        lines.append(f"\n**Error:** {validation.error_message}")
    return "\n".join(lines)


def _section_diagnosis(result: RunResult) -> str:
    diag = result.diagnosis
    lines: list[str] = ["## Diagnosis Details\n"]
    if not diag.has_issues:
        lines.append("No issues detected.")
        return "\n".join(lines)

    lines.append(f"**Primary Category:** `{diag.primary_category.value}`\n")
    for i, d in enumerate(diag.diagnoses, 1):
        lines.append(f"### Issue {i}: {d.category.value}\n")
        lines.append(f"- **Likely Cause:** {d.likely_cause}")
        lines.append(f"- **Raw Error:** `{d.raw_error}`")
        if d.suggested_steps:
            lines.append("- **Steps:**")
            for step in d.suggested_steps:
                lines.append(f"  - {step}")
        lines.append("")
    return "\n".join(lines)


def _section_suggestions(result: RunResult) -> str:
    lines: list[str] = ["## Full Suggestions\n"]
    if not result.suggestions:
        lines.append("No additional suggestions.")
        return "\n".join(lines)

    for s in sorted(result.suggestions, key=lambda x: x.priority):
        lines.append(f"### [{s.priority}] {s.title}\n")
        lines.append(f"{s.description}\n")
    return "\n".join(lines)


def _section_metadata(result: RunResult) -> str:
    lines: list[str] = [
        "## Run Metadata\n",
        f"| Property | Value |",
        f"|----------|-------|",
        f"| Run ID | `{result.run_id}` |",
        f"| Timestamp | {result.timestamp} |",
        f"| Output Dir | `{result.output_dir}` |",
        f"| Input Shape | {result.model_info.input_shape} |",
    ]
    if result.benchmark:
        lines.append(f"| Device | {result.benchmark.device_name} |")
        lines.append(f"| Compute Unit | {result.benchmark.compute_unit} |")
    return "\n".join(lines)


def generate_markdown(result: RunResult) -> str:
    sections: list[str] = [
        f"# Model2Mobile Report\n",
        _section_decision(result),
        _section_model_info(result),
        _section_conversion(result),
    ]

    if result.benchmark:
        sections.append(_section_benchmark(result.benchmark))

    if result.validation:
        sections.append(_section_validation(result.validation))

    sections.append(_section_diagnosis(result))
    sections.append(_section_suggestions(result))
    sections.append(_section_metadata(result))

    sections.append(
        "---\n\n"
        "> **Note:** These results are specific to the evaluated scenario "
        "(model, input size, device, compute unit). "
        "Different configurations may yield different results.\n"
    )

    return "\n\n".join(sections)


def save_markdown(result: RunResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.md"
    path.write_text(generate_markdown(result), encoding="utf-8")
    return path
