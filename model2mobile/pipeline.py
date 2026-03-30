"""Pipeline orchestration for the full evaluation run."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from model2mobile.config import RunConfig
from model2mobile.models import (
    BenchmarkResult,
    ConversionResult,
    DiagnosisResult,
    ModelInfo,
    ReadinessState,
    RunResult,
    ValidationResult,
    ValidationStatus,
)

logger = logging.getLogger(__name__)
console = Console()


def _determine_readiness(
    conversion: ConversionResult,
    benchmark: BenchmarkResult | None,
    validation: ValidationResult | None,
    diagnosis: DiagnosisResult,
    config: RunConfig,
) -> ReadinessState:
    if not conversion.success:
        return ReadinessState.NOT_READY

    if benchmark and not benchmark.success:
        return ReadinessState.NOT_READY

    if validation and validation.status == ValidationStatus.FAIL:
        return ReadinessState.NOT_READY

    # Check performance thresholds
    has_perf_issues = False
    if benchmark and benchmark.success:
        if benchmark.inference.mean_ms > config.latency_threshold_ms:
            has_perf_issues = True
        if 0 < benchmark.estimated_fps < config.fps_threshold:
            has_perf_issues = True
        if benchmark.peak_memory_mb and benchmark.peak_memory_mb > config.memory_threshold_mb:
            has_perf_issues = True

    has_warnings = False
    if validation and validation.status == ValidationStatus.WARNING:
        has_warnings = True
    if diagnosis.has_issues:
        has_warnings = True

    if has_perf_issues or has_warnings:
        return ReadinessState.PARTIAL

    return ReadinessState.READY


def run_pipeline(config: RunConfig) -> RunResult:
    from model2mobile.benchmark.runner import run_benchmark
    from model2mobile.convert.converter import convert_model
    from model2mobile.diagnose.analyzer import diagnose
    from model2mobile.report.html import save_html
    from model2mobile.report.json_report import save_json_reports
    from model2mobile.report.markdown import save_markdown
    from model2mobile.suggest.advisor import generate_suggestions
    from model2mobile.validate.validator import run_validation

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    output_dir = config.resolve_output_dir(run_id)

    console.print(Panel.fit(
        f"[bold]Model2Mobile[/bold] - Deployment Readiness Evaluation\n"
        f"Model: {config.model_path}\n"
        f"Task: {config.task} | Input: {config.input_size}x{config.input_size}\n"
        f"Run ID: {run_id}",
        border_style="blue",
    ))

    # --- Stage 1: Conversion ---
    console.print("\n[bold blue]Stage 1/4:[/bold blue] Converting to Core ML...")
    model_info, conversion = convert_model(config, output_dir)

    if conversion.success:
        console.print(
            f"  [green]Conversion succeeded[/green] "
            f"({conversion.conversion_time_s:.1f}s, {conversion.coreml_size_mb:.1f} MB)"
        )
    else:
        console.print(f"  [red]Conversion failed:[/red] {conversion.error_message}")

    # --- Stage 2: Benchmark ---
    benchmark: BenchmarkResult | None = None
    if conversion.success and config.benchmark_enabled:
        console.print("\n[bold blue]Stage 2/4:[/bold blue] Running benchmark...")
        benchmark = run_benchmark(conversion.coreml_path, config)  # type: ignore[arg-type]

        if benchmark.success:
            console.print(
                f"  [green]Benchmark complete[/green] "
                f"(inference: {benchmark.inference.mean_ms:.1f}ms, "
                f"FPS: {benchmark.estimated_fps:.1f})"
            )
        else:
            console.print(f"  [red]Benchmark failed:[/red] {benchmark.error_message}")
    elif not conversion.success:
        console.print("\n[dim]Stage 2/4: Benchmark skipped (conversion failed)[/dim]")
    else:
        console.print("\n[dim]Stage 2/4: Benchmark disabled[/dim]")

    # --- Stage 3: Validation ---
    validation: ValidationResult | None = None
    if conversion.success and config.validation_enabled:
        console.print("\n[bold blue]Stage 3/4:[/bold blue] Running validation...")
        validation = run_validation(config.model_path, conversion.coreml_path, config)  # type: ignore[arg-type]

        status_style = {
            ValidationStatus.PASS: "[green]PASS[/green]",
            ValidationStatus.WARNING: "[yellow]WARNING[/yellow]",
            ValidationStatus.FAIL: "[red]FAIL[/red]",
        }
        console.print(f"  Validation: {status_style.get(validation.status, validation.status.value)}")
    elif not conversion.success:
        console.print("\n[dim]Stage 3/4: Validation skipped (conversion failed)[/dim]")
    else:
        console.print("\n[dim]Stage 3/4: Validation disabled[/dim]")

    # --- Stage 4: Diagnosis, Suggestions & Report ---
    console.print("\n[bold blue]Stage 4/4:[/bold blue] Generating report...")
    diagnosis = diagnose(conversion, benchmark, validation)
    suggestions = generate_suggestions(diagnosis, benchmark, validation)

    readiness = _determine_readiness(conversion, benchmark, validation, diagnosis, config)

    result = RunResult(
        readiness=readiness,
        model_info=model_info,
        conversion=conversion,
        diagnosis=diagnosis,
        benchmark=benchmark,
        validation=validation,
        suggestions=suggestions,
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        output_dir=str(output_dir),
    )

    # --- Swift code generation ---
    swift_paths: list[Path] = []
    if conversion.success and config.codegen_enabled:
        from model2mobile.codegen.swift_generator import save_swift_code

        try:
            model_stem = Path(config.model_path).stem
            swift_paths = save_swift_code(
                coreml_path=conversion.coreml_path,  # type: ignore[arg-type]
                model_name=model_stem,
                output_dir=output_dir,
            )
            if swift_paths:
                console.print(
                    f"  [green]Generated {len(swift_paths)} Swift file(s)[/green]"
                )
        except Exception as exc:
            logger.warning("Swift code generation failed: %s", exc)
            console.print(f"  [yellow]Swift codegen skipped:[/yellow] {exc}")

    # Write all reports
    md_path = save_markdown(result, output_dir)
    json_paths = save_json_reports(result, output_dir)
    html_path = save_html(result, output_dir)

    # Save log
    log_path = output_dir / "run.log"
    log_path.write_text(
        f"Run ID: {run_id}\n"
        f"Model: {config.model_path}\n"
        f"Task: {config.task}\n"
        f"Input Size: {config.input_size}\n"
        f"Readiness: {readiness.value}\n"
        f"Output Dir: {output_dir}\n",
        encoding="utf-8",
    )

    # Print summary
    readiness_colors = {
        ReadinessState.READY: "green",
        ReadinessState.PARTIAL: "yellow",
        ReadinessState.NOT_READY: "red",
    }
    color = readiness_colors.get(readiness, "white")

    console.print(Panel.fit(
        f"[bold {color}]{readiness.value}[/bold {color}]\n\n"
        + (f"Inference: {benchmark.inference.mean_ms:.1f}ms | FPS: {benchmark.estimated_fps:.1f}\n"
           if benchmark and benchmark.success else "")
        + (f"Validation: {validation.status.value}\n" if validation else "")
        + (f"Issues: {len(diagnosis.diagnoses)}\n" if diagnosis.has_issues else "")
        + f"\nReport: {md_path}\n"
        f"HTML:   {html_path}\n"
        f"JSON:   {json_paths.get('summary', '')}"
        + ("".join(f"\nSwift:  {p}" for p in swift_paths) if swift_paths else ""),
        title="Result",
        border_style=color,
    ))

    return result
