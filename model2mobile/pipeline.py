"""Pipeline orchestration for the full evaluation run."""

from __future__ import annotations

import logging
import os
import uuid
import warnings
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
    OptimizationResult,
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
    # Suppress noisy warnings from coremltools and torch
    warnings.filterwarnings("ignore", message=".*coremltools.*")
    warnings.filterwarnings("ignore", message=".*has not been tested.*")
    warnings.filterwarnings("ignore", message=".*convert_to.*minimum_deployment_target.*")
    warnings.filterwarnings("ignore", message=".*has been renamed.*")
    warnings.filterwarnings("ignore", message=".*Tuple detected.*")
    logging.getLogger("coremltools").setLevel(logging.ERROR)

    # Suppress progress bars when not verbose
    if not config.verbose:
        os.environ["TQDM_DISABLE"] = "1"

    quiet = config.quiet

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

    if not quiet:
        console.print(Panel.fit(
            f"[bold]Model2Mobile[/bold] - Deployment Readiness Evaluation\n"
            f"Model: {config.model_path}\n"
            f"Task: {config.task} | Input: {config.input_size}x{config.input_size}\n"
            f"Run ID: {run_id}",
            border_style="blue",
        ))

    # Store the original task setting for display
    original_task = config.task

    # --- Stage 1: Conversion ---
    if not quiet:
        console.print("\n[bold blue]Stage 1/4:[/bold blue] Converting to Core ML...")
    model_info, conversion = convert_model(config, output_dir)

    # If task was "auto", convert_model will have resolved it.
    detected_task = config.task
    if original_task == "auto" and not quiet and detected_task != "auto":
        console.print(f"  [cyan]Detected task: {detected_task}[/cyan]")

    if not quiet:
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
        if not quiet:
            console.print("\n[bold blue]Stage 2/4:[/bold blue] Running benchmark...")
        benchmark = run_benchmark(conversion.coreml_path, config)  # type: ignore[arg-type]

        if not quiet:
            if benchmark.success:
                console.print(
                    f"  [green]Benchmark complete[/green] "
                    f"(inference: {benchmark.inference.mean_ms:.1f}ms, "
                    f"FPS: {benchmark.estimated_fps:.1f})"
                )
            else:
                console.print(f"  [red]Benchmark failed:[/red] {benchmark.error_message}")
    elif not quiet:
        if not conversion.success:
            console.print("\n[dim]Stage 2/4: Benchmark skipped (conversion failed)[/dim]")
        else:
            console.print("\n[dim]Stage 2/4: Benchmark disabled[/dim]")

    # --- Stage 3: Validation ---
    validation: ValidationResult | None = None
    if conversion.success and config.validation_enabled:
        if not quiet:
            console.print("\n[bold blue]Stage 3/4:[/bold blue] Running validation...")
        validation = run_validation(config.model_path, conversion.coreml_path, config)  # type: ignore[arg-type]

        if not quiet:
            status_style = {
                ValidationStatus.PASS: "[green]PASS[/green]",
                ValidationStatus.WARNING: "[yellow]WARNING[/yellow]",
                ValidationStatus.FAIL: "[red]FAIL[/red]",
            }
            console.print(f"  Validation: {status_style.get(validation.status, validation.status.value)}")
    elif not quiet:
        if not conversion.success:
            console.print("\n[dim]Stage 3/4: Validation skipped (conversion failed)[/dim]")
        else:
            console.print("\n[dim]Stage 3/4: Validation disabled[/dim]")

    # --- Optimization (optional) ---
    optimization: OptimizationResult | None = None
    if conversion.success and config.optimize_enabled:
        if not quiet:
            console.print("\n[bold blue]Optimization:[/bold blue] Searching for best quantization...")
        from model2mobile.optimize.optimizer import run_optimization

        optimization = run_optimization(conversion.coreml_path, config)  # type: ignore[arg-type]

        if not quiet:
            if optimization.recommended and optimization.recommended != "none":
                rec_variant = next(
                    (v for v in optimization.variants if v.name == optimization.recommended), None
                )
                size_info = (
                    f"{rec_variant.size_reduction_pct:.0f}% smaller"
                    if rec_variant and rec_variant.size_reduction_pct > 0
                    else ""
                )
                console.print(
                    f"  [green]Optimization: {optimization.recommended} recommended"
                    + (f" ({size_info})" if size_info else "")
                    + "[/green]"
                )
            else:
                console.print(f"  [yellow]Optimization: {optimization.recommendation_reason}[/yellow]")
    elif not quiet and config.optimize_enabled and not conversion.success:
        console.print("\n[dim]Optimization: skipped (conversion failed)[/dim]")

    # --- Stage 4: Diagnosis, Suggestions & Report ---
    if not quiet:
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
        optimization=optimization,
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
                task=config.task,
            )
            if not quiet and swift_paths:
                console.print(
                    f"  [green]Generated {len(swift_paths)} Swift file(s)[/green]"
                )
        except Exception as exc:
            logger.warning("Swift code generation failed: %s", exc)
            if not quiet:
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

    # Print result panel
    readiness_colors = {
        ReadinessState.READY: "green",
        ReadinessState.PARTIAL: "yellow",
        ReadinessState.NOT_READY: "red",
    }
    color = readiness_colors.get(readiness, "white")

    # Build result lines cleanly
    lines: list[str] = [f"[bold {color}]{readiness.value}[/bold {color}]"]
    lines.append("")  # blank separator

    if benchmark and benchmark.success:
        lines.append(
            f"Inference: {benchmark.inference.mean_ms:.1f}ms | "
            f"FPS: {benchmark.estimated_fps:.1f}"
        )
    if validation:
        lines.append(f"Validation: {validation.status.value}")
    if diagnosis.has_issues:
        lines.append(f"Issues: {len(diagnosis.diagnoses)}")
    if optimization and optimization.recommended and optimization.recommended != "none":
        rec_v = next((v for v in optimization.variants if v.name == optimization.recommended), None)
        opt_detail = optimization.recommended
        if rec_v and rec_v.size_reduction_pct > 0:
            opt_detail += f" ({rec_v.size_reduction_pct:.0f}% smaller)"
        lines.append(f"Optimization: {opt_detail}")

    lines.append("")  # blank separator before file paths
    lines.append(f"Report: {md_path}")
    lines.append(f"HTML:   {html_path}")
    summary_json = json_paths.get("summary", "")
    if summary_json:
        lines.append(f"JSON:   {summary_json}")
    for p in swift_paths:
        lines.append(f"Swift:  {p}")

    console.print(Panel.fit(
        "\n".join(lines),
        title="Result",
        border_style=color,
    ))

    return result
