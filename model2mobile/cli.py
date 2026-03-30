"""CLI entry point for Model2Mobile."""

from __future__ import annotations

import logging
import sys
import tempfile
import urllib.request
from pathlib import Path

import click
from rich.console import Console
from rich.prompt import Confirm, Prompt

from model2mobile import __version__
from model2mobile.config import RunConfig

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _resolve_model(model: str) -> str:
    """If model is a URL, download it and return the local path."""
    if model.startswith("http://") or model.startswith("https://"):
        url = model
        filename = Path(url.split("?")[0].split("/")[-1]).name or "model.pt"
        dest = Path(tempfile.gettempdir()) / "model2mobile" / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            console.print(f"  [cyan]Downloading {url}...[/cyan]")
            urllib.request.urlretrieve(url, str(dest))
            console.print(f"  [green]Saved to {dest}[/green]")
        else:
            console.print(f"  [dim]Using cached {dest}[/dim]")
        return str(dest)

    if not Path(model).exists():
        console.print(f"[red]File not found: {model}[/red]")
        sys.exit(1)
    return model


@click.group()
@click.version_option(version=__version__, prog_name="model2mobile")
def main() -> None:
    """Model2Mobile - Evaluate whether your PyTorch model is ready for on-device deployment."""


# ---------------------------------------------------------------------------
# Primary mode: run
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", "-m", required=True, help="Path or URL to PyTorch model (.pt / .pth / .torchscript)")
@click.option("--task", "-t", default="detect", type=click.Choice(["detect"]), help="Model task (default: detect)")
@click.option("--input-size", "-s", default=None, type=int, help="Input image size (auto-detected if omitted)")
@click.option("--output-dir", "-o", default="outputs", help="Output directory (default: outputs)")
@click.option("--compute-unit", default="ALL", type=click.Choice(["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"]), help="Compute unit for conversion")
@click.option("--no-benchmark", is_flag=True, help="Skip benchmark")
@click.option("--no-validation", is_flag=True, help="Skip validation")
@click.option("--compare-units", is_flag=True, help="Benchmark across all compute units")
@click.option("--no-codegen", is_flag=True, help="Skip Swift code generation")
@click.option("--warmup", default=5, type=int, help="Warmup iterations (default: 5)")
@click.option("--iterations", default=20, type=int, help="Measurement iterations (default: 20)")
@click.option("--device", "-d", default="local", type=click.Choice(["local", "iphone"]), help="Benchmark device (local Mac or connected iPhone)")
@click.option("--config", "config_file", type=click.Path(exists=True), default=None, help="YAML config file")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def run(
    model: str,
    task: str,
    input_size: int | None,
    output_dir: str,
    compute_unit: str,
    no_benchmark: bool,
    no_validation: bool,
    compare_units: bool,
    no_codegen: bool,
    warmup: int,
    iterations: int,
    device: str,
    config_file: str | None,
    verbose: bool,
) -> None:
    """Run the full deployment readiness evaluation."""
    _setup_logging(verbose)
    model = _resolve_model(model)

    if config_file:
        config = RunConfig.from_yaml(config_file)
        config.model_path = model
    else:
        config = RunConfig()
        config.model_path = model

    config.task = task
    if input_size is not None:
        config.input_size = input_size
        config.input_size_auto = False
    # else: keep defaults (640 + auto=True)
    config.output_dir = output_dir
    config.compute_unit = compute_unit
    config.benchmark_enabled = not no_benchmark
    config.validation_enabled = not no_validation
    config.compare_compute_units = compare_units
    config.codegen_enabled = not no_codegen
    config.warmup_iterations = warmup
    config.measurement_iterations = iterations
    config.device = device

    from model2mobile.pipeline import run_pipeline

    run_pipeline(config)


# ---------------------------------------------------------------------------
# Guided mode: init
# ---------------------------------------------------------------------------


@main.command()
def init() -> None:
    """Interactive guided setup for a readiness evaluation."""
    console.print("\n[bold]Model2Mobile Interactive Setup[/bold]\n")

    model_path = Prompt.ask("Model path or URL", default="./model.pt")
    model_path = _resolve_model(model_path)

    task = Prompt.ask("Task", choices=["detect"], default="detect")
    size_str = Prompt.ask("Input size (leave empty to auto-detect)", default="")
    input_size_auto = True
    input_size = 640
    if size_str.strip():
        input_size = int(size_str)
        input_size_auto = False

    compute_unit = Prompt.ask(
        "Compute unit",
        choices=["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"],
        default="ALL",
    )
    do_benchmark = Confirm.ask("Enable benchmark?", default=True)
    do_validation = Confirm.ask("Enable validation?", default=True)
    output_dir = Prompt.ask("Output directory", default="outputs")

    config = RunConfig(
        model_path=model_path,
        task=task,
        input_size=input_size,
        input_size_auto=input_size_auto,
        output_dir=output_dir,
        compute_unit=compute_unit,
        benchmark_enabled=do_benchmark,
        validation_enabled=do_validation,
    )

    console.print("\n[bold blue]Starting evaluation...[/bold blue]\n")

    from model2mobile.pipeline import run_pipeline

    run_pipeline(config)


# ---------------------------------------------------------------------------
# Expert mode: individual stage commands
# ---------------------------------------------------------------------------


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True))
@click.option("--input-size", "-s", default=640, type=int)
@click.option("--compute-unit", default="ALL", type=click.Choice(["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"]))
@click.option("--output-dir", "-o", default="outputs")
@click.option("--verbose", "-v", is_flag=True)
def convert(model: str, input_size: int, compute_unit: str, output_dir: str, verbose: bool) -> None:
    """Convert a PyTorch model to Core ML."""
    _setup_logging(verbose)

    from model2mobile.convert.converter import convert_model

    config = RunConfig(model_path=model, input_size=input_size, compute_unit=compute_unit, output_dir=output_dir)
    out_dir = config.resolve_output_dir("convert")
    model_info, result = convert_model(config, out_dir)

    if result.success:
        console.print(f"[green]Conversion succeeded[/green]: {result.coreml_path}")
    else:
        console.print(f"[red]Conversion failed[/red]: {result.error_message}")
        sys.exit(1)


@main.command()
@click.option("--coreml", "-c", required=True, type=click.Path(exists=True), help="Path to .mlpackage")
@click.option("--input-size", "-s", default=640, type=int)
@click.option("--compute-unit", default="ALL", type=click.Choice(["ALL", "CPU_ONLY", "CPU_AND_GPU", "CPU_AND_NE"]))
@click.option("--warmup", default=5, type=int)
@click.option("--iterations", default=20, type=int)
@click.option("--compare-units", is_flag=True)
@click.option("--device", "-d", default="local", type=click.Choice(["local", "iphone"]), help="Benchmark device (local Mac or connected iPhone)")
@click.option("--verbose", "-v", is_flag=True)
def benchmark(coreml: str, input_size: int, compute_unit: str, warmup: int, iterations: int, compare_units: bool, device: str, verbose: bool) -> None:
    """Run benchmark on a Core ML model."""
    _setup_logging(verbose)

    from model2mobile.benchmark.runner import run_benchmark

    config = RunConfig(
        input_size=input_size,
        compute_unit=compute_unit,
        warmup_iterations=warmup,
        measurement_iterations=iterations,
        compare_compute_units=compare_units,
        device=device,
    )
    result = run_benchmark(coreml, config)

    if result.success:
        console.print(f"[green]Benchmark complete[/green]")
        console.print(f"  Inference: {result.inference.mean_ms:.2f}ms (median: {result.inference.median_ms:.2f}ms)")
        console.print(f"  End-to-end: {result.end_to_end.mean_ms:.2f}ms")
        console.print(f"  FPS: {result.estimated_fps:.1f}")
        if result.peak_memory_mb:
            console.print(f"  Peak memory: {result.peak_memory_mb:.1f} MB")
    else:
        console.print(f"[red]Benchmark failed[/red]: {result.error_message}")
        sys.exit(1)


@main.command()
@click.option("--model", "-m", required=True, type=click.Path(exists=True), help="Path to PyTorch model")
@click.option("--coreml", "-c", required=True, type=click.Path(exists=True), help="Path to .mlpackage")
@click.option("--input-size", "-s", default=640, type=int)
@click.option("--verbose", "-v", is_flag=True)
def validate(model: str, coreml: str, input_size: int, verbose: bool) -> None:
    """Validate Core ML output against PyTorch reference."""
    _setup_logging(verbose)

    from model2mobile.validate.validator import run_validation

    config = RunConfig(model_path=model, input_size=input_size)
    result = run_validation(model, coreml, config)

    status_colors = {
        "PASS": "green",
        "WARNING": "yellow",
        "FAIL": "red",
    }
    color = status_colors.get(result.status.value, "white")
    console.print(f"Validation: [{color}]{result.status.value}[/{color}]")

    for check in result.checks:
        c = status_colors.get(check.status.value, "white")
        console.print(f"  [{c}]{check.status.value}[/{c}] {check.name}: {check.detail}")

    if result.status == "FAIL":
        sys.exit(1)


@main.command()
@click.option("--model", "-m", required=True, help="Path to .pt model or .mlpackage")
@click.option("--input-size", "-s", default=None, type=int, help="Input image size (auto-detected if omitted)")
@click.option("--target-fps", default=None, type=float, help="Target FPS to optimize for")
@click.option("--output-dir", "-o", default="outputs", help="Output directory (default: outputs)")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
def optimize(
    model: str,
    input_size: int | None,
    target_fps: float | None,
    output_dir: str,
    verbose: bool,
) -> None:
    """Find the optimal quantization for deployment."""
    _setup_logging(verbose)
    model = _resolve_model(model)

    from rich.panel import Panel
    from rich.table import Table

    from model2mobile.optimize.optimizer import run_optimization
    from model2mobile.report.optimization import save_optimization_report

    config = RunConfig(model_path=model, output_dir=output_dir)
    if input_size is not None:
        config.input_size = input_size
        config.input_size_auto = False

    model_path = Path(model)
    coreml_path: str

    # If input is a .pt/.pth file, convert to Core ML first
    if model_path.suffix.lower() in (".pt", ".pth", ".torchscript", ".ts"):
        console.print("[bold blue]Converting PyTorch model to Core ML...[/bold blue]")
        from model2mobile.convert.converter import convert_model

        out_dir = config.resolve_output_dir("optimize_convert")
        _, conv_result = convert_model(config, out_dir)
        if not conv_result.success:
            console.print(f"[red]Conversion failed:[/red] {conv_result.error_message}")
            sys.exit(1)
        coreml_path = conv_result.coreml_path  # type: ignore[assignment]
        console.print(f"[green]Converted:[/green] {coreml_path}")
    elif model_path.suffix.lower() == ".mlpackage" or model_path.is_dir():
        coreml_path = str(model_path)
    else:
        console.print(f"[red]Unsupported file type: {model_path.suffix}[/red]")
        sys.exit(1)

    console.print(
        Panel.fit(
            "[bold]Optimization Search[/bold]\n"
            f"Model: {coreml_path}\n"
            f"Input: {config.input_size}x{config.input_size}"
            + (f"\nTarget FPS: {target_fps}" if target_fps else ""),
            border_style="blue",
        )
    )

    result = run_optimization(coreml_path, config, target_fps=target_fps)

    # Print comparison table
    table = Table(
        title="Optimization Results",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Variant", style="bold")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Size Reduction", justify="right")
    table.add_column("Inference (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("FPS", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Status", justify="center")

    # Add original baseline row
    orig_fps = (
        1000.0 / result.original_inference_ms
        if result.original_inference_ms > 0
        else 0.0
    )
    table.add_row(
        "[dim]original[/dim]",
        f"{result.original_size_mb:.1f}",
        "-",
        f"{result.original_inference_ms:.2f}",
        "-",
        f"{orig_fps:.1f}",
        "-",
        "[dim]baseline[/dim]",
    )

    for v in result.variants:
        is_rec = v.name == result.recommended
        name_style = f"[bold green]{v.name}[/bold green]" if is_rec else v.name

        if v.error:
            table.add_row(
                name_style, "-", "-", "-", "-", "-", "-",
                f"[red]FAILED[/red]",
            )
        else:
            size_red_style = (
                f"[green]{v.size_reduction_pct:+.1f}%[/green]"
                if v.size_reduction_pct > 0
                else f"{v.size_reduction_pct:+.1f}%"
            )
            speedup_style = (
                f"[green]{v.speedup_pct:+.1f}%[/green]"
                if v.speedup_pct > 0
                else f"[yellow]{v.speedup_pct:+.1f}%[/yellow]"
                if v.speedup_pct < -5
                else f"{v.speedup_pct:+.1f}%"
            )
            status = "[bold green]RECOMMENDED[/bold green]" if is_rec else "[green]OK[/green]"

            table.add_row(
                name_style,
                f"{v.model_size_mb:.1f}",
                size_red_style,
                f"{v.inference_mean_ms:.2f}",
                f"{v.inference_p95_ms:.2f}",
                f"{v.estimated_fps:.1f}",
                speedup_style,
                status,
            )

    console.print()
    console.print(table)

    # Recommendation panel
    if result.recommended and result.recommended != "none":
        console.print(
            Panel.fit(
                f"[bold green]Recommended: {result.recommended}[/bold green]\n"
                f"{result.recommendation_reason}",
                title="Recommendation",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"[yellow]{result.recommendation_reason}[/yellow]",
                title="Recommendation",
                border_style="yellow",
            )
        )

    # Save results
    report_dir = Path(output_dir) / "optimize"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = save_optimization_report(result, report_dir)
    json_path = report_dir / "optimization_result.json"
    result.save_json(json_path)

    console.print(f"\n[dim]Report: {report_path}[/dim]")
    console.print(f"[dim]JSON:   {json_path}[/dim]")


@main.command()
@click.option("--run-dir", "-d", required=True, type=click.Path(exists=True), help="Path to run output directory")
@click.option("--format", "fmt", default="all", type=click.Choice(["markdown", "html", "json", "all"]))
def report(run_dir: str, fmt: str) -> None:
    """Regenerate reports from an existing run directory."""
    import json

    from model2mobile.models import RunResult
    from model2mobile.report.html import save_html
    from model2mobile.report.json_report import save_json_reports
    from model2mobile.report.markdown import save_markdown

    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.exists():
        console.print(f"[red]No summary.json found in {run_dir}[/red]")
        sys.exit(1)

    console.print(f"Report regeneration from existing runs is available after initial run.")
    console.print(f"Use 'model2mobile run' to generate initial reports.")
