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
@click.option("--warmup", default=5, type=int, help="Warmup iterations (default: 5)")
@click.option("--iterations", default=20, type=int, help="Measurement iterations (default: 20)")
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
    warmup: int,
    iterations: int,
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
    config.warmup_iterations = warmup
    config.measurement_iterations = iterations

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
@click.option("--verbose", "-v", is_flag=True)
def benchmark(coreml: str, input_size: int, compute_unit: str, warmup: int, iterations: int, compare_units: bool, verbose: bool) -> None:
    """Run benchmark on a Core ML model."""
    _setup_logging(verbose)

    from model2mobile.benchmark.runner import run_benchmark

    config = RunConfig(
        input_size=input_size,
        compute_unit=compute_unit,
        warmup_iterations=warmup,
        measurement_iterations=iterations,
        compare_compute_units=compare_units,
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
