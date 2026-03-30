"""HTML report generator for Model2Mobile pipeline results."""

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

_CSS = """\
:root {
    --color-bg: #f4f5f7;
    --color-surface: #ffffff;
    --color-header: #1e293b;
    --color-header-text: #f8fafc;
    --color-text: #1e293b;
    --color-text-muted: #64748b;
    --color-border: #e2e8f0;
    --color-green: #22c55e;
    --color-green-bg: #f0fdf4;
    --color-yellow: #eab308;
    --color-yellow-bg: #fefce8;
    --color-red: #ef4444;
    --color-red-bg: #fef2f2;
    --color-blue: #3b82f6;
    --color-blue-bg: #eff6ff;
    --color-bar-pre: #60a5fa;
    --color-bar-inf: #3b82f6;
    --color-bar-post: #818cf8;
    --color-bar-e2e: #1e293b;
    --radius: 8px;
    --shadow: 0 1px 3px rgba(0,0,0,0.08), 0 1px 2px rgba(0,0,0,0.06);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--color-bg);
    color: var(--color-text);
    line-height: 1.6;
}
.header {
    background: var(--color-header);
    color: var(--color-header-text);
    padding: 32px 24px;
    text-align: center;
}
.header h1 { font-size: 24px; font-weight: 700; margin-bottom: 12px; }
.badge {
    display: inline-block;
    padding: 6px 20px;
    border-radius: 20px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-ready { background: var(--color-green); color: #fff; }
.badge-partial { background: var(--color-yellow); color: #1e293b; }
.badge-not-ready { background: var(--color-red); color: #fff; }
.container {
    max-width: 960px;
    margin: 0 auto;
    padding: 24px 16px 48px;
}
.card {
    background: var(--color-surface);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    border: 1px solid var(--color-border);
    margin-bottom: 20px;
    overflow: hidden;
}
.card-title {
    font-size: 16px;
    font-weight: 600;
    padding: 16px 20px;
    border-bottom: 1px solid var(--color-border);
    background: #f8fafc;
}
.card-body { padding: 20px; }
table { width: 100%; border-collapse: collapse; }
th, td {
    text-align: left;
    padding: 8px 12px;
    border-bottom: 1px solid var(--color-border);
    font-size: 14px;
}
th {
    font-weight: 600;
    color: var(--color-text-muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
tr:last-child td { border-bottom: none; }
code, .mono {
    font-family: "SF Mono", "Fira Code", "Cascadia Code", Consolas, monospace;
    font-size: 13px;
}
.status-pass { color: var(--color-green); font-weight: 600; }
.status-warning { color: var(--color-yellow); font-weight: 600; }
.status-fail { color: var(--color-red); font-weight: 600; }
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 12px;
}
.summary-item {
    padding: 12px 16px;
    border-radius: 6px;
    border: 1px solid var(--color-border);
}
.summary-item .label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--color-text-muted);
    margin-bottom: 4px;
}
.summary-item .value { font-size: 18px; font-weight: 700; }
.bar-chart { padding: 4px 0; }
.bar-row {
    display: flex;
    align-items: center;
    margin-bottom: 10px;
}
.bar-label {
    width: 110px;
    font-size: 13px;
    font-weight: 500;
    flex-shrink: 0;
}
.bar-track {
    flex: 1;
    height: 24px;
    background: var(--color-bg);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
}
.bar-fill {
    height: 100%;
    border-radius: 4px;
    min-width: 2px;
    transition: width 0.3s ease;
}
.bar-value {
    width: 80px;
    text-align: right;
    font-size: 13px;
    font-weight: 600;
    flex-shrink: 0;
    padding-left: 8px;
}
.bar-fill.preprocess { background: var(--color-bar-pre); }
.bar-fill.inference { background: var(--color-bar-inf); }
.bar-fill.postprocess { background: var(--color-bar-post); }
.bar-fill.e2e { background: var(--color-bar-e2e); }
.suggestion-item {
    padding: 12px 16px;
    border-left: 3px solid var(--color-blue);
    background: var(--color-blue-bg);
    border-radius: 0 6px 6px 0;
    margin-bottom: 10px;
}
.suggestion-item .s-title { font-weight: 600; font-size: 14px; }
.suggestion-item .s-desc { font-size: 13px; color: var(--color-text-muted); margin-top: 2px; }
.suggestion-item .s-priority {
    display: inline-block;
    background: var(--color-blue);
    color: #fff;
    font-size: 11px;
    font-weight: 600;
    padding: 1px 8px;
    border-radius: 10px;
    margin-right: 6px;
}
.diagnosis-item {
    padding: 12px 16px;
    border-left: 3px solid var(--color-red);
    background: var(--color-red-bg);
    border-radius: 0 6px 6px 0;
    margin-bottom: 10px;
}
.diagnosis-item .d-cat { font-weight: 600; font-size: 14px; }
.diagnosis-item .d-cause { font-size: 13px; color: var(--color-text-muted); margin-top: 2px; }
.note {
    background: var(--color-yellow-bg);
    border: 1px solid var(--color-yellow);
    border-radius: 6px;
    padding: 12px 16px;
    font-size: 13px;
    color: var(--color-text-muted);
    margin-top: 20px;
}
"""


def _esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _badge_class(state: ReadinessState) -> str:
    return {
        ReadinessState.READY: "badge-ready",
        ReadinessState.PARTIAL: "badge-partial",
        ReadinessState.NOT_READY: "badge-not-ready",
    }.get(state, "badge-not-ready")


def _status_class(status: ValidationStatus) -> str:
    return {
        ValidationStatus.PASS: "status-pass",
        ValidationStatus.WARNING: "status-warning",
        ValidationStatus.FAIL: "status-fail",
    }.get(status, "")


def _primary_bottleneck(result: RunResult) -> str:
    if result.diagnosis.has_issues:
        return result.diagnosis.primary_category.value
    if result.benchmark and not result.benchmark.success:
        return "runtime_failure"
    if not result.conversion.success:
        return "conversion_failure"
    return "none"


def _card(title: str, body: str) -> str:
    return (
        f'<div class="card">'
        f'<div class="card-title">{_esc(title)}</div>'
        f'<div class="card-body">{body}</div>'
        f"</div>"
    )


def _kv_row(key: str, value: str) -> str:
    return f"<tr><td>{_esc(key)}</td><td class='mono'>{value}</td></tr>"


def _section_summary(result: RunResult) -> str:
    conv_ok = result.conversion.success
    rt_ok = result.benchmark.success if result.benchmark else None
    val_status = result.validation.status.value if result.validation else "N/A"

    items = [
        ("Conversion", "Pass" if conv_ok else "Fail",
         "color:var(--color-green)" if conv_ok else "color:var(--color-red)"),
        ("Runtime", "Pass" if rt_ok else ("Fail" if rt_ok is not None else "N/A"),
         ("color:var(--color-green)" if rt_ok else "color:var(--color-red)")
         if rt_ok is not None else "color:var(--color-text-muted)"),
        ("Validation", val_status,
         f"color:var(--color-green)" if val_status == "PASS"
         else f"color:var(--color-yellow)" if val_status == "WARNING"
         else f"color:var(--color-red)" if val_status == "FAIL"
         else "color:var(--color-text-muted)"),
        ("Bottleneck", _primary_bottleneck(result), ""),
    ]
    if result.benchmark and result.benchmark.estimated_fps > 0:
        items.append(("Est. FPS", f"{result.benchmark.estimated_fps:.1f}", ""))
    if result.benchmark and result.benchmark.peak_memory_mb is not None:
        items.append(("Peak Memory", f"{result.benchmark.peak_memory_mb:.1f} MB", ""))

    html_items = []
    for label, value, style in items:
        style_attr = f' style="{style}"' if style else ""
        html_items.append(
            f'<div class="summary-item">'
            f'<div class="label">{_esc(label)}</div>'
            f'<div class="value"{style_attr}>{_esc(value)}</div>'
            f"</div>"
        )
    body = f'<div class="summary-grid">{"".join(html_items)}</div>'
    return _card("Summary", body)


def _section_model_info(result: RunResult) -> str:
    info = result.model_info
    rows = [
        _kv_row("Path", f"<code>{_esc(info.path)}</code>"),
        _kv_row("Architecture", _esc(info.architecture)),
        _kv_row("Parameters", f"{info.parameter_count:,}"),
        _kv_row("Input Shape", _esc(str(info.input_shape))),
        _kv_row("Estimated Size", f"{info.estimated_size_mb:.1f} MB"),
        _kv_row("Dynamic Shapes", str(info.has_dynamic_shapes)),
    ]
    body = f"<table>{''.join(rows)}</table>"
    return _card("Model Info", body)


def _section_conversion(result: RunResult) -> str:
    conv = result.conversion
    rows = [
        _kv_row("Success", str(conv.success)),
        _kv_row("Compute Unit", _esc(conv.compute_unit)),
        _kv_row("Conversion Time", f"{conv.conversion_time_s:.2f}s"),
        _kv_row("CoreML Size", f"{conv.coreml_size_mb:.1f} MB"),
    ]
    if conv.coreml_path:
        rows.append(_kv_row("CoreML Path", f"<code>{_esc(conv.coreml_path)}</code>"))
    body = f"<table>{''.join(rows)}</table>"
    if conv.error_message:
        body += f"<p style='color:var(--color-red);margin-top:12px'><strong>Error:</strong> {_esc(conv.error_message)}</p>"
    if conv.warnings:
        body += "<ul style='margin-top:12px;color:var(--color-text-muted);font-size:13px'>"
        for w in conv.warnings:
            body += f"<li>{_esc(w)}</li>"
        body += "</ul>"
    return _card("Conversion Details", body)


def _section_benchmark(benchmark: BenchmarkResult) -> str:
    rows = [
        _kv_row("Device", _esc(benchmark.device_name)),
        _kv_row("Compute Unit", _esc(benchmark.compute_unit)),
        _kv_row("Warmup Iterations", str(benchmark.warmup_iterations)),
        _kv_row("Measurement Iterations", str(benchmark.measurement_iterations)),
        _kv_row("Estimated FPS", f"{benchmark.estimated_fps:.1f}"),
    ]
    if benchmark.peak_memory_mb is not None:
        rows.append(_kv_row("Peak Memory", f"{benchmark.peak_memory_mb:.1f} MB"))
    body = f"<table>{''.join(rows)}</table>"

    # Bar chart
    stages = [
        ("Preprocess", benchmark.preprocess, "preprocess"),
        ("Inference", benchmark.inference, "inference"),
        ("Postprocess", benchmark.postprocess, "postprocess"),
        ("End-to-End", benchmark.end_to_end, "e2e"),
    ]
    active_stages = [(n, s, c) for n, s, c in stages if s.samples > 0]

    if active_stages:
        max_val = max(s.mean_ms for _, s, _ in active_stages) if active_stages else 1.0
        max_val = max(max_val, 0.01)  # prevent division by zero

        body += '<div class="bar-chart" style="margin-top:16px">'
        for name, stats, css_class in active_stages:
            pct = (stats.mean_ms / max_val) * 100
            body += (
                f'<div class="bar-row">'
                f'<div class="bar-label">{_esc(name)}</div>'
                f'<div class="bar-track">'
                f'<div class="bar-fill {css_class}" style="width:{pct:.1f}%"></div>'
                f"</div>"
                f'<div class="bar-value">{stats.mean_ms:.2f} ms</div>'
                f"</div>"
            )
        body += "</div>"

        # Detailed table
        body += (
            '<table style="margin-top:16px">'
            "<tr><th>Stage</th><th>Mean</th><th>Median</th>"
            "<th>P95</th><th>Min</th><th>Max</th><th>Std</th></tr>"
        )
        for name, stats, _ in active_stages:
            body += (
                f"<tr><td>{_esc(name)}</td>"
                f"<td class='mono'>{stats.mean_ms:.2f}</td>"
                f"<td class='mono'>{stats.median_ms:.2f}</td>"
                f"<td class='mono'>{stats.p95_ms:.2f}</td>"
                f"<td class='mono'>{stats.min_ms:.2f}</td>"
                f"<td class='mono'>{stats.max_ms:.2f}</td>"
                f"<td class='mono'>{stats.std_ms:.2f}</td></tr>"
            )
        body += "</table>"

    if benchmark.error_message:
        body += f"<p style='color:var(--color-red);margin-top:12px'><strong>Error:</strong> {_esc(benchmark.error_message)}</p>"

    return _card("Benchmark Breakdown", body)


def _section_validation(validation: ValidationResult) -> str:
    body = (
        f"<p style='margin-bottom:12px'><strong>Overall:</strong> "
        f"<span class='{_status_class(validation.status)}'>{validation.status.value}</span> "
        f"&mdash; {validation.pass_count} pass, "
        f"{validation.warning_count} warning, "
        f"{validation.fail_count} fail</p>"
    )
    if validation.checks:
        body += "<table><tr><th>Check</th><th>Status</th><th>Detail</th></tr>"
        for c in validation.checks:
            sc = _status_class(c.status)
            body += (
                f"<tr><td>{_esc(c.name)}</td>"
                f"<td class='{sc}'>{c.status.value}</td>"
                f"<td style='font-size:13px'>{_esc(c.detail)}</td></tr>"
            )
        body += "</table>"
    if validation.error_message:
        body += f"<p style='color:var(--color-red);margin-top:12px'><strong>Error:</strong> {_esc(validation.error_message)}</p>"
    return _card("Validation Details", body)


def _section_diagnosis(result: RunResult) -> str:
    diag = result.diagnosis
    if not diag.has_issues:
        body = "<p style='color:var(--color-green)'>No issues detected.</p>"
        return _card("Diagnosis", body)

    body = f"<p style='margin-bottom:12px'><strong>Primary Category:</strong> <code>{_esc(diag.primary_category.value)}</code></p>"
    for d in diag.diagnoses:
        steps_html = ""
        if d.suggested_steps:
            steps_html = "<ul style='margin-top:4px;margin-bottom:0;font-size:13px'>"
            for step in d.suggested_steps:
                steps_html += f"<li>{_esc(step)}</li>"
            steps_html += "</ul>"
        body += (
            f'<div class="diagnosis-item">'
            f'<div class="d-cat">{_esc(d.category.value)}</div>'
            f'<div class="d-cause">{_esc(d.likely_cause)}</div>'
            f"{steps_html}"
            f"</div>"
        )
    return _card("Diagnosis", body)


def _section_suggestions(result: RunResult) -> str:
    if not result.suggestions:
        body = "<p style='color:var(--color-text-muted)'>No suggestions.</p>"
        return _card("Suggestions", body)

    body = ""
    for s in sorted(result.suggestions, key=lambda x: x.priority):
        body += (
            f'<div class="suggestion-item">'
            f'<span class="s-priority">P{s.priority}</span>'
            f'<span class="s-title">{_esc(s.title)}</span>'
            f'<div class="s-desc">{_esc(s.description)}</div>'
            f"</div>"
        )
    return _card("Suggestions", body)


def _section_metadata(result: RunResult) -> str:
    rows = [
        _kv_row("Run ID", f"<code>{_esc(result.run_id)}</code>"),
        _kv_row("Timestamp", _esc(result.timestamp)),
        _kv_row("Output Dir", f"<code>{_esc(result.output_dir)}</code>"),
        _kv_row("Input Shape", _esc(str(result.model_info.input_shape))),
    ]
    if result.benchmark:
        rows.append(_kv_row("Device", _esc(result.benchmark.device_name)))
        rows.append(_kv_row("Compute Unit", _esc(result.benchmark.compute_unit)))
    body = f"<table>{''.join(rows)}</table>"
    return _card("Run Metadata", body)


def generate_html(result: RunResult) -> str:
    badge_cls = _badge_class(result.readiness)
    readiness_label = result.readiness.value.replace("_", " ")

    sections = [
        _section_summary(result),
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

    note = (
        '<div class="note">'
        "<strong>Note:</strong> These results are specific to the evaluated scenario "
        "(model, input size, device, compute unit). "
        "Different configurations may yield different results."
        "</div>"
    )

    body_content = "\n".join(sections) + note

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Model2Mobile Report</title>\n"
        f"<style>\n{_CSS}</style>\n"
        "</head>\n"
        "<body>\n"
        '<div class="header">\n'
        "<h1>Model2Mobile Report</h1>\n"
        f'<span class="badge {badge_cls}">{_esc(readiness_label)}</span>\n'
        "</div>\n"
        f'<div class="container">\n{body_content}\n</div>\n'
        "</body>\n"
        "</html>\n"
    )


def save_html(result: RunResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.html"
    path.write_text(generate_html(result), encoding="utf-8")
    return path
