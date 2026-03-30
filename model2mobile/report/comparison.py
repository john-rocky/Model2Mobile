"""HTML comparison report generator for Model2Mobile."""

from __future__ import annotations

from model2mobile.report.html import _CSS, _esc


def _badge_html(readiness: str) -> str:
    """Return a badge span for a readiness state string."""
    mapping = {
        "READY": "badge-ready",
        "PARTIAL": "badge-partial",
        "NOT_READY": "badge-not-ready",
    }
    cls = mapping.get(readiness, "badge-not-ready")
    label = readiness.replace("_", " ")
    return f'<span class="badge {cls}">{_esc(label)}</span>'


def _safe_get(data: dict, *keys: str, default: object = None) -> object:
    """Safely traverse nested dicts."""
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k, default)
        else:
            return default
    return current


def _fmt_float(val: object, fmt: str = ".2f", suffix: str = "") -> str:
    if val is None:
        return "N/A"
    try:
        return f"{float(val):{fmt}}{suffix}"
    except (TypeError, ValueError):
        return str(val)


def _delta_cell(val_a: object, val_b: object, fmt: str = ".2f",
                suffix: str = "", lower_is_better: bool = True) -> str:
    """Return an HTML table cell showing the delta between two values.

    Green = improved, red = worse.
    """
    if val_a is None or val_b is None:
        return '<td class="mono" style="color:var(--color-text-muted)">--</td>'
    try:
        a = float(val_a)
        b = float(val_b)
    except (TypeError, ValueError):
        return '<td class="mono" style="color:var(--color-text-muted)">--</td>'

    delta = b - a
    if abs(delta) < 1e-9:
        return f'<td class="mono" style="color:var(--color-text-muted)">0{suffix}</td>'

    if lower_is_better:
        color = "var(--color-green)" if delta < 0 else "var(--color-red)"
    else:
        color = "var(--color-green)" if delta > 0 else "var(--color-red)"

    sign = "+" if delta > 0 else ""
    return f'<td class="mono" style="color:{color};font-weight:600">{sign}{delta:{fmt}}{suffix}</td>'


def _metric_row(label: str, val_a: object, val_b: object,
                fmt: str = ".2f", suffix: str = "",
                lower_is_better: bool = True) -> str:
    """Build a comparison table row with delta."""
    return (
        f"<tr>"
        f"<td>{_esc(label)}</td>"
        f'<td class="mono">{_fmt_float(val_a, fmt, suffix)}</td>'
        f'<td class="mono">{_fmt_float(val_b, fmt, suffix)}</td>'
        f"{_delta_cell(val_a, val_b, fmt, suffix, lower_is_better)}"
        f"</tr>"
    )


def _str_row(label: str, val_a: object, val_b: object) -> str:
    """Build a comparison row for string/status values (no delta)."""
    sa = str(val_a) if val_a is not None else "N/A"
    sb = str(val_b) if val_b is not None else "N/A"
    diff_style = ""
    if sa != sb:
        diff_style = ' style="color:var(--color-yellow);font-weight:600"'
    return (
        f"<tr>"
        f"<td>{_esc(label)}</td>"
        f'<td class="mono">{_esc(sa)}</td>'
        f'<td class="mono">{_esc(sb)}</td>'
        f"<td{diff_style}>{_esc('--' if sa == sb else 'differs')}</td>"
        f"</tr>"
    )


def _card(title: str, body: str) -> str:
    return (
        f'<div class="card">'
        f'<div class="card-title">{_esc(title)}</div>'
        f'<div class="card-body">{body}</div>'
        f"</div>"
    )


def generate_comparison_html(
    data_a: dict, data_b: dict, label_a: str, label_b: str
) -> str:
    """Generate a self-contained HTML comparison report for two runs.

    Parameters
    ----------
    data_a, data_b : dict
        Parsed summary.json contents for each run.
    label_a, label_b : str
        Human-readable labels (e.g. directory names) for each run.
    """
    readiness_a = data_a.get("readiness", "UNKNOWN")
    readiness_b = data_b.get("readiness", "UNKNOWN")

    # ------ Header with badges ------
    header_html = (
        '<div class="header">\n'
        "<h1>Model2Mobile Comparison Report</h1>\n"
        '<div style="display:flex;justify-content:center;gap:40px;margin-top:12px">\n'
        f'<div style="text-align:center"><div style="font-size:13px;opacity:0.7;margin-bottom:4px">{_esc(label_a)}</div>{_badge_html(readiness_a)}</div>\n'
        f'<div style="font-size:24px;align-self:center;opacity:0.4">vs</div>\n'
        f'<div style="text-align:center"><div style="font-size:13px;opacity:0.7;margin-bottom:4px">{_esc(label_b)}</div>{_badge_html(readiness_b)}</div>\n'
        "</div>\n"
        "</div>\n"
    )

    sections: list[str] = []

    # ------ Key Metrics Comparison Table ------
    table_header = (
        "<table>"
        "<tr><th>Metric</th>"
        f"<th>{_esc(label_a)}</th>"
        f"<th>{_esc(label_b)}</th>"
        "<th>Delta</th></tr>"
    )
    rows: list[str] = []

    # Readiness
    rows.append(_str_row("Readiness", readiness_a, readiness_b))

    # Conversion
    conv_a = _safe_get(data_a, "conversion", "success")
    conv_b = _safe_get(data_b, "conversion", "success")
    rows.append(_str_row("Conversion", conv_a, conv_b))

    # Model size (PyTorch)
    pt_size_a = _safe_get(data_a, "model_info", "estimated_size_mb")
    pt_size_b = _safe_get(data_b, "model_info", "estimated_size_mb")
    rows.append(_metric_row("PyTorch Size (MB)", pt_size_a, pt_size_b, ".1f", " MB", lower_is_better=True))

    # Model size (CoreML)
    cm_size_a = _safe_get(data_a, "conversion", "coreml_size_mb")
    cm_size_b = _safe_get(data_b, "conversion", "coreml_size_mb")
    rows.append(_metric_row("CoreML Size (MB)", cm_size_a, cm_size_b, ".1f", " MB", lower_is_better=True))

    # Inference latency
    inf_mean_a = _safe_get(data_a, "benchmark", "inference", "mean_ms")
    inf_mean_b = _safe_get(data_b, "benchmark", "inference", "mean_ms")
    rows.append(_metric_row("Inference Mean (ms)", inf_mean_a, inf_mean_b, ".2f", " ms"))

    inf_p95_a = _safe_get(data_a, "benchmark", "inference", "p95_ms")
    inf_p95_b = _safe_get(data_b, "benchmark", "inference", "p95_ms")
    rows.append(_metric_row("Inference P95 (ms)", inf_p95_a, inf_p95_b, ".2f", " ms"))

    # FPS (higher is better)
    fps_a = _safe_get(data_a, "benchmark", "estimated_fps")
    fps_b = _safe_get(data_b, "benchmark", "estimated_fps")
    rows.append(_metric_row("Estimated FPS", fps_a, fps_b, ".1f", "", lower_is_better=False))

    # Validation
    val_a = _safe_get(data_a, "validation", "status")
    val_b = _safe_get(data_b, "validation", "status")
    rows.append(_str_row("Validation", val_a, val_b))

    # Peak memory
    mem_a = _safe_get(data_a, "benchmark", "peak_memory_mb")
    mem_b = _safe_get(data_b, "benchmark", "peak_memory_mb")
    rows.append(_metric_row("Peak Memory (MB)", mem_a, mem_b, ".1f", " MB"))

    # Bottleneck
    def _bottleneck(data: dict) -> str:
        diag = data.get("diagnosis", {})
        primary = diag.get("primary_category", "unknown")
        if primary and primary != "unknown":
            return primary
        if data.get("benchmark") and not data["benchmark"].get("success", True):
            return "runtime_failure"
        conv = data.get("conversion", {})
        if not conv.get("success", True):
            return "conversion_failure"
        return "none"

    rows.append(_str_row("Main Bottleneck", _bottleneck(data_a), _bottleneck(data_b)))

    metrics_body = table_header + "".join(rows) + "</table>"
    sections.append(_card("Key Metrics Comparison", metrics_body))

    # ------ Benchmark Breakdown Bar Chart ------
    bench_a = data_a.get("benchmark")
    bench_b = data_b.get("benchmark")
    if bench_a or bench_b:
        stages = ["preprocess", "inference", "postprocess", "end_to_end"]
        stage_labels = {
            "preprocess": "Preprocess",
            "inference": "Inference",
            "postprocess": "Postprocess",
            "end_to_end": "End-to-End",
        }
        stage_colors_a = {
            "preprocess": "#60a5fa",
            "inference": "#3b82f6",
            "postprocess": "#818cf8",
            "end_to_end": "#1e293b",
        }
        stage_colors_b = {
            "preprocess": "#93c5fd",
            "inference": "#60a5fa",
            "postprocess": "#a5b4fc",
            "end_to_end": "#64748b",
        }

        # Find max for scaling
        max_val = 0.01
        for stage in stages:
            for bench in (bench_a, bench_b):
                if bench:
                    val = _safe_get(bench, stage, "mean_ms")
                    if val is not None:
                        try:
                            max_val = max(max_val, float(val))
                        except (TypeError, ValueError):
                            pass

        chart_html = (
            '<div style="margin-bottom:12px;font-size:13px;color:var(--color-text-muted)">'
            f'<span style="display:inline-block;width:12px;height:12px;background:#3b82f6;border-radius:2px;margin-right:4px;vertical-align:middle"></span> {_esc(label_a)}'
            f'&nbsp;&nbsp;&nbsp;'
            f'<span style="display:inline-block;width:12px;height:12px;background:#93c5fd;border-radius:2px;margin-right:4px;vertical-align:middle"></span> {_esc(label_b)}'
            '</div>'
            '<div class="bar-chart">'
        )

        for stage in stages:
            val_a_s = _safe_get(bench_a, stage, "mean_ms") if bench_a else None
            val_b_s = _safe_get(bench_b, stage, "mean_ms") if bench_b else None
            label = stage_labels[stage]

            fa = float(val_a_s) if val_a_s is not None else 0
            fb = float(val_b_s) if val_b_s is not None else 0
            pct_a = (fa / max_val) * 100
            pct_b = (fb / max_val) * 100

            chart_html += (
                f'<div class="bar-row">'
                f'<div class="bar-label">{_esc(label)}</div>'
                f'<div class="bar-track" style="display:flex;flex-direction:column;height:auto;gap:2px;padding:2px 0">'
                f'<div style="height:12px;width:{pct_a:.1f}%;background:{stage_colors_a[stage]};border-radius:3px;min-width:2px"></div>'
                f'<div style="height:12px;width:{pct_b:.1f}%;background:{stage_colors_b[stage]};border-radius:3px;min-width:2px"></div>'
                f'</div>'
                f'<div class="bar-value" style="display:flex;flex-direction:column;font-size:12px;line-height:14px;gap:2px;padding:2px 0 2px 8px">'
                f'<div>{_fmt_float(val_a_s, ".2f", " ms")}</div>'
                f'<div>{_fmt_float(val_b_s, ".2f", " ms")}</div>'
                f'</div>'
                f'</div>'
            )

        chart_html += "</div>"
        sections.append(_card("Benchmark Comparison", chart_html))

    # ------ Validation Comparison ------
    val_data_a = data_a.get("validation")
    val_data_b = data_b.get("validation")
    if val_data_a or val_data_b:
        val_body = '<table><tr><th>Check</th><th>{a}</th><th>{b}</th></tr>'.format(
            a=_esc(label_a), b=_esc(label_b)
        )

        # Collect all check names from both sides
        checks_a = {c["name"]: c for c in (val_data_a or {}).get("checks", [])}
        checks_b = {c["name"]: c for c in (val_data_b or {}).get("checks", [])}
        all_checks = list(dict.fromkeys(list(checks_a.keys()) + list(checks_b.keys())))

        status_class_map = {
            "PASS": "status-pass",
            "WARNING": "status-warning",
            "FAIL": "status-fail",
        }

        for name in all_checks:
            ca = checks_a.get(name)
            cb = checks_b.get(name)
            sa = ca["status"] if ca else "N/A"
            sb = cb["status"] if cb else "N/A"
            cls_a = status_class_map.get(sa, "")
            cls_b = status_class_map.get(sb, "")
            val_body += (
                f"<tr>"
                f"<td>{_esc(name)}</td>"
                f'<td class="{cls_a}">{_esc(sa)}</td>'
                f'<td class="{cls_b}">{_esc(sb)}</td>'
                f"</tr>"
            )

        val_body += "</table>"
        sections.append(_card("Validation Comparison", val_body))

    # ------ Run Metadata ------
    meta_body = (
        "<table>"
        f"<tr><th></th><th>{_esc(label_a)}</th><th>{_esc(label_b)}</th></tr>"
    )
    meta_body += (
        f"<tr><td>Run ID</td>"
        f'<td class="mono">{_esc(str(data_a.get("run_id", "N/A")))}</td>'
        f'<td class="mono">{_esc(str(data_b.get("run_id", "N/A")))}</td></tr>'
    )
    meta_body += (
        f"<tr><td>Timestamp</td>"
        f'<td class="mono">{_esc(str(data_a.get("timestamp", "N/A")))}</td>'
        f'<td class="mono">{_esc(str(data_b.get("timestamp", "N/A")))}</td></tr>'
    )
    device_a = _safe_get(data_a, "benchmark", "device_name") or "N/A"
    device_b = _safe_get(data_b, "benchmark", "device_name") or "N/A"
    meta_body += (
        f"<tr><td>Device</td>"
        f'<td class="mono">{_esc(str(device_a))}</td>'
        f'<td class="mono">{_esc(str(device_b))}</td></tr>'
    )
    cu_a = _safe_get(data_a, "benchmark", "compute_unit") or _safe_get(data_a, "conversion", "compute_unit") or "N/A"
    cu_b = _safe_get(data_b, "benchmark", "compute_unit") or _safe_get(data_b, "conversion", "compute_unit") or "N/A"
    meta_body += (
        f"<tr><td>Compute Unit</td>"
        f'<td class="mono">{_esc(str(cu_a))}</td>'
        f'<td class="mono">{_esc(str(cu_b))}</td></tr>'
    )
    meta_body += "</table>"
    sections.append(_card("Run Metadata", meta_body))

    # ------ Note ------
    note = (
        '<div class="note">'
        "<strong>Note:</strong> Differences in device, compute unit, input size, "
        "or model configuration may affect comparability of results."
        "</div>"
    )

    # ------ Additional CSS for comparison-specific styles ------
    extra_css = """
.compare-header-row {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 12px;
}
"""

    body_content = "\n".join(sections) + note

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="UTF-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        "<title>Model2Mobile Comparison Report</title>\n"
        f"<style>\n{_CSS}\n{extra_css}</style>\n"
        "</head>\n"
        "<body>\n"
        f"{header_html}"
        f'<div class="container">\n{body_content}\n</div>\n'
        "</body>\n"
        "</html>\n"
    )
