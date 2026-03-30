"""Report generators for Model2Mobile pipeline results."""

from model2mobile.report.comparison import generate_comparison_html
from model2mobile.report.html import generate_html, save_html
from model2mobile.report.json_report import save_json_reports
from model2mobile.report.markdown import generate_markdown, save_markdown
from model2mobile.report.optimization import (
    generate_optimization_report,
    save_optimization_report,
)

__all__ = [
    "generate_comparison_html",
    "generate_html",
    "generate_markdown",
    "generate_optimization_report",
    "save_html",
    "save_json_reports",
    "save_markdown",
    "save_optimization_report",
]
