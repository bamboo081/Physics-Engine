"""Self-contained experiment exports; plotting and PDF work stay off the GUI thread.

Every curve is drawn within a fitted segment. The export never draws a velocity
or energy line through a gap, a target change, or an impact discontinuity.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from . import __version__
from .engine import AnalysisResult, AnalysisSettings

GRAPH_FILES = {
    "trajectory": "trajectory.png",
    "position": "position_time.png",
    "velocity": "velocity_time.png",
    "acceleration": "acceleration_time.png",
    "energy": "energy_time.png",
    "segmentation": "segmentation_debug.png",
}
GRAPH_LABELS = {
    "trajectory": "Trajectory",
    "position": "Position over time",
    "velocity": "Velocity",
    "acceleration": "Acceleration",
    "energy": "Energy",
    "segmentation": "Fitting intervals",
}
UNITS = {
    "Time": "s",
    "X": "m",
    "Y": "m",
    "X_model": "m",
    "Y_model": "m",
    "Vx_model": "m/s",
    "Vy_model": "m/s",
    "Ax_model": "m/s²",
    "Ay_model": "m/s²",
    "Speed_model": "m/s",
    "PE": "J",
    "KE": "J",
    "E_total": "J",
    "Confidence": "detector score, not measurement uncertainty",
    "X_fit_se": "m",
    "Y_fit_se": "m",
    "Vx_fit_se": "m/s",
    "Vy_fit_se": "m/s",
    "Ax_fit_se": "m/s²",
    "Ay_fit_se": "m/s²",
    "SourceTime": "s",
    "Frame": "zero-based source frame index",
    "TrackID": "target continuity identifier",
    "ContinuityID": "continuity interval identifier",
    "SegmentID": "fitted interval identifier",
}
SCHEMA_VERSION = 2
_PLOT_COLORS = ("#2463a6", "#b75520", "#25765f")
# Matplotlib and ReportLab font registration use process-global mutable state.
_EXPORT_LOCK = threading.RLock()


def new_run_dir(base: str | Path) -> Path:
    """Reserve a new directory exclusively; an existing experiment is never replaced."""
    root = Path(base).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(10):
        name = datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%SZ-") + uuid4().hex[:8]
        path = root / name
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise FileExistsError("Could not create a unique output directory")


def _save_fig(fig: Figure, path: Path) -> None:
    try:
        fig.tight_layout(pad=1.5)
        fig.savefig(path, dpi=160, facecolor="white")
    finally:
        fig.clear()


def _axes(title: str, xlabel: str, ylabel: str):
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(alpha=0.18, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    return fig, ax


def _plot_segments(ax, result: AnalysisResult, column: str, label: str, color: str) -> None:
    for index, segment in enumerate(result.segments):
        interval = result.data.iloc[segment.start_index : segment.end_index]
        ax.plot(
            interval["Time"],
            interval[column],
            color=color,
            linewidth=1.5,
            label=label if index == 0 else "_nolegend_",
        )


def _make_plots(result: AnalysisResult, output: Path) -> dict[str, Path]:
    df = result.data
    t = df["Time"].to_numpy(dtype=float)
    paths = {key: output / name for key, name in GRAPH_FILES.items()}

    fig, ax = _axes("Trajectory", "Horizontal position x (m)", "Vertical position y (m)")
    ax.scatter(df["X"], df["Y"], s=10, color="#64748b", alpha=0.65, label="Measured")
    if {"X_model", "Y_model"}.issubset(df.columns):
        for index, segment in enumerate(result.segments):
            interval = df.iloc[segment.start_index : segment.end_index]
            ax.plot(
                interval["X_model"],
                interval["Y_model"],
                color=_PLOT_COLORS[0],
                label="Model" if index == 0 else "_nolegend_",
            )
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(frameon=False, fontsize=9)
    _save_fig(fig, paths["trajectory"])

    fig, ax = _axes("Position", "Elapsed time (s)", "Position (m)")
    for axis_name, color in zip(("X", "Y"), _PLOT_COLORS, strict=False):
        ax.scatter(
            t, df[axis_name], s=9, alpha=0.4, color=color, label=f"Measured {axis_name.lower()}"
        )
        modeled = f"{axis_name}_model"
        if modeled in df:
            _plot_segments(ax, result, modeled, f"Model {axis_name.lower()}", color)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    _save_fig(fig, paths["position"])

    specifications = (
        (
            "velocity",
            ("Vx_model", "Vy_model"),
            ("Horizontal", "Vertical"),
            "Modeled velocity",
            "Velocity (m/s)",
        ),
        (
            "acceleration",
            ("Ax_model", "Ay_model"),
            ("Horizontal", "Vertical"),
            "Modeled acceleration",
            "Acceleration (m/s²)",
        ),
        (
            "energy",
            ("PE", "KE", "E_total"),
            ("Potential", "Kinetic", "Total"),
            "Energy (relative to the chosen y = 0)",
            "Energy (J)",
        ),
    )
    for key, columns, labels, title, ylabel in specifications:
        fig, ax = _axes(title, "Elapsed time (s)", ylabel)
        for column, label, color in zip(columns, labels, _PLOT_COLORS, strict=False):
            _plot_segments(ax, result, column, label, color)
        ax.legend(frameon=False, fontsize=9)
        _save_fig(fig, paths[key])

    fig, ax = _axes("Fitting intervals", "Elapsed time (s)", "Measured vertical position (m)")
    missing = df["SegmentID"].isna()
    if missing.any():
        ax.scatter(
            t[missing],
            df.loc[missing, "Y"],
            s=12,
            color="#a8adb5",
            marker="x",
            label="Too few samples",
        )
    for index, segment in enumerate(result.segments):
        interval = df.iloc[segment.start_index : segment.end_index]
        ax.scatter(
            interval["Time"],
            interval["Y"],
            s=13,
            color=_PLOT_COLORS[index % len(_PLOT_COLORS)],
            label=f"Interval {segment.index}" if len(result.segments) <= 8 else None,
        )
    if len(result.segments) <= 8:
        ax.legend(frameon=False, fontsize=8, ncol=min(3, len(result.segments) + int(missing.any())))
    _save_fig(fig, paths["segmentation"])
    return paths


def _number(value: Any, digits: int = 3) -> str:
    return "Not available" if value is None else f"{value:.{digits}f}"


def summary_text(result: AnalysisResult, experiment: str) -> str:
    """A readable summary shared by the desktop, CLI, and plain-text export."""
    stats = result.stats
    lines = [
        f"Experiment: {experiment}",
        f"Samples: {stats['samples']} ({stats['modeled_samples']} modeled)",
        f"Fitting intervals: {len(result.segments)}",
        f"Duration: {stats['duration_s']:.3f} s",
        f"Maximum measured y: {stats['max_height_m']:.3f} m (chosen datum)",
        f"Maximum modeled speed: {_number(stats.get('max_modeled_speed_m_s'))} m/s",
    ]
    gravity = stats.get("gravity_estimate_m_s2")
    lines.append(
        f"Qualified gravity estimate: {gravity:.3f} m/s²"
        if gravity is not None
        else "Gravity: not established by qualifying measurements"
    )
    gravity_se = stats.get("gravity_fit_standard_error_m_s2")
    if gravity is not None and gravity_se is not None:
        lines.append(
            f"Gravity fit standard error: {gravity_se:.3g} m/s² (not total measurement uncertainty)"
        )
    restitution = stats.get("restitution_estimate")
    lines.append(
        f"Qualified restitution estimate: {restitution:.3f}"
        if restitution is not None
        else "Restitution: not established by qualifying measurements"
    )
    lines.append("\nFitting intervals:")
    for segment in result.segments:
        lines.append(
            f"{segment.index}: {segment.t_start:.3f}–{segment.t_end:.3f} s; "
            f"R²(x)={segment.r2_x:.3f}, R²(y)={segment.r2_y:.3f}"
        )
    if result.warnings:
        lines.append("\nMeasurement notes:")
        lines.extend(f"- {warning}" for warning in result.warnings)
    return "\n".join(lines)


def _pdf_styles():
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for name, filename in (
        ("MotionSans", "DejaVuSans.ttf"),
        ("MotionSans-Bold", "DejaVuSans-Bold.ttf"),
    ):
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    pdfmetrics.registerFontFamily("MotionSans", normal="MotionSans", bold="MotionSans-Bold")
    styles = getSampleStyleSheet()
    for style in styles.byName.values():
        style.fontName = "MotionSans"
        style.textColor = colors.HexColor("#202b38")
    styles["Normal"].fontSize = 9
    styles["Normal"].leading = 13
    styles["Title"].fontName = "MotionSans-Bold"
    styles["Title"].fontSize = 22
    styles["Title"].leading = 27
    styles["Title"].alignment = 0
    for name in ("Heading2", "Heading3"):
        styles[name].fontName = "MotionSans-Bold"
    styles.add(ParagraphStyle("SmallText", parent=styles["Normal"], fontSize=8, leading=11))
    return styles


def _paragraph(value: Any, style) -> Paragraph:
    # ReportLab paragraphs parse XML-like markup; all external text is escaped.
    return Paragraph(escape(str(value)).replace("\n", "<br/>"), style)


def _pdf_report(
    result: AnalysisResult,
    settings: AnalysisSettings,
    paths: dict[str, Path],
    dest: Path,
    context: dict,
) -> None:
    styles = _pdf_styles()
    doc = SimpleDocTemplate(
        str(dest),
        pagesize=letter,
        leftMargin=48,
        rightMargin=48,
        topMargin=45,
        bottomMargin=45,
        title="Motion analysis",
        author="Physics Motion Analyzer",
    )
    elements = [
        Paragraph("Motion analysis", styles["Title"]),
        _paragraph(settings.experiment_type, styles["Heading2"]),
        Spacer(1, 8),
    ]
    info = [
        ("Source", context.get("origin", context.get("source", "Recorded measurements"))),
        ("Object", context.get("object_name", "Not specified")),
        ("Mass", f"{settings.mass_kg:.6g} kg (supplied by the operator)"),
        (
            "Samples",
            f"{result.stats['samples']} measured; {result.stats['modeled_samples']} modeled",
        ),
        ("Duration", f"{result.stats['duration_s']:.3f} s"),
        ("Maximum y", f"{result.stats['max_height_m']:.3f} m (chosen datum)"),
        ("Maximum speed", f"{_number(result.stats.get('max_modeled_speed_m_s'))} m/s (modeled)"),
    ]
    rows = [
        [_paragraph(key, styles["Normal"]), _paragraph(value, styles["Normal"])]
        for key, value in info
    ]
    table = Table(rows, colWidths=[125, doc.width - 125], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.HexColor("#f4f6f8"), colors.white]),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    elements += [table, Spacer(1, 12), Paragraph("Physics estimates", styles["Heading2"])]
    for label, key, unit in (
        ("Gravity", "gravity_estimate_m_s2", "m/s²"),
        ("Restitution", "restitution_estimate", ""),
    ):
        value = result.stats.get(key)
        text = (
            f"{label}: {value:.4g} {unit}"
            if value is not None
            else f"{label}: not established by qualifying measurements."
        )
        elements.append(_paragraph(text, styles["Normal"]))
        if key == "gravity_estimate_m_s2" and value is not None:
            error = result.stats.get("gravity_fit_standard_error_m_s2")
            if error is not None:
                elements.append(
                    _paragraph(
                        f"Gravity fit standard error: {error:.3g} m/s². "
                        "This excludes calibration and other measurement systematics.",
                        styles["SmallText"],
                    )
                )
    elements += [Spacer(1, 10), Paragraph("Method", styles["Heading2"])]
    model_types = {getattr(segment, "model_type", "quadratic") for segment in result.segments}
    if any("local" in str(model) for model in model_types):
        method = (
            f"General motion uses local polynomial fits against the recorded timestamps "
            f"(requested window: {getattr(settings, 'smoothing_window', 9)} samples). "
            "Derivatives describe the smoothed path and depend on the chosen window."
        )
    else:
        method = (
            "Ballistic intervals use least-squares quadratic position models in each axis "
            "against the recorded timestamps. Velocity and acceleration are analytic derivatives."
        )
    elements.append(_paragraph(method, styles["Normal"]))
    elements.append(
        _paragraph(
            "Missing observations and interval boundaries are not interpolated. Potential energy uses "
            "g = 9.81 m/s² and the chosen y = 0 datum; kinetic energy uses modeled speed. Fit standard "
            "errors, when present in the CSV/JSON, do not include calibration or camera systematics.",
            styles["Normal"],
        )
    )
    elements += [Spacer(1, 12), Paragraph("Measurement notes", styles["Heading2"])]
    for warning in result.warnings:
        elements.extend([_paragraph(f"• {warning}", styles["SmallText"]), Spacer(1, 4)])
    elements += [Spacer(1, 10), Paragraph("Fitting intervals", styles["Heading2"])]
    header = ("Interval", "Time (s)", "Samples", "R² x", "R² y")
    interval_rows = [[_paragraph(value, styles["SmallText"]) for value in header]]
    for segment in result.segments[:100]:
        values = (
            segment.index,
            f"{segment.t_start:.3f}–{segment.t_end:.3f}",
            segment.end_index - segment.start_index,
            f"{segment.r2_x:.4f}",
            f"{segment.r2_y:.4f}",
        )
        interval_rows.append([_paragraph(value, styles["SmallText"]) for value in values])
    interval_table = Table(interval_rows, colWidths=[62, 180, 82, 90, 102], repeatRows=1)
    interval_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef4")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.HexColor("#b8c3d0")),
            ]
        )
    )
    elements.append(interval_table)
    if len(result.segments) > 100:
        elements.append(
            _paragraph("All intervals are included in summary.json.", styles["SmallText"])
        )
    elements.append(PageBreak())
    for index, (key, img) in enumerate(paths.items()):
        if index > 0 and index % 2 == 0:
            elements.append(PageBreak())
        elements.append(
            KeepTogether(
                [
                    _paragraph(GRAPH_LABELS[key], styles["Heading2"]),
                    Image(str(img), width=doc.width, height=doc.width * 4.5 / 8),
                    Spacer(1, 14),
                ]
            )
        )

    def footer(canvas, document):
        canvas.saveState()
        canvas.setFont("MotionSans", 8)
        canvas.setFillColor(colors.HexColor("#657182"))
        canvas.drawString(48, 24, f"Physics Motion Analyzer {__version__}")
        canvas.drawRightString(letter[0] - 48, 24, str(document.page))
        canvas.restoreState()

    doc.build(elements, onFirstPage=footer, onLaterPages=footer)


def _json_value(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported metadata type: {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=_json_value)
        + "\n",
        encoding="utf-8",
    )


def export_run(
    result: AnalysisResult,
    settings: AnalysisSettings,
    root: str | Path,
    context: dict | None = None,
    raw_data: object | None = None,
) -> Path:
    """Export a complete run, with a checksum manifest written last.

    A failed export removes only its own newly reserved directory. Callers retain
    their measurements and can retry another destination without re-recording.
    """
    settings.validate()
    if result.settings is not None and result.settings != settings:
        raise ValueError("Export settings must match the settings used to calculate this result")
    context = dict(context or {})
    folder = new_run_dir(root)
    try:
        result.data.to_csv(folder / "analysis_data.csv", index=False, encoding="utf-8")
        if raw_data is None:
            raw_data = getattr(result, "raw_samples", None)
        if raw_data is None:
            raw_columns = [
                name
                for name in ("Time", "X", "Y", "Confidence", "TrackID", "Frame", "SourceTime")
                if name in result.data
            ]
            raw_data = result.data[raw_columns]
        pd.DataFrame(raw_data).to_csv(folder / "raw_samples.csv", index=False, encoding="utf-8")
        created = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _write_json(
            folder / "config.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": created,
                "software": {"name": "Physics Motion Analyzer", "version": __version__},
                "settings": asdict(settings),
                "context": context,
                "units": UNITS,
                "coefficient_convention": "Descending powers of tau = Time - time_origin_s; local fits have no global coefficients.",
            },
        )
        _write_json(
            folder / "summary.json",
            {
                "schema_version": SCHEMA_VERSION,
                "stats": result.stats,
                "segments": [segment.to_dict() for segment in result.segments],
                "warnings": result.warnings,
            },
        )
        (folder / "summary.txt").write_text(
            summary_text(result, settings.experiment_type) + "\n", encoding="utf-8"
        )
        with _EXPORT_LOCK:
            paths = _make_plots(result, folder)
            _pdf_report(result, settings, paths, folder / "Analysis_Report.pdf", context)
        files = {}
        for path in sorted(folder.iterdir()):
            contents = path.read_bytes()
            files[path.name] = {
                "bytes": len(contents),
                "sha256": hashlib.sha256(contents).hexdigest(),
            }
        _write_json(
            folder / "manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": created,
                "software_version": __version__,
                "files": files,
            },
        )
        return folder
    except BaseException:
        # KeyboardInterrupt/SystemExit must not leave a seemingly complete report either.
        shutil.rmtree(folder, ignore_errors=True)
        raise
