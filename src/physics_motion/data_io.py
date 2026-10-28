"""Strict, local CSV import shared by the desktop and command-line tools."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

import pandas as pd

MAX_CSV_BYTES = 128 * 1024 * 1024
MAX_CSV_ROWS = 250_000


def read_samples_csv(
    path: str | Path, *, max_bytes: int = MAX_CSV_BYTES, max_rows: int = MAX_CSV_ROWS
) -> pd.DataFrame:
    """Read observations without silently accepting duplicate or shifted columns.

    Headers are case-sensitive, except surrounding whitespace and a UTF-8 BOM
    are ignored. Units remain the caller's responsibility: seconds and meters.
    Extra columns are preserved. Inputs are local files, never URLs.
    """
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"CSV file not found: {source}")
    if source.stat().st_size > max_bytes:
        raise ValueError(f"CSV is too large; the import limit is {max_bytes // (1024 * 1024)} MiB")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise ValueError("CSV file is empty")
            header = [name.strip() for name in header]
            duplicate = [name for name, count in Counter(header).items() if count > 1]
            if duplicate:
                raise ValueError(f"Duplicate CSV column names: {', '.join(duplicate)}")
            if not all(header):
                raise ValueError("CSV column names must not be blank")
            if not {"Time", "X", "Y"}.issubset(header):
                raise ValueError("CSV must contain Time, X, Y columns in seconds, meters, meters")
            rows = []
            for row in reader:
                if not row or (len(row) == 1 and not row[0].strip()):
                    continue
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV line {reader.line_num} has {len(row)} columns; expected {len(header)}"
                    )
                rows.append(row)
                if len(rows) > max_rows:
                    raise ValueError(f"CSV is too large; the import limit is {max_rows:,} rows")
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f"Could not read CSV as UTF-8 comma-separated data: {exc}") from exc
    if not rows:
        raise ValueError("CSV contains a header but no measurements")
    frame = pd.DataFrame(rows, columns=header)
    for name in ("Time", "X", "Y", "Confidence", "Frame", "SourceTime"):
        if name in frame:
            try:
                frame[name] = pd.to_numeric(frame[name], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(f"CSV column {name} must contain numeric measurements") from exc
    return frame
