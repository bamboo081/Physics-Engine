"""CSV import must never silently drop columns or infer an unintended index."""

import pandas as pd
import pytest

from physics_motion.data_io import read_samples_csv


def test_import_normalizes_bom_and_header_spaces(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("\ufeff Time , X ,Y,TrackID\n0,1,2,0\n1,2,3,0\n\n", encoding="utf-8")
    result = read_samples_csv(path)
    assert list(result) == ["Time", "X", "Y", "TrackID"]
    assert result.shape == (2, 4)
    assert pd.to_numeric(result["X"]).tolist() == [1, 2]


@pytest.mark.parametrize(
    "content, message",
    [
        ("Time,X,X,Y\n0,1,2,3\n", "Duplicate"),
        ("Time, X ,X,Y\n0,1,2,3\n", "Duplicate"),
        ("Time,X,Y\n0,1,2,3\n", "columns"),
        ("Time,X,Y\n0,1\n", "columns"),
        ("", "empty"),
        ("Time,X,Y\n", "measurements"),
        ("Time,X,Height\n0,1,2\n", "Time, X, Y"),
    ],
)
def test_invalid_csv_fails_clearly(tmp_path, content, message):
    path = tmp_path / "source.csv"
    path.write_text(content)
    with pytest.raises(ValueError, match=message):
        read_samples_csv(path)


def test_csv_size_guard(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text("Time,X,Y\n0,1,2\n")
    with pytest.raises(ValueError, match="large"):
        read_samples_csv(path, max_bytes=4)


def test_import_preserves_quoted_metadata(tmp_path):
    path = tmp_path / "source.csv"
    path.write_text('Time,X,Y,Note\n0,1,2,"ball, first frame"\n')
    result = read_samples_csv(path)
    assert result.loc[0, "Note"] == "ball, first frame"
