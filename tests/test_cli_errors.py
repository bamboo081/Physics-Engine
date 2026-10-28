import subprocess
import sys


def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "physics_motion.cli", *args],
        text=True,
        capture_output=True,
        timeout=40,
    )


def test_version_has_no_desktop_dependency():
    result = run_cli("--version")
    assert result.returncode == 0
    assert "Physics Motion Analyzer" in result.stdout


def test_doctor_reports_desktop_availability_without_camera():
    result = run_cli("doctor")
    assert result.returncode == 0
    assert "Python" in result.stdout
    assert "OpenCV" in result.stdout


def test_duplicate_csv_header_not_silently_accepted(tmp_path):
    source = tmp_path / "duplicate.csv"
    source.write_text("Time,X,X,Y\n" + "\n".join(f"{i},1,2,3" for i in range(20)))
    result = run_cli("analyze", "--input", str(source), "--output", str(tmp_path / "results"))
    assert result.returncode == 2
    assert "Duplicate" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "results").exists()


def test_bad_mass_is_clean_error_and_no_run(tmp_path):
    result = run_cli("demo", "--mass", "nan", "--output", str(tmp_path / "results"))
    assert result.returncode == 2
    assert "Mass" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "results").exists()


def test_missing_input_is_clean_error(tmp_path):
    result = run_cli("analyze", "--input", str(tmp_path / "absent.csv"))
    assert result.returncode == 2
    assert "Traceback" not in result.stderr


def test_bad_smoothing_window_is_clean_error(tmp_path):
    result = run_cli("demo", "--smoothing-window", "8", "--output", str(tmp_path / "results"))
    assert result.returncode == 2
    assert "odd" in result.stderr
    assert "Traceback" not in result.stderr
