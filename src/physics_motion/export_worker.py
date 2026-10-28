"""Contain desktop report-rendering failures in a fresh, short-lived process.

Only trusted in-memory analysis objects cross the private multiprocessing pipe;
no input file is deserialized as Python objects. The command-line exporter stays
in-process. The desktop calls this boundary from its numerical worker thread.
"""

from __future__ import annotations

import multiprocessing
import shutil
from multiprocessing.connection import Connection, wait
from pathlib import Path
from tempfile import TemporaryDirectory

from .engine import AnalysisResult, AnalysisSettings


def _render_export(
    connection: Connection,
    result: AnalysisResult,
    settings: AnalysisSettings,
    stage: Path,
    context: dict,
    raw_data: object | None,
) -> None:
    """Run without Qt objects and return only a folder name or an error string."""
    try:
        from .output import export_run

        folder = export_run(result, settings, stage, context, raw_data)
        connection.send(("ok", folder.name))
    except BaseException as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def export_run_isolated(
    result: AnalysisResult,
    settings: AnalysisSettings,
    root: str | Path,
    context: dict | None = None,
    raw_data: object | None = None,
) -> Path:
    """Render a desktop report in a spawned child and publish only a complete run.

    The parent owns the staging directory, so even a native child crash removes
    partial files. A successful IPC reply is insufficient: the child must also
    exit cleanly before files are promoted, with the checksum manifest last.
    Existing reports and the caller's observations are never modified.
    """
    settings.validate()
    if result.settings is not None and result.settings != settings:
        raise ValueError("Export settings must match the settings used to calculate this result")
    context = dict(context or {})
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)

    # Never fork an active Qt process or change the application's global method.
    spawn = multiprocessing.get_context("spawn")
    with TemporaryDirectory(prefix=".report-", dir=root) as temporary:
        stage = Path(temporary)
        receive, send = spawn.Pipe(duplex=False)
        process = spawn.Process(
            target=_render_export,
            args=(send, result, settings, stage, context, raw_data),
            name="physics-motion-report",
        )
        response = None
        try:
            process.start()
            # Only the child owns the write end now; a crash must produce EOF.
            send.close()
            wait((receive, process.sentinel))
            if receive.poll():
                try:
                    response = receive.recv()
                except (EOFError, OSError):
                    pass
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    "Report renderer exited unexpectedly "
                    f"(exit code {process.exitcode}). Your input samples are retained; "
                    "try analyzing again."
                )
            if response is None:
                raise RuntimeError("Report renderer exited without returning a completed report")
            status, detail = response
            if status == "error":
                raise RuntimeError(f"Report export failed: {detail}")
            if status != "ok" or not isinstance(detail, str) or Path(detail).name != detail:
                raise RuntimeError("Report renderer returned an invalid output location")
            folder = stage / detail
            if not folder.is_dir() or not (folder / "manifest.json").is_file():
                raise RuntimeError("Report renderer did not produce a complete report")
        finally:
            receive.close()
            send.close()
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                process.join()
            process.close()

        from .output import new_run_dir

        published = new_run_dir(root)
        try:
            for path in sorted(folder.iterdir(), key=lambda item: item.name == "manifest.json"):
                path.replace(published / path.name)
            return published
        except BaseException:
            shutil.rmtree(published, ignore_errors=True)
            raise
