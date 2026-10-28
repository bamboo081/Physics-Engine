# Sharing and publishing the source

The complete project can be shared as this ZIP or placed in a Git repository. It does not require a web server, hosted account, API key, or cloud database.

## Before a release

1. Install `.[desktop,dev]` in a clean Python 3.12 environment.
2. Run `python -m pytest -q`, `python -m ruff check .`, and `python -m ruff format --check .`.
3. Run the CLI demo and the included marker-video workflow. Inspect the PDF and interval boundaries.
4. Run `python -m build`; inspect the wheel/source distribution and test the wheel from outside the source directory.
5. Record the actual OS/interpreter/dependency versions and any physical-device checks in `docs/validation.md`.
6. Keep version and schema/methodology documentation consistent. The Python package version is read from `physics_motion.__version__`.

The GitHub Actions workflow runs core checks on Windows, macOS, and Linux with Python 3.10–3.13, and desktop checks on all three operating systems with Python 3.12. Confirm those jobs pass on your repository before claiming that CI matrix has been validated.

## What to include

Include source, tests, README, license, launchers, package configuration, documentation, and the synthetic examples. Preserve the notices for any third-party components you add. Keep the old data archive labeled as unverified.

Do not include `.venv`, Python caches, local experiment folders, personal recordings, credentials, downloaded model weights, temporary screenshots, crash dumps, or build caches. The provided `.gitignore` covers the usual local artifacts.

## Source versus packaged application

This release is a source distribution. The launcher creates a local Python environment on the recipient's machine. Building a signed Windows installer or macOS application is a separate platform-specific packaging step; do not rename the ZIP to imply it contains those binaries.

If you bundle dependencies or optional model weights into an executable, review their licenses and distribution requirements. The project's MIT license covers its own source, not all dependencies. See `third-party-notices.md`.
