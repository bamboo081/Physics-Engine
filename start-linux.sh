#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")" || exit 1
for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        exec "$candidate" launch.py "$@"
    fi
done
printf '%s\n' 'Python 3.10–3.13 is required. Install Python 3.12, then run this file again.' >&2
exit 1
