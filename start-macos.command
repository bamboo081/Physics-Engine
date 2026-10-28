#!/usr/bin/env bash
set -u
cd -- "$(dirname -- "$0")" || exit 1
status=1
for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        "$candidate" launch.py "$@"
        status=$?
        break
    fi
done
if [ "$status" -ne 0 ]; then
    printf '%s\n' 'The app could not start. See the message above and docs/troubleshooting.md.'
    read -r -p 'Press Return to close this window.' _reply
fi
exit "$status"
