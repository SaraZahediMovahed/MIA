#!/usr/bin/env bash
# HTCondor (mia.sub): trains, writes submission.csv, then uploads via task_template.py.
# Execute node needs outbound HTTPS to the course server or the job fails at submit.

set -euo pipefail

# Many Linux images only ship `python3`, not `python`.
PY="${PY:-python3}"
command -v "$PY" >/dev/null || PY=python
command -v "$PY" >/dev/null || { echo "Need python3 or python on PATH" >&2; exit 1; }

echo "Python: $($PY --version)"
echo "Working dir: $(pwd)"

"$PY" -m pip install --no-cache-dir -r requirements.txt

export PYTHONUNBUFFERED=1
exec "$PY" -u task_template.py "$@"
