#!/usr/bin/env bash
# Copy every live job store to a tracked snapshot and commit + push it, so results survive the loss
# of the machine running the jobs. Usage: bash revision/snapshot.sh "message"
set -euo pipefail
cd "$(dirname "$0")/.."
for f in revision/results/*/*store.jsonl; do
  [ -f "$f" ] && cp "$f" "${f%.jsonl}.snapshot.jsonl"
done
git add revision/results .gitignore revision/snapshot.sh
git diff --cached --quiet || git commit -q -m "${1:-Snapshot revision results}"
git push -q origin "$(git branch --show-current)"
