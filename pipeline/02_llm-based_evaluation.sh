#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/evaluation/llm-based_evaluation.py" \
    --answers-root "${repo_root}/answers/" \
    --out-root "${repo_root}/evaluation/varchat/" \
    --questions "${repo_root}/questions/questions.json" \
    --rs-gene "${repo_root}/questions/pubtator3/rs_30_with_symbol.txt" \
    --prompt "${repo_root}/evaluation/gpt-4o/prompt.md" \
    --dotenv "${repo_root}/.env"
