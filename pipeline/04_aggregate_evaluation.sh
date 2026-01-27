#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/aggregate_human_evaluation.py" \
    --in-dir "${repo_root}/evaluation/human-primary/" \
    --out "${repo_root}/evaluation/human-primary/aggregate_human.json"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/aggregate_human_evaluation.py" \
    --in-dir "${repo_root}/evaluation/human-audit/" \
    --out "${repo_root}/evaluation/human-audit/aggregate_human.json"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/aggregate_llm_evaluation.py" \
    --in-dir "${repo_root}/evaluation/gpt-4o/" \
    --questions "${repo_root}/questions/questions.json" \
    --out "${repo_root}/evaluation/gpt-4o/aggregate_llm.json"
