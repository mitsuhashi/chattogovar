#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/aggregate_human_evaluation.py" \
    --in-dir "${repo_root}/evaluation/human-primary/" \
    --out "${repo_root}/evaluation/human-primary/aggregate_human.json"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/aggregate_human_evaluation.py" \
    --in-dir "${repo_root}/evaluation/human-secondary/" \
    --out "${repo_root}/evaluation/human-secondary/aggregate_human.json"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/make_human_mean.py" \
    --primary "${repo_root}/evaluation/human-primary/aggregate_human.json" \
    --secondary "${repo_root}/evaluation/human-secondary/aggregate_human.json" \
    --out "${repo_root}/evaluation/human-mean/aggregate_human.json"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/aggregate_llm_evaluation.py" \
    --in-dir "${repo_root}/evaluation/gpt-4o/" \
    --questions "${repo_root}/questions/questions.json" \
    --out "${repo_root}/evaluation/gpt-4o/aggregate_llm.json"
