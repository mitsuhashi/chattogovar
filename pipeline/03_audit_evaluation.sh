#!/usr/bin/env bash
#set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/evaluation/audit_evaluation.py" \
    "${repo_root}/evaluation/human-primary/" \
    --glob "evaluation*.json" \
    --out "${repo_root}/audit/audit_evaluation_human-primary.tsv"

PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/evaluation/audit_evaluation.py" \
    "${repo_root}/evaluation/human-audit/" \
    --glob "evaluation*.json" \
    --out "${repo_root}/audit/audit_evaluation_human-audit.tsv"