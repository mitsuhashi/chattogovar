#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

## compare primary and audit evaluation to calcularte ICC and Kappa
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/calc_icc_kappa.py" \
  --primary "${repo_root}/evaluation/human-primary/aggregate_human.json" \
  --audit "${repo_root}/evaluation/human-audit/aggregate_human.json" \
  --out "${repo_root}/evaluation/audit/primary_audit_agreement.json"