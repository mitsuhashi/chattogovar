#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

## generate Table 3
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/table3.py" \
  --manual "${repo_root}/evaluation/human-final/aggregate_human.json" \
  --llm "${repo_root}/evaluation/gpt-4o/aggregate_llm.json" \
  --out "${repo_root}/evaluation/tables/table3.md"

## generate Table 4 and Supplementary Table 4
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/table4.py" \
  --manual "${repo_root}/evaluation/human-final/aggregate_human.json" \
  --llm "${repo_root}/evaluation/gpt-4o/aggregate_llm.json" \
  --out_main "${repo_root}/evaluation/tables/table4.md" \
  --out_supp "${repo_root}/evaluation/tables/table4_supplementary.md"

## generate Table 5 and Supplementary Table 4
PYTHONPATH="${repo_root}/src" python "${repo_root}/src/stats/table5.py" \
  --manual "${repo_root}/evaluation/human-final/aggregate_human.json" \
  --llm "${repo_root}/evaluation/gpt-4o/aggregate_llm.json" \
  --q2cat "${repo_root}/questions/question_categories.json" \
  --out_main "${repo_root}/evaluation/tables/table5.md" \
  --out_supp "${repo_root}/evaluation/tables/table5_supplementary.md"