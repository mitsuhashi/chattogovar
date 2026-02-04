#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

## generate Table 3
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/table3.py" \
  --manual "${repo_root}/evaluation/human-mean/aggregate_human.json" \
  --llm "${repo_root}/evaluation/gpt-4o/aggregate_llm.json" \
  --out "${repo_root}/evaluation/tables/table3.md" \
  --manual_primary "${repo_root}/evaluation/human-primary/aggregate_human.json" \
  --manual_secondary "${repo_root}/evaluation/human-secondary/aggregate_human.json" \
  --out_supp "${repo_root}/evaluation/tables/table_s1_supplement_to_table3.md"

## generate Table 4 and Supplementary Table 4
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/table4.py" \
  --manual "${repo_root}/evaluation/human-mean/aggregate_human.json" \
  --manual_primary "${repo_root}/evaluation/human-primary/aggregate_human.json" \
  --manual_secondary "${repo_root}/evaluation/human-secondary/aggregate_human.json" \
  --llm "${repo_root}/evaluation/gpt-4o/aggregate_llm.json" \
  --supp_include_llm \
  --out_main "${repo_root}/evaluation/tables/table4.md" \
  --out_supp "${repo_root}/evaluation/tables/table_s2_supplement_to_table4.md"

## generate Table 5 and Supplementary Table 5
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/table5.py" \
  --manual "${repo_root}/evaluation/human-mean/aggregate_human.json" \
  --manual_primary "${repo_root}/evaluation/human-primary/aggregate_human.json" \
  --manual_secondary "${repo_root}/evaluation/human-secondary/aggregate_human.json" \
  --llm "${repo_root}/evaluation/gpt-4o/aggregate_llm.json" \
  --q2cat "${repo_root}/questions/question_categories.json" \
  --out_main "${repo_root}/evaluation/tables/table5.md" \
  --out_supp "${repo_root}/evaluation/tables/table_s3_supplement_to_table5.md"
## generate table of human audit agreement
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/stats/table_icc.py" \
    --primary "${repo_root}/evaluation/human-primary/aggregate_human.json" \
    --secondary "${repo_root}/evaluation/human-secondary/aggregate_human.json" \
    --out "${repo_root}/evaluation/tables/table_s4_supplement_to_table4.md"