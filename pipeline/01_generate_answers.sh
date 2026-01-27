#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

# GPT-4o generation
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/generation/chat_gpt.py" \
  --questions "${repo_root}/questions/questions.json" \
  --rs-list "${repo_root}/questions/pubtator3/rs_30.txt" \
  --prompt "${repo_root}/answers/gpt-4o/prompt.md" \
  --outdir "${repo_root}/answers/gpt-4o" \
  --env "${repo_root}/.env"

# ChatTogoVar generation
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/generation/chat_togovar.py" \
  --questions "${repo_root}/questions/questions.json" \
  --rs-list "${repo_root}/questions/pubtator3/rs_30.txt" \
  --prompt "${repo_root}/answers/chat_togovar/prompt.md" \
  --outdir "${repo_root}/answers/chat_togovar" \
  --env "${repo_root}/.env"

# VarChat generation
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/generation/varchat.py" \
  --rs-list "${repo_root}/questions/pubtator3/rs_30.txt" \
  --outdir "${repo_root}/answers/varchat" \
  --env "${repo_root}/.env"

# format Varchat answers
PYTHONPATH="${repo_root}/src" python3 "${repo_root}/src/generation/varchat_format.py" \
    --in-dir "${repo_root}/answers/varchat/original" \
    --out-dir "${repo_root}/answers/varchat" \
    --translate-ja
