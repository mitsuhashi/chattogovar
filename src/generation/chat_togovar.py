#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chat_togovar.py

Generate ChatTogoVar answers (Azure OpenAI + TogoVar JSON evidence) for (questions.json × rs list)
and save as Markdown.

Features
- All input/output paths are provided via CLI options (no fixed paths).
- Optional rs limiting for quick smoke tests (--rs-limit).
- English-only comments.
- Injects TogoVar API JSON response into prompt template using {togovar_response} placeholder.

Example:
  python3 chat_togovar.py \
    --questions questions.json \
    --rs-file pubtator3/rs.txt \
    --prompt pipeline/prompts/chat_togovar/prompt.md \
    --outdir /tmp/answers/chat_togovar \
    --rs-limit 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import json5
import requests
from dotenv import load_dotenv

# Ensure repo src/common is importable regardless of current working directory
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common.utils import load_rs_gene_data, save_answer_to_markdown  # noqa: E402
from common.open_ai_azure import OpenAIAzure  # noqa: E402


class ChatTogoVar(OpenAIAzure):
    def __init__(self, prompt_path: str, togovar_api: str):
        super().__init__()
        self.prompt_path = prompt_path
        self.togovar_api = togovar_api.rstrip("/")

    def search_togovar(self, rs: str) -> dict | None:
        """Query TogoVar API for a given variant identifier (rs number)."""
        try:
            resp = requests.post(
                f"{self.togovar_api}/api/search/variant",
                json={"query": {"id": [rs]}},
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] TogoVar API error: {e}")
            return None

    def generate_prompt(self, rs: str) -> str:
        """Load prompt template and inject TogoVar JSON evidence."""
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        togovar_response = self.search_togovar(rs)
        if togovar_response is None:
            togovar_json = "{}"
        else:
            # Keep evidence as JSON string for prompt injection
            togovar_json = json.dumps(togovar_response, ensure_ascii=False)

        return prompt_template.format(togovar_response=togovar_json)

    def query(self, question_statement: str, rs: str) -> str | None:
        prompt = self.generate_prompt(rs)
        return super().query_azure_openai(prompt, question_statement)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True, help="Path to questions.json (JSON5 allowed)")
    ap.add_argument("--rs-list", required=True, help="Path to rs list file (e.g., pubtator3/rs.txt)")
    ap.add_argument("--prompt", required=True, help="Path to prompt.md template")
    ap.add_argument("--outdir", required=True, help="Output directory for Markdown answers")
    ap.add_argument(
        "--togovar-api",
        default="https://grch38.togovar.org",
        help="Base URL for TogoVar (default: https://grch38.togovar.org)",
    )
    ap.add_argument("--rs-limit", type=int, default=0, help="Limit number of rs entries (0 = no limit)")
    ap.add_argument("--env", default="", help="Optional path to .env (default: load from current dir/repo)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    if args.env:
        load_dotenv(args.env)
    else:
        load_dotenv()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json5.load(f)

    rs_gene_list = load_rs_gene_data(args.rs_list)
    if args.rs_limit and args.rs_limit > 0:
        rs_gene_list = rs_gene_list[: args.rs_limit]

    client = ChatTogoVar(prompt_path=args.prompt, togovar_api=args.togovar_api)

    for question_no, question_statement_template in questions.items():
        print(f"[INFO] Processing question {question_no}: {question_statement_template}")
        (out_dir / question_no).mkdir(parents=True, exist_ok=True)

        for entry in rs_gene_list:
            rs = entry["rs_id"]
            question_statement = question_statement_template.format(rs=rs)
            print(f"[INFO] Processing: {question_statement}")

            answer = client.query(question_statement, rs)
            if answer:
                file_path = out_dir / question_no / f"{rs}.md"
                save_answer_to_markdown(str(file_path), answer)
                print("[INFO] done.")
            else:
                print("[WARN] No response from Azure OpenAI")


if __name__ == "__main__":
    main()