#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
chat_gpt.py

Batch-generate answers using Azure OpenAI (GPT-4o) for multiple question templates and rs IDs.

Changes from the original:
  1) Input/output paths are configurable via command-line options.
  2) All comments/docstrings are written in English.

Inputs:
  - questions.json (JSON5): a mapping like {"q1": "Tell me ... {rs}.", ...}
  - rs list file: a text/TSV file consumed by utils.load_rs_gene_data()
  - prompt template file: markdown file used as the system prompt

Outputs:
  - Markdown files saved under: <outdir>/<question_no>/<rs>.md
"""

import argparse
import os
import sys
from pathlib import Path

import json5
from dotenv import load_dotenv

from common.utils import load_rs_gene_data, save_answer_to_markdown
from common.open_ai_azure import OpenAIAzure


class ChatGPT(OpenAIAzure):
    def __init__(self, prompt_path: str):
        super().__init__()
        self.prompt_path = prompt_path

    def generate_prompt(self) -> str:
        """Load and format the prompt template."""
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        return prompt_template.format()

    def query(self, question_statement: str) -> str:
        """Query Azure OpenAI with the system prompt and user question."""
        prompt = self.generate_prompt()
        return super().query_azure_openai(prompt, question_statement)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-generate answers with Azure OpenAI using question templates and rs IDs."
    )
    p.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSON/JSON5 (e.g., questions.json).",
    )
    p.add_argument(
        "--rs-list",
        required=True,
        help="Path to rs list file consumed by utils.load_rs_gene_data() (e.g., pubtator3/rs.txt).",
    )
    p.add_argument(
        "--prompt",
        required=True,
        help="Path to prompt template markdown (e.g., prompt.md).",
    )
    p.add_argument(
        "--outdir",
        required=True,
        help="Output directory where answers will be written.",
    )
    p.add_argument(
        "--env",
        default=None,
        help="Optional path to a .env file (default: load from current working directory).",
    )
    p.add_argument(
        "--limit-rs",
        type=int,
        default=None,
        help="Optional limit on number of rs IDs to process (for quick tests).",
    )
    p.add_argument(
        "--limit-questions",
        type=int,
        default=None,
        help="Optional limit on number of question templates to process (for quick tests).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not call the API; just print planned outputs.",
    )
    return p.parse_args()


def load_questions(path: str) -> dict:
    """Load question templates from a JSON/JSON5 file."""
    with open(path, "r", encoding="utf-8") as f:
        return json5.load(f)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    # Load environment variables
    if args.env:
        load_dotenv(args.env)
    else:
        load_dotenv()

    questions_path = os.path.abspath(args.questions)
    rs_list_path = os.path.abspath(args.rs_list)
    prompt_path = os.path.abspath(args.prompt)
    outdir = os.path.abspath(args.outdir)

    # Basic validation
    for fp in [questions_path, rs_list_path, prompt_path]:
        if not os.path.exists(fp):
            print(f"ERROR: File not found: {fp}", file=sys.stderr)
            sys.exit(1)

    ensure_dir(outdir)

    # Load inputs
    questions = load_questions(questions_path)
    rs_gene_list = load_rs_gene_data(rs_list_path)

    # Apply optional limits
    question_items = list(questions.items())
    if args.limit_questions is not None:
        question_items = question_items[: args.limit_questions]

    if args.limit_rs is not None:
        rs_gene_list = rs_gene_list[: args.limit_rs]

    # Instantiate the client once (more efficient than per-request)
    client = ChatGPT(prompt_path=prompt_path)

    # Process each (question template × rs) combination
    for question_no, question_template in question_items:
        q_dir = os.path.join(outdir, str(question_no))
        ensure_dir(q_dir)

        for entry in rs_gene_list:
            rs = entry.get("rs_id")
            if not rs:
                continue

            question_statement = str(question_template).format(rs=rs)
            file_path = os.path.join(q_dir, f"{rs}.md")

            print(f"Processing: {question_no} / {rs} -> {file_path}")

            if args.dry_run:
                continue

            try:
                content = client.query(question_statement)
            except Exception as e:
                print(f"ERROR: API call failed for {question_no} / {rs}: {e}", file=sys.stderr)
                continue

            if content:
                save_answer_to_markdown(file_path, content)
            else:
                print(f"WARNING: Empty response for {question_no} / {rs}", file=sys.stderr)

    print("Done.")


if __name__ == "__main__":
    main()