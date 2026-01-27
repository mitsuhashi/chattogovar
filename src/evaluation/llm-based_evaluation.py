#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
answer_quality_evaluator.py

Batch evaluation runner (GPT-4o/Azure OpenAI) that scores answers from:
  - ChatTogoVar
  - GPT-4o
  - VarChat

Refactor points:
  1) Input/output paths are provided via CLI arguments.
  2) Japanese comments are translated to English.
  3) Common helpers are imported from common/utils.py.
  4) Adds --rs-limit for quick smoke tests.

Expected answer layout under --answers-root (customizable):
  answers_root/
    chat_togovar/q1/rs123.md
    chat_gpt/q1/rs123.md
    varchat/rs123.md

Output layout under --out-root:
  out_root/
    q1/rs123.md
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import json5
from dotenv import load_dotenv

from common.open_ai_azure import OpenAIAzure

# repo_root/src must be on PYTHONPATH
from common.utils import load_rs_gene_data, save_answer_to_markdown, get_file_content


class AnswerQualityEvaluator(OpenAIAzure):
    def __init__(self) -> None:
        super().__init__()

    def query_with_prompt(self, prompt: str) -> Optional[str]:
        # The parent expects (prompt, question). We keep question empty because
        # the prompt already contains the question and evidence.
        return super().query_azure_openai(prompt, "")

    def generate_prompt(
        self,
        prompt_path: Path,
        question: str,
        gene_symbols: str,
        chat_togovar_md: str,
        chat_gpt_md: str,
        varchat_md: str,
    ) -> str:
        """Render the evaluation prompt template."""
        prompt_template = prompt_path.read_text(encoding="utf-8")
        return prompt_template.format(
            chat_togovar=chat_togovar_md,
            chat_gpt=chat_gpt_md,
            varchat=varchat_md,
            question=question,
            gene_symbols=gene_symbols,
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run LLM-based answer quality evaluation (AQE).")
    ap.add_argument("--questions", required=True, help="Path to questions.json (JSON5 supported).")
    ap.add_argument("--rs-gene", required=True, help="Path to rs list file (e.g., pubtator3/rs_30_with_gene_symbol.txt).")
    ap.add_argument(
        "--answers-root",
        required=True,
        help="Root directory containing generated answers (chat_togovar/, gpt-4o/, varchat/).",
    )
    ap.add_argument("--prompt", required=True, help="Prompt template path (e.g., evaluation/gpt-4o/prompt.md).")
    ap.add_argument("--out-root", required=True, help="Output root directory (e.g., evaluation/gpt-4o).")
    ap.add_argument(
        "--rs-limit",
        type=int,
        default=0,
        help="Limit the number of rs entries for a quick test (0 = no limit).",
    )
    ap.add_argument("--sleep", type=float, default=0.0, help="Optional sleep seconds between requests.")
    ap.add_argument("--dotenv", default=".env", help="Path to .env file (default: .env).")
    ap.add_argument("--dry-run", action="store_true", help="Do not call Azure OpenAI; only print planned file paths.")
    return ap.parse_args()


def build_answer_paths(answers_root: Path, question_no: str, rs: str) -> tuple[Path, Path, Path]:
    """Return (chat_togovar_md, chat_gpt_md, varchat_md) paths."""
    chat_togovar = answers_root / "chat_togovar" / question_no / f"{rs}.md"
    chat_gpt = answers_root / "gpt-4o" / question_no / f"{rs}.md"
    varchat = answers_root / "varchat" / f"{rs}.md"
    return chat_togovar, chat_gpt, varchat


def main() -> None:
    args = parse_args()
    load_dotenv(args.dotenv)

    questions_path = Path(args.questions)
    rs_gene_path = Path(args.rs_gene)
    answers_root = Path(args.answers_root)
    prompt_path = Path(args.prompt)
    out_root = Path(args.out_root)

    print(f"[INFO] Questions: {questions_path}")
    print(f"[INFO] RS list: {rs_gene_path}")
    print(f"[INFO] Answers root: {answers_root}")
    print(f"[INFO] Prompt: {prompt_path}")
    print(f"[INFO] Output root: {out_root}")

    with questions_path.open("r", encoding="utf-8") as f:
        questions = json5.load(f)

    rs_gene_list = load_rs_gene_data(str(rs_gene_path))
    if args.rs_limit and args.rs_limit > 0:
        rs_gene_list = rs_gene_list[: args.rs_limit]
        print(f"[INFO] rs-limit applied: {len(rs_gene_list)} entries")

    client = AnswerQualityEvaluator()

    for question_no, question_template in questions.items():
        print(f"[INFO] Processing question {question_no}: {question_template}")
        for entry in rs_gene_list:
            rs = entry["rs_id"]
            gene_symbols = ", ".join(entry.get("gene_symbol", []))
            question_text = question_template.format(rs=rs)

            chat_togovar_path, chat_gpt_path, varchat_path = build_answer_paths(answers_root, question_no, rs)

            if args.dry_run:
                print(f"[DRY] {question_no} {rs}: {chat_togovar_path} | {chat_gpt_path} | {varchat_path}")
                continue

            chat_togovar_md = get_file_content(str(chat_togovar_path))
            chat_gpt_md = get_file_content(str(chat_gpt_path))
            varchat_md = get_file_content(str(varchat_path))

            prompt = client.generate_prompt(
                prompt_path=prompt_path,
                question=question_text,
                gene_symbols=gene_symbols,
                chat_togovar_md=chat_togovar_md,
                chat_gpt_md=chat_gpt_md,
                varchat_md=varchat_md,
            )

            print(f"[INFO] Evaluating: {question_no} {rs}")
            resp = client.query_with_prompt(prompt)
            if resp:
                out_path = out_root / question_no / f"{rs}.md"
                save_answer_to_markdown(str(out_path), resp)
                print("[INFO] done.")
            else:
                print("[WARN] No response from Azure OpenAI")

            if args.sleep and args.sleep > 0:
                import time
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()