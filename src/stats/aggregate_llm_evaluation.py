#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_llm_evaluation.py

Aggregate GPT-4o evaluation markdown files into a single JSON (like aggregate_aqes.json),
in a CLI-argument style similar to the human aggregation script.

Input directory structure (default expectation):
  <in_dir>/<qNo>/*.md
e.g.
  evaluation/gpt-4o/q1/rs794726784.md

Each markdown is assumed to contain patterns like:
  Best Answer: ChatTogoVar
  Total Score for ChatTogoVar: 46/50
  Total Score for GPT-4o: 30/50
  Total Score for VarChat: 40/50
and per-answer blocks like:
  ## Answer ChatTogoVar
  - Accuracy Score: 10/10
  ...

Output JSON record example:
{
  "QuestionNumber": "q1",
  "Question": "Tell me the basic information about rs794726784.",
  "Filename_Text": "rs794726784.md",
  "BestAnswer": "ChatTogoVar",
  "ChatTogoVar": 46,
  "GPT-4o": 30,
  "VarChat": 40,
  "ChatTogoVar_Accuracy": 10,
  ...
}

Notes:
- Only JSON is generated (no Excel).
- Question templates are loaded from questions.json (JSON5 allowed).
- Model labels are standardized to: ChatTogoVar, GPT-4o, VarChat.
"""

from __future__ import annotations

import argparse
import json5
import os
import pandas as pd
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# 設定

models = ["ChatTogoVar", "GPT-4o", "VarChat"]
categories = ["Accuracy", "Completeness", "Logical Consistency", "Clarity and Conciseness", "Evidence Support"]

score_patterns = {
    "BestAnswer": r"Best Answer: (.+)",
    "ChatTogoVar": r"Total Score for ChatTogoVar: (\d+)/\d+",
    "GPT-4o": r"Total Score for GPT-4o: (\d+)/\d+",
    "VarChat": r"Total Score for VarChat: (\d+)/\d+"
}

def load_questions(path):
    with open(path, "r") as f:
        return json5.load(f)

def parse_markdown_file(filepath, question_no, question_template):
    rs = os.path.basename(filepath).replace(".md", "")
    with open(filepath, encoding='utf-8') as f:
        content = f.read()

    record = {
        "QuestionNumber": question_no,
        "Question": question_template.format(rs=rs),
        "Filename_Text": f"{rs}.md"
    }

    for key, pattern in score_patterns.items():
        match = re.search(pattern, content)
        record[key] = match.group(1) if match else None

    for model in models:
        block_pattern = rf"## Answer {re.escape(model)}\n(.*?)(?:\n## Answer |\Z)"
        match = re.search(block_pattern, content, re.DOTALL)
        if match:
            block = match.group(1)
            for cat in categories:
                cat_pattern = rf"- {cat} Score: (\d+)/10"
                cat_match = re.search(cat_pattern, block)
                record[f"{model}_{cat}"] = int(cat_match.group(1)) if cat_match else None
        else:
            for cat in categories:
                record[f"{model}_{cat}"] = None
    return record


def calculate_category_averages(df):
    records = []
    for criterion in categories:
        record = {"Criterion": criterion}
        for model in models:
            col = f"{model}_{criterion}"
            if col in df.columns:
                record[model] = df[col].mean()
        records.append(record)
    return pd.DataFrame(records)


def export_to_json(df, path):
    df.to_json(path, orient="records", force_ascii=False, indent=4)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        required=True,
        help="Input directory of GPT-4o evaluation markdowns (contains q1/, q2/, ...)",
    )
    ap.add_argument(
        "--questions",
        required=True,
        help="questions.json (JSON5 allowed): {\"q1\": \"...{rs}...\", ...}",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path (e.g., evaluation/gpt-4o/aggregate_aqes.json)",
    )
    args = ap.parse_args()

    directory_path = Path(args.in_dir)
    output_json = Path(args.out)

    questions = load_questions(args.questions)


    df = pd.DataFrame()

    for q_no, template in questions.items():
        q_dir = os.path.join(directory_path, q_no)
        if not os.path.isdir(q_dir):
            continue
        for fname in os.listdir(q_dir):
            if fname.endswith(".md"):
                filepath = os.path.join(q_dir, fname)
                rec = parse_markdown_file(filepath, q_no, template)
                df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)

    for col in df.columns:
        if any(model in col for model in models):
            df[col] = pd.to_numeric(df[col], errors='coerce').astype("Int64")


    export_to_json(df, output_json)

    print(f"{output_json} was generated with {len(df)} records.")

if __name__ == "__main__":
    main()