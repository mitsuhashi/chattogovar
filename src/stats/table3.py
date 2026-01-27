#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
table3.py
Generate Table 3 (counts of highest-scoring answers) for:
- Manual evaluation JSON (e.g., evaluation/human-primary/aggregate_human.json)
- LLM-based evaluation JSON (e.g., evaluation/gpt-4o/aggregate_aqes.json)

This version:
- DOES NOT use the JSON field "BestAnswer".
- Determines the winner by comparing TOTAL scores across ChatTogoVar / GPT-4o / VarChat.
- Adds a "Tie (no unique winner)" row so that totals sum to n (150 / 1,500).
- Optionally adds an "NA (missing totals)" row if any record lacks required totals.

Output:
- Markdown table written to --out.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


METHODS = ["ChatTogoVar", "GPT-4o", "VarChat"]


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        float(x)
        return True
    except Exception:
        return False


def get_total_score(row: Dict[str, Any], method: str) -> Optional[float]:
    """
    Manual JSON totals:
      ChatTogoVar_Total, GPT-4o_Total, VarChat_Total
    LLM JSON totals:
      ChatTogoVar, GPT-4o, VarChat
    Supports both (prefers *_Total if present).
    """
    keys: List[str]
    if method == "ChatTogoVar":
        keys = ["ChatTogoVar_Total", "ChatTogoVar"]
    elif method == "GPT-4o":
        keys = ["GPT-4o_Total", "GPT4o_Total", "GPT-4o", "GPT4o"]
    elif method == "VarChat":
        keys = ["VarChat_Total", "VarChat"]
    else:
        keys = [method]

    for k in keys:
        if k in row and is_number(row[k]):
            return float(row[k])
    return None


def determine_status(row: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """
    Returns (status, winner):
      status in {"winner", "tie", "na"}
      winner is one of METHODS when status == "winner", else None
    """
    scores: Dict[str, float] = {}
    for m in METHODS:
        v = get_total_score(row, m)
        if v is None:
            return ("na", None)
        scores[m] = v

    maxv = max(scores.values())
    tied = [m for m, v in scores.items() if v == maxv]

    if len(tied) == 1:
        return ("winner", tied[0])
    return ("tie", None)


def count_by_winner(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Counts:
      - per-method winners
      - tie
      - na (missing totals)
    """
    counts = {m: 0 for m in METHODS}
    counts["Tie"] = 0
    counts["NA"] = 0

    for r in records:
        status, winner = determine_status(r)
        if status == "winner" and winner is not None:
            counts[winner] += 1
        elif status == "tie":
            counts["Tie"] += 1
        else:
            counts["NA"] += 1

    return counts


def fmt_cell(k: int, denom: int) -> str:
    pct = (100.0 * k / denom) if denom > 0 else 0.0
    return f"{k:,} ({pct:.1f}%)"


def build_table3(manual_records: List[Dict[str, Any]], llm_records: List[Dict[str, Any]]) -> pd.DataFrame:
    cm = count_by_winner(manual_records)
    cl = count_by_winner(llm_records)

    n_manual = len(manual_records)
    n_llm = len(llm_records)

    rows: List[Dict[str, str]] = []
    for m in METHODS:
        rows.append(
            {
                "Model": m,
                "Manual Evaluation": fmt_cell(cm[m], n_manual),
                "LLM-Based Evaluation": fmt_cell(cl[m], n_llm),
            }
        )

    # Always show Tie row (to make totals sum to n)
    rows.append(
        {
            "Model": "Tie (no unique winner)",
            "Manual Evaluation": fmt_cell(cm["Tie"], n_manual),
            "LLM-Based Evaluation": fmt_cell(cl["Tie"], n_llm),
        }
    )

    # Show NA row only if present in either dataset
    if cm["NA"] > 0 or cl["NA"] > 0:
        rows.append(
            {
                "Model": "NA (missing totals)",
                "Manual Evaluation": fmt_cell(cm["NA"], n_manual),
                "LLM-Based Evaluation": fmt_cell(cl["NA"], n_llm),
            }
        )

    # Add Total row for clarity
    total_manual = sum(cm[m] for m in METHODS) + cm["Tie"] + cm["NA"]
    total_llm = sum(cl[m] for m in METHODS) + cl["Tie"] + cl["NA"]
    rows.append(
        {
            "Model": "Total",
            "Manual Evaluation": f"{total_manual:,} (100.0%)" if n_manual > 0 else "0 (0.0%)",
            "LLM-Based Evaluation": f"{total_llm:,} (100.0%)" if n_llm > 0 else "0 (0.0%)",
        }
    )

    return pd.DataFrame(rows, columns=["Model", "Manual Evaluation", "LLM-Based Evaluation"])


def save_markdown(df: pd.DataFrame, out_path: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    # pandas.to_markdown uses tabulate (assumed installed)
    md = df.to_markdown(index=False) + "\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True, help="Manual evaluation JSON (list of records)")
    ap.add_argument("--llm", required=True, help="LLM-based evaluation JSON (list of records)")
    ap.add_argument("--out", required=True, help="Output markdown path (e.g., evaluation/tables/table3.md)")
    args = ap.parse_args()

    manual = load_json(args.manual)
    llm = load_json(args.llm)

    if not isinstance(manual, list) or not isinstance(llm, list):
        raise SystemExit("ERROR: both inputs must be JSON arrays (lists).")

    df = build_table3(manual, llm)
    save_markdown(df, args.out)


if __name__ == "__main__":
    main()