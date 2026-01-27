#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
aggregate_human_eval_to_json.py

Aggregate per-question human evaluation JSON files into a single JSON (records list).

- Input files: evaluation_*.json under --in-dir
  Expected filename format:
    evaluation_<zero_based_index>_q<NUM>_rs<RSDIGITS>.json
  Example:
    evaluation_0_q1_rs704341.json

- Output: one JSON file (list of dicts) with fields ordered like:
    ID, QuestionNumber, rsID, BestAnswer,
    ChatTogoVar_Total, GPT-4o_Total, VarChat_Total,
    ChatTogoVar_<criteria...>, GPT-4o_<criteria...>, VarChat_<criteria...>

Notes:
- Input model key is assumed to include "GPT4o" (no hyphen) but output uses "GPT-4o".
- Reasons (reason_ja / reason_en) are ignored; only numeric scores are aggregated.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Optional


# ---- Config ----
CRITERIA_KEYS = [
    "Accuracy",
    "Clarity and Conciseness",
    "Completeness",
    "Evidence Support",
    "Logical Consistency",
]

# Input keys in evaluation JSON
INPUT_MODELS = ["ChatTogoVar", "GPT4o", "VarChat"]

# Output keys (rename GPT4o -> GPT-4o)
OUTPUT_MODEL_MAP = {
    "ChatTogoVar": "ChatTogoVar",
    "GPT4o": "GPT-4o",
    "VarChat": "VarChat",
}

FILENAME_RE = re.compile(r"^evaluation_(\d+)_q(\d+)_rs(.+)\.json$", re.IGNORECASE)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_int(x: Any, default: int = 0) -> int:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return default
        if isinstance(x, int):
            return int(x)
        if isinstance(x, float):
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if s == "":
                return default
            return int(float(s))
    except Exception:
        return default
    return default


def extract_score(model_block: Dict[str, Any], criterion: str) -> int:
    """
    model_block[criterion] can be dict with {"score": ...} or a raw number.
    """
    if not isinstance(model_block, dict):
        return 0
    v = model_block.get(criterion, None)
    if isinstance(v, dict):
        return safe_int(v.get("score", 0), 0)
    return safe_int(v, 0)


def compute_totals(evaluation_obj: Dict[str, Any]) -> Dict[str, int]:
    """
    Returns totals by INPUT model name.
    """
    totals: Dict[str, int] = {}
    ev = evaluation_obj.get("evaluation", {})
    if not isinstance(ev, dict):
        return {m: 0 for m in INPUT_MODELS}

    for m in INPUT_MODELS:
        mblock = ev.get(m, {})
        total = 0
        for c in CRITERIA_KEYS:
            total += extract_score(mblock, c)
        totals[m] = total
    return totals


def determine_best_answer(totals: Dict[str, int]) -> str:
    """
    If tie for max, return "Tie". Otherwise return output model name.
    """
    if not totals:
        return "Tie"
    maxv = max(totals.values())
    winners = [m for m, v in totals.items() if v == maxv]
    if len(winners) != 1:
        return "Tie"
    return OUTPUT_MODEL_MAP.get(winners[0], winners[0])


def build_record(
    eval_id: int,
    question_no: str,
    rs_id_full: str,
    data: Dict[str, Any],
) -> OrderedDict:
    """
    Build one output record with requested field order.
    """
    totals_in = compute_totals(data)
    best = determine_best_answer(totals_in)

    rec: OrderedDict = OrderedDict()
    rec["ID"] = eval_id
    rec["QuestionNumber"] = question_no
    rec["rsID"] = rs_id_full
    rec["BestAnswer"] = best

    # Totals first (in requested order)
    rec["ChatTogoVar_Total"] = totals_in.get("ChatTogoVar", 0)
    rec["GPT-4o_Total"] = totals_in.get("GPT4o", 0)
    rec["VarChat_Total"] = totals_in.get("VarChat", 0)

    # Criteria afterwards, grouped by model (matching your example style)
    ev = data.get("evaluation", {})
    if not isinstance(ev, dict):
        ev = {}

    for in_model in ["ChatTogoVar", "GPT4o", "VarChat"]:
        out_model = OUTPUT_MODEL_MAP[in_model]
        mblock = ev.get(in_model, {})
        for c in CRITERIA_KEYS:
            rec[f"{out_model}_{c}"] = extract_score(mblock, c)

    return rec


def aggregate_dir(in_dir: str) -> List[OrderedDict]:
    rows: List[OrderedDict] = []

    for fn in sorted(os.listdir(in_dir)):
        if not fn.lower().endswith(".json"):
            continue
        m = FILENAME_RE.match(fn)
        if not m:
            continue

        eval_id_raw, qnum, rs_id = m.groups()
        eval_id = int(eval_id_raw) + 1  # make 1-based
        question_no = f"q{int(qnum)}"
        rs_id_full = f"rs{rs_id}"

        path = os.path.join(in_dir, fn)
        data = load_json(path)

        row = build_record(eval_id, question_no, rs_id_full, data)
        rows.append(row)

    # sort by ID
    rows.sort(key=lambda r: int(r.get("ID", 0)))
    return rows


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate human evaluation JSON files into one JSON list."
    )
    ap.add_argument(
        "--in-dir",
        required=True,
        help="Directory containing evaluation_*.json (e.g., ./evaluation/human)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output JSON path (e.g., ./evaluation/human-primary/aggregate_human.json)",
    )
    args = ap.parse_args()

    in_dir = os.path.abspath(args.in_dir)
    out_path = os.path.abspath(args.out)

    rows = aggregate_dir(in_dir)

    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=4, ensure_ascii=False, separators=(",", ":"))

    print(f"[INFO] Wrote {len(rows)} records to: {out_path}")


if __name__ == "__main__":
    main()