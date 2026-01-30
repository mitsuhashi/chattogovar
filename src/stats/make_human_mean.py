#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make averaged aggregate_human.json from two raters' aggregate_human.json files.

- Merge records by (ID, QuestionNumber, rsID) by default.
- Average all numeric score fields.
- Recompute *_Total as the sum of averaged 5 metrics to avoid inconsistencies.
- Recompute BestAnswer from averaged totals; ties -> "Tie".
"""

import argparse
import json
import math
from typing import Any, Dict, List, Tuple


MODELS = ["ChatTogoVar", "GPT-4o", "VarChat"]
METRICS = [
    "Accuracy",
    "Clarity and Conciseness",
    "Completeness",
    "Evidence Support",
    "Logical Consistency",
]

KEY_FIELDS_DEFAULT = ["ID", "QuestionNumber", "rsID"]


def is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def key_of(rec: Dict[str, Any], key_fields: List[str]) -> Tuple[Any, ...]:
    return tuple(rec.get(k) for k in key_fields)


def avg(a: Any, b: Any) -> Any:
    if is_number(a) and is_number(b):
        return (float(a) + float(b)) / 2.0
    return a if a == b else a  # keep primary if mismatch (non-numeric)


def maybe_int(x: float) -> Any:
    # Convert 12.0 -> 12, keep 12.5 -> 12.5
    if abs(x - round(x)) < 1e-9:
        return int(round(x))
    return x


def recompute_totals_and_best(rec: Dict[str, Any]) -> None:
    # Recompute totals from averaged metrics
    totals = {}
    for m in MODELS:
        s = 0.0
        ok = True
        for metric in METRICS:
            k = f"{m}_{metric}"
            v = rec.get(k)
            if not is_number(v):
                ok = False
                break
            s += float(v)
        if ok:
            rec[f"{m}_Total"] = maybe_int(s)
            totals[m] = s

    # Recompute BestAnswer (tie allowed)
    if len(totals) == len(MODELS):
        maxv = max(totals.values())
        winners = [m for m, v in totals.items() if abs(v - maxv) < 1e-9]
        rec["BestAnswer"] = winners[0] if len(winners) == 1 else "Tie"


def merge_records(p: Dict[str, Any], s: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # union of keys
    keys = set(p.keys()) | set(s.keys())
    for k in keys:
        pv = p.get(k)
        sv = s.get(k)

        # If one side is missing, keep the existing one
        if k not in p:
            out[k] = sv
            continue
        if k not in s:
            out[k] = pv
            continue

        # Average numeric, else keep primary unless exactly equal
        if is_number(pv) and is_number(sv):
            out[k] = maybe_int((float(pv) + float(sv)) / 2.0)
        else:
            out[k] = pv if pv == sv else pv

    # After merge, recompute totals and BestAnswer for consistency
    recompute_totals_and_best(out)
    return out


def load_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must be a JSON list.")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="primary rater aggregate_human.json")
    ap.add_argument("--secondary", required=True, help="secondary rater aggregate_human.json")
    ap.add_argument("--out", required=True, help="output averaged aggregate_human.json")
    ap.add_argument(
        "--key-fields",
        default=",".join(KEY_FIELDS_DEFAULT),
        help="comma-separated key fields to match records (default: ID,QuestionNumber,rsID)",
    )
    args = ap.parse_args()

    key_fields = [x.strip() for x in args.key_fields.split(",") if x.strip()]

    primary = load_json(args.primary)
    secondary = load_json(args.secondary)

    p_map = {key_of(r, key_fields): r for r in primary}
    s_map = {key_of(r, key_fields): r for r in secondary}

    missing_in_secondary = [k for k in p_map.keys() if k not in s_map]
    missing_in_primary = [k for k in s_map.keys() if k not in p_map]

    if missing_in_secondary or missing_in_primary:
        msg = []
        if missing_in_secondary:
            msg.append(f"Missing in secondary: {len(missing_in_secondary)}")
        if missing_in_primary:
            msg.append(f"Missing in primary: {len(missing_in_primary)}")
        raise SystemExit("Key mismatch between files. " + ", ".join(msg))

    merged: List[Dict[str, Any]] = []
    for k in sorted(p_map.keys(), key=lambda t: (t[0] if t else 0)):
        merged.append(merge_records(p_map[k], s_map[k]))

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {args.out} (n={len(merged)})")


if __name__ == "__main__":
    main()
