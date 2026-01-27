#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Scan a directory of evaluation JSON files and report missing/invalid fields.

Usage:
  python scan_eval_json.py /path/to/json_dir
  python scan_eval_json.py /path/to/json_dir --glob "*.json"
  python scan_eval_json.py /path/to/json_dir --recursive
  python scan_eval_json.py /path/to/json_dir --fail-fast
  python scan_eval_json.py /path/to/json_dir --out report.tsv

Exit codes:
  0: no issues
  2: issues found (missing/invalid fields or parse errors)
  1: execution error (bad args, no files, etc.)
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple


# Required keys at the top-level of each JSON file
REQUIRED_TOP_KEYS = ["qY", "rsXXXX", "evaluation"]

# Expected model blocks under "evaluation"
MODELS = ["ChatTogoVar", "GPT4o", "VarChat"]

# Expected metric blocks under each model
METRICS = [
    "Accuracy",
    "Completeness",
    "Logical Consistency",
    "Clarity and Conciseness",
    "Evidence Support",
]

# Required keys inside each metric block
REQUIRED_METRIC_KEYS = ["score"]


def is_missing_value(v: Any) -> bool:
    """Return True if the value should be treated as missing."""
    if v is None:
        return True
    if isinstance(v, str) and v.strip() == "":
        return True
    if isinstance(v, (list, tuple, set)) and len(v) == 0:
        return True
    if isinstance(v, dict) and len(v) == 0:
        return True
    return False


def validate_score(score: Any) -> Tuple[bool, str]:
    """Validate that score is int-convertible and in the range 1..9."""
    if is_missing_value(score):
        return False, "score is missing/empty"
    try:
        n = int(str(score).strip())
    except Exception:
        return False, f"score is not int-convertible: {score!r}"
    if not (0 <= n <= 10):
        return False, f"score out of range (0..10): {n}"
    return True, ""


def check_payload(obj: Dict[str, Any]) -> List[str]:
    """Check one JSON object and return a list of issue strings."""
    issues: List[str] = []

    # Check required top-level keys
    for k in REQUIRED_TOP_KEYS:
        if k not in obj or is_missing_value(obj.get(k)):
            issues.append(f"[MISSING] top-level key/value: {k}")

    # Validate "evaluation"
    ev = obj.get("evaluation", {})
    if not isinstance(ev, dict) or is_missing_value(ev):
        issues.append("[MISSING] evaluation is missing/empty or not a dict")
        return issues

    # Validate each model and its metrics
    for m in MODELS:
        if m not in ev:
            issues.append(f"[MISSING] evaluation.{m} (model block not found)")
            continue

        mblk = ev.get(m)
        if not isinstance(mblk, dict) or is_missing_value(mblk):
            issues.append(f"[MISSING] evaluation.{m} is empty or not a dict")
            continue

        for metric in METRICS:
            if metric not in mblk:
                issues.append(f"[MISSING] evaluation.{m}.{metric} (metric block not found)")
                continue

            metblk = mblk.get(metric)
            if not isinstance(metblk, dict) or is_missing_value(metblk):
                issues.append(f"[MISSING] evaluation.{m}.{metric} is empty or not a dict")
                continue

            for rk in REQUIRED_METRIC_KEYS:
                if rk not in metblk or is_missing_value(metblk.get(rk)):
                    issues.append(f"[MISSING] evaluation.{m}.{metric}.{rk}")
                else:
                    if rk == "score":
                        ok, msg = validate_score(metblk.get("score"))
                        if not ok:
                            issues.append(f"[INVALID] evaluation.{m}.{metric}.score: {msg}")

    return issues


def iter_files(root_dir: str, pattern: str, recursive: bool) -> List[str]:
    """Collect matching files under root_dir."""
    root_dir = os.path.abspath(root_dir)
    if recursive:
        # Use globstar to match nested files
        g = os.path.join(root_dir, "**", pattern)
        return sorted(glob.glob(g, recursive=True))
    else:
        g = os.path.join(root_dir, pattern)
        return sorted(glob.glob(g))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="Directory containing JSON files")
    ap.add_argument("--glob", default="*.json", help='File pattern (default: "*.json")')
    ap.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    ap.add_argument("--fail-fast", action="store_true", help="Stop at the first detected issue")
    ap.add_argument("--out", default="", help="Write TSV report to this path (default: stdout)")
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"ERROR: not a directory: {args.dir}", file=sys.stderr)
        return 1

    files = iter_files(args.dir, args.glob, args.recursive)
    if not files:
        print(
            f"ERROR: no files matched: dir={args.dir} glob={args.glob} recursive={args.recursive}",
            file=sys.stderr,
        )
        return 1

    # Collected report rows: (file path, issue string)
    rows: List[Tuple[str, str]] = []

    n_ok = 0
    n_ng = 0
    n_parse_err = 0

    for fp in files:
        # Parse JSON
        try:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception as e:
            n_parse_err += 1
            rows.append((fp, f"[PARSE_ERROR] {type(e).__name__}: {e}"))
            if args.fail_fast:
                break
            continue

        # Root must be an object/dict
        if not isinstance(obj, dict):
            n_ng += 1
            rows.append((fp, "[INVALID] root is not an object/dict"))
            if args.fail_fast:
                break
            continue

        # Check required fields
        issues = check_payload(obj)
        if issues:
            n_ng += 1
            for it in issues:
                rows.append((fp, it))
            if args.fail_fast:
                break
        else:
            n_ok += 1

    # Emit TSV report (file, issue)
    header = "file\tissue\n"
    out_lines = [header] + [f"{os.path.basename(f)}\t{i}\n" for f, i in rows]

    if args.out:
        out_path = os.path.abspath(args.out)
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as w:
            w.writelines(out_lines)
    else:
        sys.stdout.writelines(out_lines)

    # Print a short summary to stderr
    print(
        f"SUMMARY: files={len(files)} ok={n_ok} ng={n_ng} parse_error={n_parse_err}",
        file=sys.stderr,
    )

    # Exit status
    if n_ng > 0 or n_parse_err > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
