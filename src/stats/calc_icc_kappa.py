#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calc_icc_kappa.py

Compute agreement between primary and secondary aggregated manual-evaluation JSONs
(assumed list of dicts like:
  {
    ID: 1,
    QuestionNumber: q1,
    "rsID": "rs704341",
    "ChatTogoVar_Total": 39,
    "GPT-4o_Total": 19,
    "VarChat_Total": 40,
    ...
  }
)

Targets:
  - Total score only (per method and pooled)

Metrics:
  - ICC(2,1) absolute agreement (two-way random effects, single measures)
  - Quadratic-weighted Cohen's kappa (integer totals)
  - |Δ| summaries and change counts

Usage:
  python calc_icc_kappa.py \
    --primary evaluation/human-primary/aggregate_human.json \
    --secondary evaluation/human-final/aggregate_human.json \
    --out audit/primary_audit_agreement.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


CANON_METHODS = ["ChatTogoVar", "GPT-4o", "VarChat"]


# ----------------------------
# IO / helpers
# ----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def save_json(path: str, obj: Any) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def is_int_like(x: Any) -> bool:
    if isinstance(x, bool):
        return False
    if isinstance(x, int):
        return True
    if isinstance(x, float) and x.is_integer():
        return True
    if isinstance(x, str):
        s = x.strip()
        return s.isdigit() or (s.startswith("-") and s[1:].isdigit())
    return False


def to_int(x: Any) -> int:
    if isinstance(x, int) and not isinstance(x, bool):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    return int(str(x).strip())


def record_key(rec: Dict[str, Any]) -> Optional[str]:
    """
    Primary key:
      - Use ID if present
      - Fallback: QuestionNumber + rsID
    """
    if "ID" in rec and is_int_like(rec["ID"]):
        return f"ID:{to_int(rec['ID'])}"
    q = str(rec.get("QuestionNumber") or "").strip()
    rs = str(rec.get("rsID") or "").strip()
    if q and rs:
        return f"QR:{q}:{rs}"
    return None


def extract_totals(rec: Dict[str, Any]) -> Dict[str, int]:
    """
    Read <Method>_Total values, return {method: total_int}.
    Assumes method naming already standardized: ChatTogoVar / GPT-4o / VarChat.
    """
    out: Dict[str, int] = {}
    for m in CANON_METHODS:
        k = f"{m}_Total"
        if k in rec and is_int_like(rec[k]):
            out[m] = to_int(rec[k])
    return out


# ----------------------------
# Agreement metrics
# ----------------------------
def icc2_1_absolute(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """
    ICC(2,1) absolute agreement for two raters (A,B),
    two-way random effects, single measures.

    pairs: list of (A, B)
    """
    n = len(pairs)
    k = 2
    if n < 2:
        return None

    X = [[float(a), float(b)] for a, b in pairs]
    grand = sum(sum(row) for row in X) / (n * k)

    mean_target = [sum(row) / k for row in X]
    mean_rater = [
        sum(X[i][0] for i in range(n)) / n,
        sum(X[i][1] for i in range(n)) / n,
    ]

    ss_target = k * sum((mt - grand) ** 2 for mt in mean_target)
    ss_rater = n * sum((mr - grand) ** 2 for mr in mean_rater)

    ss_error = 0.0
    for i in range(n):
        for j in range(k):
            ss_error += (X[i][j] - mean_target[i] - mean_rater[j] + grand) ** 2

    df_target = n - 1
    df_rater = k - 1
    df_error = (n - 1) * (k - 1)

    if df_target <= 0 or df_error <= 0:
        return None

    ms_target = ss_target / df_target
    ms_rater = ss_rater / df_rater if df_rater > 0 else 0.0
    ms_error = ss_error / df_error

    denom = (ms_target + (k - 1) * ms_error + (k * (ms_rater - ms_error) / n))
    if denom == 0:
        return None

    return (ms_target - ms_error) / denom


def weighted_kappa_quadratic(pairs: List[Tuple[int, int]]) -> Optional[float]:
    """
    Quadratic-weighted Cohen's kappa for integer ratings.
    """
    if len(pairs) < 2:
        return None

    a_vals = [a for a, _ in pairs]
    b_vals = [b for _, b in pairs]
    min_cat = min(min(a_vals), min(b_vals))
    max_cat = max(max(a_vals), max(b_vals))
    m = max_cat - min_cat + 1
    if m <= 1:
        return None

    O = [[0.0 for _ in range(m)] for _ in range(m)]
    for a, b in pairs:
        O[a - min_cat][b - min_cat] += 1.0
    N = float(len(pairs))

    row = [sum(O[i][j] for j in range(m)) for i in range(m)]
    col = [sum(O[i][j] for i in range(m)) for j in range(m)]
    E = [[(row[i] * col[j]) / N for j in range(m)] for i in range(m)]

    W = [[0.0 for _ in range(m)] for _ in range(m)]
    denom = float((m - 1) ** 2)
    for i in range(m):
        for j in range(m):
            W[i][j] = ((i - j) ** 2) / denom

    num = 0.0
    den = 0.0
    for i in range(m):
        for j in range(m):
            num += W[i][j] * O[i][j]
            den += W[i][j] * E[i][j]
    if den == 0:
        return None
    return 1.0 - (num / den)


def summarize_pairs_int(pairs: List[Tuple[int, int]]) -> Dict[str, Any]:
    if not pairs:
        return {
            "n": 0,
            "icc2_1": None,
            "kappa_quadratic": None,
            "mean_abs_delta": None,
            "median_abs_delta": None,
            "max_abs_delta": None,
        }

    deltas = [b - a for a, b in pairs]
    absd = sorted(abs(d) for d in deltas)
    n = len(pairs)

    icc = icc2_1_absolute([(float(a), float(b)) for a, b in pairs])
    kap = weighted_kappa_quadratic(pairs)

    mean_abs = sum(absd) / n
    med_abs = absd[n // 2] if n % 2 == 1 else (absd[n // 2 - 1] + absd[n // 2]) / 2
    max_abs = absd[-1]

    return {
        "n": n,
        "icc2_1": icc,
        "kappa_quadratic": kap,
        "mean_abs_delta": mean_abs,
        "median_abs_delta": med_abs,
        "max_abs_delta": max_abs,
    }


# ----------------------------
# Core
# ----------------------------
@dataclass
class ChangeCounter:
    n_questions_paired: int = 0
    n_questions_changed_any_method: int = 0
    n_cells_paired: int = 0
    n_cells_changed: int = 0


def compute(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> Dict[str, Any]:
    # index by key
    p_map: Dict[str, Dict[str, Any]] = {}
    s_map: Dict[str, Dict[str, Any]] = {}

    skipped_p = 0
    skipped_s = 0

    for rec in primary:
        if not isinstance(rec, dict):
            continue
        k = record_key(rec)
        if k is None:
            skipped_p += 1
            continue
        p_map[k] = rec

    for rec in secondary:
        if not isinstance(rec, dict):
            continue
        k = record_key(rec)
        if k is None:
            skipped_s += 1
            continue
        s_map[k] = rec

    keys = sorted(set(p_map.keys()) & set(s_map.keys()))
    missing_in_secondary = sorted(set(p_map.keys()) - set(s_map.keys()))
    missing_in_primary = sorted(set(s_map.keys()) - set(p_map.keys()))

    # pairs per method and pooled
    pairs_by_method: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    pairs_pooled: List[Tuple[int, int]] = []

    # change tracking
    cc = ChangeCounter()
    changed_questions: List[Dict[str, Any]] = []  # store small trace for changed items

    for k in keys:
        p_rec = p_map[k]
        s_rec = s_map[k]
        p_tot = extract_totals(p_rec)
        s_tot = extract_totals(s_rec)

        # require all three methods? For agreement, we pair per method if both exist.
        # Question-level paired count: count if at least one method is pairable.
        method_pairs_in_question = 0
        any_changed = False
        per_method_delta: Dict[str, int] = {}

        for m in CANON_METHODS:
            if m in p_tot and m in s_tot:
                a = p_tot[m]
                b = s_tot[m]
                pairs_by_method[m].append((a, b))
                pairs_pooled.append((a, b))
                method_pairs_in_question += 1
                cc.n_cells_paired += 1
                if a != b:
                    cc.n_cells_changed += 1
                    any_changed = True
                    per_method_delta[m] = b - a

        if method_pairs_in_question > 0:
            cc.n_questions_paired += 1
            if any_changed:
                cc.n_questions_changed_any_method += 1
                # keep compact trace
                changed_questions.append(
                    {
                        "key": k,
                        "QuestionNumber": p_rec.get("QuestionNumber", s_rec.get("QuestionNumber")),
                        "rsID": p_rec.get("rsID", s_rec.get("rsID")),
                        "deltas": per_method_delta,
                    }
                )

    out: Dict[str, Any] = {
        "counts": {
            "n_primary_records": len(primary) if isinstance(primary, list) else None,
            "n_secondary_records": len(secondary) if isinstance(secondary, list) else None,
            "n_skipped_primary_no_key": skipped_p,
            "n_skipped_secondary_no_key": skipped_s,
            "n_paired_questions": cc.n_questions_paired,
            "n_questions_changed_any_method": cc.n_questions_changed_any_method,
            "n_paired_cells_total": cc.n_cells_paired,
            "n_cells_changed": cc.n_cells_changed,
            "n_missing_in_secondary": len(missing_in_secondary),
            "n_missing_in_primary": len(missing_in_primary),
        },
        "agreement_total_score": {
            "pooled": summarize_pairs_int(pairs_pooled),
            "by_method": {m: summarize_pairs_int(pairs_by_method[m]) for m in CANON_METHODS},
        },
        # Keep traces small; adjust if you want full lists
        "examples_changed_questions": changed_questions[:50],
        "missing_keys": {
            "missing_in_secondary": missing_in_secondary[:200],
            "missing_in_primary": missing_in_primary[:200],
        },
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary", required=True, help="Primary aggregated JSON (list of dicts)")
    ap.add_argument("--secondary", required=True, help="Secondary aggregated JSON (list of dicts)")
    ap.add_argument("--out", required=True, help="Output JSON path")
    args = ap.parse_args()

    p_obj = load_json(args.primary)
    s_obj = load_json(args.secondary)

    if not isinstance(p_obj, list):
        raise SystemExit("--primary must be a JSON array (list of dicts).")
    if not isinstance(s_obj, list):
        raise SystemExit("--secondary must be a JSON array (list of dicts).")

    results = compute(p_obj, s_obj)
    save_json(args.out, results)
    print(f"Wrote: {args.out}")


if __name__ == "__main__":
    main()