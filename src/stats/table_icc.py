#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
calc_icc_kappa.py

Compute inter-rater agreement statistics directly from two aggregate_human.json files:
  - --primary    : evaluation/human-primary/aggregate_human.json
  - --secondary  : evaluation/human-secondary/aggregate_human.json
and write a supplementary Markdown table to --out.

What is computed (Total score only):
  - Pooled across methods (ChatTogoVar/GPT-4o/VarChat): n = N_items * 3
  - By method: n = N_items

Metrics per group:
  - ICC(2,1)  : Two-way random, absolute agreement, single measurement
  - ICC(2,2)  : Two-way random, absolute agreement, mean of 2 raters (derived from ICC(2,1))
  - Weighted kappa (quadratic weights)
  - |Δ| mean / median / max  (absolute differences between raters' total scores)

Dependencies:
  - pandas, numpy, tabulate (for DataFrame.to_markdown)
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

METHODS = ["ChatTogoVar", "GPT-4o", "VarChat"]
K_RATERS = 2


# ----------------------------
# IO
# ----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: str, text: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def is_number(x: Any) -> bool:
    try:
        if x is None:
            return False
        v = float(x)
        return math.isfinite(v)
    except Exception:
        return False


def to_float(x: Any) -> float:
    return float(x)


# ----------------------------
# Extract totals from aggregate_human.json
# ----------------------------
def build_id_map(records: List[Dict[str, Any]], label: str) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for rec in records:
        if "ID" not in rec:
            raise SystemExit(f"ERROR: {label} record missing 'ID'.")
        rid = int(rec["ID"])
        if rid in out:
            raise SystemExit(f"ERROR: duplicated ID={rid} in {label}.")
        out[rid] = rec
    return out


def get_total(rec: Dict[str, Any], method: str) -> float:
    """
    Expected keys in aggregate_human.json:
      - ChatTogoVar_Total
      - GPT-4o_Total
      - VarChat_Total
    (Also tolerates GPT4o_Total just in case.)
    """
    keys = [f"{method}_Total"]
    if method == "GPT-4o":
        keys += ["GPT4o_Total"]

    for k in keys:
        if k in rec and is_number(rec[k]):
            return to_float(rec[k])
    raise KeyError(f"Missing total for method={method} (expected {keys}) in record ID={rec.get('ID')}")


def align_totals(
    primary: List[Dict[str, Any]],
    secondary: List[Dict[str, Any]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]]:
    """
    Returns dict:
      method -> (x_primary, y_secondary, ids)
    where arrays are aligned by ID and length n_items.
    """
    pmap = build_id_map(primary, "primary")
    smap = build_id_map(secondary, "secondary")

    common_ids = sorted(set(pmap.keys()) & set(smap.keys()))
    if not common_ids:
        raise SystemExit("ERROR: no overlapping IDs between primary and secondary.")

    # Strictness: require same set of IDs
    missing_in_secondary = sorted(set(pmap.keys()) - set(smap.keys()))
    missing_in_primary = sorted(set(smap.keys()) - set(pmap.keys()))
    if missing_in_secondary or missing_in_primary:
        raise SystemExit(
            "ERROR: ID sets differ between primary and secondary.\n"
            f"  missing_in_secondary (count={len(missing_in_secondary)}): {missing_in_secondary[:10]}\n"
            f"  missing_in_primary   (count={len(missing_in_primary)}): {missing_in_primary[:10]}\n"
        )

    out: Dict[str, Tuple[np.ndarray, np.ndarray, List[int]]] = {}
    for m in METHODS:
        xs: List[float] = []
        ys: List[float] = []
        for rid in common_ids:
            xs.append(get_total(pmap[rid], m))
            ys.append(get_total(smap[rid], m))
        out[m] = (np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), common_ids)

    return out


# ----------------------------
# ICC(2,1)
# ----------------------------
def icc_2_1(y: np.ndarray) -> float:
    """
    ICC(2,1): two-way random effects, absolute agreement, single rater/measurement.
    Input y: shape (n_targets, k_raters) with k=2.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or y.shape[1] != K_RATERS:
        raise ValueError(f"y must be (n, {K_RATERS})")

    # Drop rows with non-finite
    mask = np.isfinite(y).all(axis=1)
    y = y[mask]
    n, k = y.shape
    if n < 2:
        return float("nan")

    mean_target = y.mean(axis=1, keepdims=True)  # (n,1)
    mean_rater = y.mean(axis=0, keepdims=True)   # (1,k)
    grand = y.mean()

    ss_target = k * np.sum((mean_target[:, 0] - grand) ** 2)
    ss_rater = n * np.sum((mean_rater[0, :] - grand) ** 2)
    ss_error = np.sum((y - mean_target - mean_rater + grand) ** 2)

    df_target = n - 1
    df_rater = k - 1
    df_error = (n - 1) * (k - 1)

    ms_target = ss_target / df_target if df_target > 0 else float("nan")
    ms_rater = ss_rater / df_rater if df_rater > 0 else float("nan")
    ms_error = ss_error / df_error if df_error > 0 else float("nan")

    denom = ms_target + (k - 1) * ms_error + (k * (ms_rater - ms_error) / n)
    if denom == 0 or not np.isfinite(denom):
        return float("nan")

    return float((ms_target - ms_error) / denom)


def icc_2_2_from_icc_2_1(r: float, k: int = 2) -> float:
    """
    ICC(2,k) from ICC(2,1), for mean of k raters.
      ICC(2,k) = (k * r) / (1 + (k - 1) * r)
    """
    if not np.isfinite(r):
        return float("nan")
    denom = 1.0 + (k - 1) * r
    if denom == 0:
        return float("nan")
    return float((k * r) / denom)


# ----------------------------
# Quadratic weighted kappa
# ----------------------------
def weighted_kappa_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Quadratic weighted kappa.
    Treats unique score values as ordered categories.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n == 0:
        return float("nan")

    # Use ordered unique categories from pooled observed values
    cats = np.array(sorted(set(x.tolist()) | set(y.tolist())))
    m = len(cats)
    if m <= 1:
        return 1.0

    idx = {v: i for i, v in enumerate(cats.tolist())}
    O = np.zeros((m, m), dtype=float)
    for a, b in zip(x, y):
        O[idx[a], idx[b]] += 1.0
    O = O / O.sum()

    px = O.sum(axis=1, keepdims=True)
    py = O.sum(axis=0, keepdims=True)
    E = px @ py

    # Disagreement weights (quadratic)
    i = np.arange(m).reshape(-1, 1)
    j = np.arange(m).reshape(1, -1)
    d = ((i - j) ** 2) / float((m - 1) ** 2)

    num = float(np.sum(d * O))
    den = float(np.sum(d * E))
    if den == 0:
        return 1.0
    return float(1.0 - (num / den))


# ----------------------------
# Formatting / table
# ----------------------------
def f3(v: Any) -> str:
    try:
        return f"{float(v):.3f}"
    except Exception:
        return ""


def f1(v: Any) -> str:
    try:
        return f"{float(v):.1f}"
    except Exception:
        return ""


def compute_block(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = int(len(x))

    if n == 0:
        return {
            "n": 0,
            "icc2_1": float("nan"),
            "icc2_2": float("nan"),
            "kappa_quadratic": float("nan"),
            "mean_abs_delta": float("nan"),
            "median_abs_delta": float("nan"),
            "max_abs_delta": float("nan"),
        }

    Y = np.column_stack([x, y])  # (n,2)
    r = icc_2_1(Y)
    r2 = icc_2_2_from_icc_2_1(r, k=2)
    x_int = np.rint(x).astype(int)
    y_int = np.rint(y).astype(int)
    kappa = weighted_kappa_quadratic(x_int, y_int)

    d = np.abs(x - y)
    return {
        "n": n,
        "icc2_1": r,
        "icc2_2": r2,
        "kappa_quadratic": kappa,
        "mean_abs_delta": float(np.mean(d)),
        "median_abs_delta": float(np.median(d)),
        "max_abs_delta": float(np.max(d)),
    }


def build_markdown_table(primary: List[Dict[str, Any]], secondary: List[Dict[str, Any]]) -> str:
    aligned = align_totals(primary, secondary)

    rows: List[Dict[str, Any]] = []

    # pooled = concatenate across methods (n = n_items * 3)
    xs_all = []
    ys_all = []
    for m in METHODS:
        x, y, _ids = aligned[m]
        xs_all.append(x)
        ys_all.append(y)
    x_pool = np.concatenate(xs_all)
    y_pool = np.concatenate(ys_all)
    pooled = compute_block(x_pool, y_pool)

    rows.append(
        {
            "Group": "Pooled (all models)",
            "n": pooled["n"],
            "ICC(2,1)": f3(pooled["icc2_1"]),
            "ICC(2,2)": f3(pooled["icc2_2"]),
            "Weighted κ (quadratic)": f3(pooled["kappa_quadratic"]),
            "abs(delta) mean": f3(pooled["mean_abs_delta"]),
            "abs(delta) median": f1(pooled["median_abs_delta"]),
            "abs(delta) max": int(pooled["max_abs_delta"]) if np.isfinite(pooled["max_abs_delta"]) else "",
        }
    )

    # by method
    for m in METHODS:
        x, y, _ids = aligned[m]
        blk = compute_block(x, y)
        rows.append(
            {
                "Group": m,
                "n": blk["n"],
                "ICC(2,1)": f3(blk["icc2_1"]),
                "ICC(2,2)": f3(blk["icc2_2"]),
                "Weighted κ (quadratic)": f3(blk["kappa_quadratic"]),
                "abs(delta) mean": f3(blk["mean_abs_delta"]),
                "abs(delta) median": f1(blk["median_abs_delta"]),
                "abs(delta) max": int(blk["max_abs_delta"]) if np.isfinite(blk["max_abs_delta"]) else "",
            }
        )

    df = pd.DataFrame(
        rows,
        columns=["Group", "n", "ICC(2,1)", "ICC(2,2)", "Weighted κ (quadratic)", "abs(delta) mean", "abs(delta) median", "abs(delta) max"],
    )

    md: List[str] = []
    md.append("# Supplementary Table X. Inter-rater agreement (Total score)\n")
    md.append(
        "ICC is reported as ICC(2,1) and ICC(2,2) (mean of two raters). "
        "Weighted κ uses quadratic weights. "
        "|Δ| is the absolute difference between two raters’ total scores per item.\n"
    )
    md.append(df.to_markdown(index=False))
    md.append("")
    return "\n".join(md)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Compute ICC and quadratic weighted kappa from two aggregate_human.json files.")
    ap.add_argument("--primary", required=True, help="Primary rater aggregate_human.json")
    ap.add_argument("--secondary", required=True, help="Secondary rater aggregate_human.json")
    ap.add_argument("--out", required=True, help="Output markdown path (supplementary table)")
    args = ap.parse_args()

    primary = load_json(args.primary)
    secondary = load_json(args.secondary)

    if not isinstance(primary, list) or not isinstance(secondary, list):
        raise SystemExit("ERROR: both --primary and --secondary must be JSON arrays (lists).")

    md = build_markdown_table(primary, secondary)
    save_text(args.out, md)


if __name__ == "__main__":
    main()