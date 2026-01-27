#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
table4.py
Generate:
  - table4.md                : main table (means + 95% CI, Friedman p, Kendall's W)
  - table4_supplementary.md  : supplementary table (pairwise Wilcoxon + Holm adjustment)

Inputs:
  --manual  : manual evaluation JSON (list of dicts; has "ID" field)
  --llm     : LLM-based evaluation JSON (list of dicts; ID is array index + 1)

Assumptions (as you stated):
  - Labels are consistent across human and LLM JSON:
      ChatTogoVar_Accuracy, GPT-4o_Accuracy, VarChat_Accuracy, ... etc
  - Totals exist as:
      Manual:  ChatTogoVar_Total / GPT-4o_Total / VarChat_Total
      LLM:     ChatTogoVar / GPT-4o / VarChat   (and sometimes *_Total; we accept both)

Dependencies:
  pip install pandas numpy scipy tabulate
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


CANON_METHODS = ["ChatTogoVar", "GPT-4o", "VarChat"]

# Order for nicer tables (we keep any additional metrics after these, if present)
PREFERRED_METRIC_ORDER = [
    "Total",
    "Accuracy",
    "Completeness",
    "Logical Consistency",
    "Clarity and Conciseness",
    "Evidence Support",
]


# ----------------------------
# IO / helpers
# ----------------------------
def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def is_number(x: Any) -> bool:
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return math.isfinite(float(x))
    if isinstance(x, str):
        try:
            v = float(x)
            return math.isfinite(v)
        except Exception:
            return False
    return False


def to_float(x: Any) -> float:
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return float(x)
    return float(str(x))


def bootstrap_mean_ci(
    x: np.ndarray,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 1,
) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for the mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    if n == 1:
        return (float(x[0]), float(x[0]))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = x[idx].mean(axis=1)
    lo = np.quantile(means, alpha / 2.0)
    hi = np.quantile(means, 1.0 - alpha / 2.0)
    return (float(lo), float(hi))


def holm_adjust(pvals: List[float]) -> List[float]:
    """
    Holm-Bonferroni adjustment.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.zeros(m, dtype=float)
    prev = 0.0
    for k, i in enumerate(order):
        val = (m - k) * pvals[i]
        val = min(1.0, max(val, prev))
        adj[i] = val
        prev = val
    return adj.tolist()


def fmt_mean_ci(mean: float, lo: float, hi: float, digits: int = 2) -> str:
    if not np.isfinite(mean):
        return ""
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{mean:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"
    return f"{mean:.{digits}f}"


def fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def fmt_w(w: float) -> str:
    if not np.isfinite(w):
        return ""
    return f"{w:.3f}"


# ----------------------------
# Parsing into long format
# ----------------------------
def parse_records_to_long_df(records: List[Dict[str, Any]], eval_type: str) -> pd.DataFrame:
    """
    Returns long df: columns = [eval_type, ID, method, metric, score]
    eval_type: "Manual" or "LLM"
    ID rule:
      - Manual: use rec["ID"]
      - LLM: use array index + 1
    """
    rows: List[Tuple[str, int, str, str, float]] = []

    for i, rec in enumerate(records):
        if eval_type.lower().startswith("man"):
            if "ID" not in rec:
                raise ValueError("Manual JSON record missing 'ID'.")
            qid = int(rec["ID"])
        else:
            qid = i + 1

        # 1) Totals: accept both styles
        #   Manual: ChatTogoVar_Total / GPT-4o_Total / VarChat_Total
        #   LLM: ChatTogoVar / GPT-4o / VarChat  (and possibly *_Total)
        for m in CANON_METHODS:
            if m in rec and is_number(rec[m]):
                rows.append((eval_type, qid, m, "Total", to_float(rec[m])))
            k_total = f"{m}_Total"
            if k_total in rec and is_number(rec[k_total]):
                rows.append((eval_type, qid, m, "Total", to_float(rec[k_total])))

        # 2) Per-metric keys like "{method}_{metric}"
        for k, v in rec.items():
            if not isinstance(k, str):
                continue
            if "_" not in k:
                continue
            if not is_number(v):
                continue

            # split at first underscore to preserve metric names containing underscores (unlikely)
            method, metric = k.split("_", 1)
            if method not in CANON_METHODS:
                continue
            if metric == "Total":
                # already handled above
                continue

            rows.append((eval_type, qid, method, metric, to_float(v)))

    df = pd.DataFrame(rows, columns=["eval_type", "ID", "method", "metric", "score"])

    # Normalize metric names if needed (optional; keeping as-is because your JSON uses spaces already)
    df["metric"] = df["metric"].astype(str)
    return df


def metric_sort_key(m: str) -> Tuple[int, str]:
    try:
        idx = PREFERRED_METRIC_ORDER.index(m)
        return (0, f"{idx:02d}")
    except ValueError:
        return (1, m.lower())


# ----------------------------
# Friedman / pairwise
# ----------------------------
@dataclass
class FriedmanResult:
    chi2: float
    p: float
    kendalls_w: float
    n_complete: int


def friedman_for_metric(df_long: pd.DataFrame, eval_type: str, metric: str) -> Optional[FriedmanResult]:
    sub = df_long[(df_long["eval_type"] == eval_type) & (df_long["metric"] == metric)]
    piv = sub.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=CANON_METHODS, how="any")
    n = len(piv)
    if n < 2:
        return None

    a = piv["ChatTogoVar"].to_numpy()
    b = piv["GPT-4o"].to_numpy()
    c = piv["VarChat"].to_numpy()

    chi2, p = friedmanchisquare(a, b, c)

    k = 3
    w = float(chi2) / float(n * (k - 1)) if n > 0 else np.nan
    return FriedmanResult(float(chi2), float(p), float(w), int(n))


def pairwise_wilcoxon(df_long: pd.DataFrame, eval_type: str, metric: str, a: str, b: str) -> Tuple[float, float, int]:
    sub = df_long[(df_long["eval_type"] == eval_type) & (df_long["metric"] == metric)]
    piv = sub.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=[a, b], how="any")
    n = len(piv)
    if n < 1:
        return (np.nan, np.nan, 0)

    x = piv[a].to_numpy()
    y = piv[b].to_numpy()
    diff = x - y
    if np.allclose(diff, 0):
        # all ties
        return (0.0, 1.0, int(n))

    stat, p = wilcoxon(x, y, alternative="two-sided", mode="auto")
    return (float(stat), float(p), int(n))


# ----------------------------
# Aggregation for Table 4
# ----------------------------
def compute_mean_ci_block(
    df_long: pd.DataFrame,
    eval_type: str,
    metric: str,
    method: str,
    n_boot: int,
    seed: int,
    alpha: float,
) -> Tuple[float, float, float, int]:
    sub = df_long[
        (df_long["eval_type"] == eval_type)
        & (df_long["metric"] == metric)
        & (df_long["method"] == method)
    ]["score"].to_numpy(dtype=float)

    sub = sub[np.isfinite(sub)]
    n = int(len(sub))
    if n == 0:
        return (np.nan, np.nan, np.nan, 0)

    mean = float(np.mean(sub))
    lo, hi = bootstrap_mean_ci(sub, alpha=alpha, n_boot=n_boot, seed=seed)
    return (mean, lo, hi, n)


def build_main_table(
    df_long: pd.DataFrame,
    n_boot: int,
    seed: int,
    alpha: float,
) -> pd.DataFrame:
    eval_types = ["Manual", "LLM"]
    metrics = sorted(df_long["metric"].unique().tolist(), key=metric_sort_key)

    out_rows = []
    for et in eval_types:
        for met in metrics:
            row: Dict[str, Any] = {"Evaluation": et, "Metric": met}

            # mean [CI]
            for m in CANON_METHODS:
                mean, lo, hi, _n = compute_mean_ci_block(df_long, et, met, m, n_boot, seed, alpha)
                row[m] = fmt_mean_ci(mean, lo, hi, digits=2)

            # Friedman + W (paired across methods)
            fr = friedman_for_metric(df_long, et, met)
            if fr is None:
                row["Friedman p"] = ""
                row["Kendall's W"] = ""
                row["n (paired)"] = ""
            else:
                row["Friedman p"] = fmt_p(fr.p)
                row["Kendall's W"] = fmt_w(fr.kendalls_w)
                row["n (paired)"] = fr.n_complete

            out_rows.append(row)

    df = pd.DataFrame(out_rows, columns=["Evaluation", "Metric"] + CANON_METHODS + ["Friedman p", "Kendall's W", "n (paired)"])
    return df


def build_pairwise_table(
    df_long: pd.DataFrame,
) -> pd.DataFrame:
    eval_types = ["Manual", "LLM"]
    metrics = sorted(df_long["metric"].unique().tolist(), key=metric_sort_key)

    comparisons = [
        ("ChatTogoVar", "GPT-4o"),
        ("ChatTogoVar", "VarChat"),
        ("GPT-4o", "VarChat"),
    ]

    rows = []
    pvals = []
    meta = []  # store (row_index) mapping for Holm later

    for et in eval_types:
        for met in metrics:
            for a, b in comparisons:
                stat, p, n = pairwise_wilcoxon(df_long, et, met, a, b)
                rows.append(
                    {
                        "Evaluation": et,
                        "Metric": met,
                        "Comparison": f"{a} vs {b}",
                        "n (paired)": n,
                        "Wilcoxon W": "" if not np.isfinite(stat) else f"{stat:.1f}",
                        "p": "" if not np.isfinite(p) else p,
                    }
                )
                if np.isfinite(p):
                    pvals.append(float(p))
                    meta.append(len(rows) - 1)

    # Holm adjustment only for rows with finite p
    p_adj = holm_adjust(pvals) if pvals else []
    for adj, idx in zip(p_adj, meta):
        rows[idx]["p (Holm)"] = fmt_p(adj)

    # Fill missing adjusted p cells
    for r in rows:
        if "p (Holm)" not in r:
            r["p (Holm)"] = ""

        # format raw p
        if isinstance(r["p"], float):
            r["p"] = fmt_p(r["p"])

    df = pd.DataFrame(rows, columns=["Evaluation", "Metric", "Comparison", "n (paired)", "Wilcoxon W", "p", "p (Holm)"])
    return df


def df_to_markdown(df: pd.DataFrame) -> str:
    # Requires tabulate installed; pandas will use it.
    return df.to_markdown(index=False)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Table 4 (main + supplementary) as Markdown.")
    ap.add_argument("--manual", required=True, help="Manual evaluation JSON path (list of dicts with 'ID').")
    ap.add_argument("--llm", required=True, help="LLM evaluation JSON path (list of dicts; ID = index+1).")
    ap.add_argument("--out_main", default="table4.md", help="Output path for main table markdown.")
    ap.add_argument("--out_supp", default="table4_supplementary.md", help="Output path for supplementary markdown.")
    ap.add_argument("--n_boot", type=int, default=5000, help="Bootstrap iterations for mean CI.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for bootstrap.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for (1-alpha) CI. default=0.05 => 95% CI")
    args = ap.parse_args()

    manual = load_json(args.manual)
    llm = load_json(args.llm)

    if not isinstance(manual, list) or not isinstance(llm, list):
        raise ValueError("Both input JSON files must be arrays (lists) of records.")

    df_manual = parse_records_to_long_df(manual, "Manual")
    df_llm = parse_records_to_long_df(llm, "LLM")
    df_long = pd.concat([df_manual, df_llm], ignore_index=True)

    # Main Table
    df_main = build_main_table(df_long, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)
    md_main = "# Table 4\n\n" + df_to_markdown(df_main) + "\n"
    save_text(args.out_main, md_main)

    # Supplementary (pairwise)
    df_pair = build_pairwise_table(df_long)
    md_supp = "# Table 4 (Supplementary)\n\n" + df_to_markdown(df_pair) + "\n"
    save_text(args.out_supp, md_supp)


if __name__ == "__main__":
    main()