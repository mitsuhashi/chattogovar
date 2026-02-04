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

Optional (for sensitivity analysis):
  --manual_primary   : primary rater manual JSON
  --manual_secondary : secondary rater manual JSON

Behavior:
  - Main table uses:
      Manual (from --manual; typically human-mean) + LLM
  - Supplementary table:
      * Default (no primary/secondary): one combined table for Manual + LLM (same as before).
      * If primary & secondary are provided: outputs 3 blocks (Manual mean / Manual primary / Manual secondary).
        Holm adjustment is applied WITHIN each block (recommended for sensitivity analysis).
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


def fmt_diff(x: float, digits: int = 2) -> str:
    if not np.isfinite(x):
        return ""
    return f"{x:.{digits}f}"


# ----------------------------
# Parsing into long format
# ----------------------------
def parse_records_to_long_df(records: List[Dict[str, Any]], eval_type: str) -> pd.DataFrame:
    """
    Returns long df: columns = [eval_type, ID, method, metric, score]
    eval_type: e.g., "Manual", "LLM", "Manual (mean)", ...
    ID rule:
      - Manual*: use rec["ID"]
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

            method, metric = k.split("_", 1)
            if method not in CANON_METHODS:
                continue
            if metric == "Total":
                continue

            rows.append((eval_type, qid, method, metric, to_float(v)))

    df = pd.DataFrame(rows, columns=["eval_type", "ID", "method", "metric", "score"])
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
        return (0.0, 1.0, int(n))

    stat, p = wilcoxon(x, y, alternative="two-sided", mode="auto")
    return (float(stat), float(p), int(n))


def pairwise_diff_stats(
    df_long: pd.DataFrame, eval_type: str, metric: str, a: str, b: str
) -> Tuple[float, float, int]:
    """
    Paired difference summary for a vs b within one (eval_type, metric).
    Returns (mean_diff, median_diff, n_pairs). Includes zero differences.
    diff := a - b
    """
    sub = df_long[(df_long["eval_type"] == eval_type) & (df_long["metric"] == metric)]
    piv = sub.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=[a, b], how="any")
    n = len(piv)
    if n < 1:
        return (np.nan, np.nan, 0)

    diff = (piv[a] - piv[b]).to_numpy(dtype=float)
    return (float(np.mean(diff)), float(np.median(diff)), int(n))


# ----------------------------
# Aggregation for Table 4 (Main)
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
    manual_label: str,
) -> pd.DataFrame:
    # Main table shows Manual(mean) and LLM
    eval_types = [manual_label, "LLM"]
    metrics = sorted(df_long["metric"].unique().tolist(), key=metric_sort_key)

    out_rows = []
    for et in eval_types:
        for met in metrics:
            row: Dict[str, Any] = {"Evaluation": et, "Metric": met}

            for m in CANON_METHODS:
                mean, lo, hi, _n = compute_mean_ci_block(df_long, et, met, m, n_boot, seed, alpha)
                row[m] = fmt_mean_ci(mean, lo, hi, digits=2)

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

    df = pd.DataFrame(
        out_rows,
        columns=["Evaluation", "Metric"] + CANON_METHODS + ["Friedman p", "Kendall's W", "n (paired)"],
    )
    return df


# ----------------------------
# Supplementary: pairwise Wilcoxon
# ----------------------------
def build_pairwise_table_for_evaltypes(
    df_long: pd.DataFrame,
    eval_types: List[str],
    holm_within_eval: bool,
) -> pd.DataFrame:
    metrics = sorted(df_long["metric"].unique().tolist(), key=metric_sort_key)
    comparisons = [
        ("ChatTogoVar", "GPT-4o"),
        ("ChatTogoVar", "VarChat"),
        ("GPT-4o", "VarChat"),
    ]

    rows: List[Dict[str, Any]] = []

    if holm_within_eval:
        # Apply Holm separately within each eval_type (recommended for sensitivity analysis blocks)
        for et in eval_types:
            block_rows_idx: List[int] = []
            block_pvals: List[float] = []

            for met in metrics:
                for a, b in comparisons:
                    mean_d, med_d, n2 = pairwise_diff_stats(df_long, et, met, a, b)
                    stat, p, n = pairwise_wilcoxon(df_long, et, met, a, b)
                    n_disp = n2 if n2 > 0 else n

                    rows.append(
                        {
                            "Evaluation": et,
                            "Metric": met,
                            "Comparison": f"{a} vs {b}",
                            "n (paired)": n_disp,
                            "Mean diff": fmt_diff(mean_d, digits=2),
                            "Median diff": fmt_diff(med_d, digits=2),
                            "Wilcoxon W": "" if not np.isfinite(stat) else f"{stat:.1f}",
                            "p": "" if not np.isfinite(p) else p,
                        }
                    )
                    if np.isfinite(p):
                        block_pvals.append(float(p))
                        block_rows_idx.append(len(rows) - 1)

            # Holm adjustment within this block
            if block_pvals:
                p_adj = holm_adjust(block_pvals)
                for adj, idx in zip(p_adj, block_rows_idx):
                    rows[idx]["p (Holm)"] = fmt_p(adj)

    else:
        # Original behavior: Holm across all rows across eval_types
        pvals: List[float] = []
        idx_map: List[int] = []

        for et in eval_types:
            for met in metrics:
                for a, b in comparisons:
                    mean_d, med_d, n2 = pairwise_diff_stats(df_long, et, met, a, b)
                    stat, p, n = pairwise_wilcoxon(df_long, et, met, a, b)
                    n_disp = n2 if n2 > 0 else n

                    rows.append(
                        {
                            "Evaluation": et,
                            "Metric": met,
                            "Comparison": f"{a} vs {b}",
                            "n (paired)": n_disp,
                            "Mean diff": fmt_diff(mean_d, digits=2),
                            "Median diff": fmt_diff(med_d, digits=2),
                            "Wilcoxon W": "" if not np.isfinite(stat) else f"{stat:.1f}",
                            "p": "" if not np.isfinite(p) else p,
                        }
                    )
                    if np.isfinite(p):
                        pvals.append(float(p))
                        idx_map.append(len(rows) - 1)

        if pvals:
            p_adj = holm_adjust(pvals)
            for adj, idx in zip(p_adj, idx_map):
                rows[idx]["p (Holm)"] = fmt_p(adj)

    # Fill missing adjusted p cells + format raw p
    for r in rows:
        if "p (Holm)" not in r:
            r["p (Holm)"] = ""
        if isinstance(r["p"], float):
            r["p"] = fmt_p(r["p"])

    df = pd.DataFrame(
        rows,
        columns=[
            "Evaluation",
            "Metric",
            "Comparison",
            "n (paired)",
            "Mean diff",
            "Median diff",
            "Wilcoxon W",
            "p",
            "p (Holm)",
        ],
    )
    return df


def df_to_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def split_eval_blocks_markdown(df: pd.DataFrame, title: str) -> str:
    """
    Split one pairwise df (with Evaluation column) into multiple markdown tables per Evaluation.
    Keeps the same columns; each block has a heading.
    """
    parts: List[str] = [title, "\n\n"]
    for et, sub in df.groupby("Evaluation", sort=False):
        parts.append(f"## {et}\n\n")
        parts.append(df_to_markdown(sub) + "\n\n")
    return "".join(parts)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Table 4 (main + supplementary) as Markdown.")
    ap.add_argument("--manual", required=True, help="Manual evaluation JSON path (list of dicts with 'ID').")
    ap.add_argument("--llm", required=True, help="LLM evaluation JSON path (list of dicts; ID = index+1).")

    # Sensitivity analysis (optional)
    ap.add_argument("--manual_primary", default=None, help="Primary rater manual JSON path (optional).")
    ap.add_argument("--manual_secondary", default=None, help="Secondary rater manual JSON path (optional).")

    ap.add_argument("--out_main", default="table4.md", help="Output path for main table markdown.")
    ap.add_argument("--out_supp", default="table4_supplementary.md", help="Output path for supplementary markdown.")
    ap.add_argument("--n_boot", type=int, default=5000, help="Bootstrap iterations for mean CI.")
    ap.add_argument("--seed", type=int, default=1, help="Random seed for bootstrap.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for (1-alpha) CI. default=0.05 => 95% CI")

    # Optional: include LLM in supplementary even in sensitivity mode
    ap.add_argument(
        "--supp_include_llm",
        action="store_true",
        help="Include LLM in supplementary table even when primary/secondary are provided (default: False).",
    )
    args = ap.parse_args()

    manual = load_json(args.manual)
    llm = load_json(args.llm)

    if not isinstance(manual, list) or not isinstance(llm, list):
        raise ValueError("Both input JSON files must be arrays (lists) of records.")

    # Labels
    manual_mean_label = "Manual (mean)"

    df_manual_mean = parse_records_to_long_df(manual, manual_mean_label)
    df_llm = parse_records_to_long_df(llm, "LLM")

    df_parts = [df_manual_mean, df_llm]

    # Optional primary/secondary
    have_raters = bool(args.manual_primary) and bool(args.manual_secondary)
    if have_raters:
        manual_p = load_json(args.manual_primary)
        manual_s = load_json(args.manual_secondary)
        if not isinstance(manual_p, list) or not isinstance(manual_s, list):
            raise ValueError("--manual_primary/--manual_secondary must be JSON arrays (lists).")
        df_parts.append(parse_records_to_long_df(manual_p, "Manual (primary)"))
        df_parts.append(parse_records_to_long_df(manual_s, "Manual (secondary)"))

    df_long = pd.concat(df_parts, ignore_index=True)

    # Main Table (Friedman)
    df_main = build_main_table(df_long, n_boot=args.n_boot, seed=args.seed, alpha=args.alpha, manual_label=manual_mean_label)
    md_main = "# Table 4\n\n" + df_to_markdown(df_main) + "\n"
    save_text(args.out_main, md_main)

    # Supplementary (Wilcoxon)
    if have_raters:
        eval_types = [manual_mean_label, "Manual (primary)", "Manual (secondary)"]
        if args.supp_include_llm:
            eval_types = eval_types + ["LLM"]

        df_pair = build_pairwise_table_for_evaltypes(df_long, eval_types=eval_types, holm_within_eval=True)
        header = "# Table 4 (Supplementary)\n\n"
        header += "_Pairwise Wilcoxon tests with Holm adjustment applied within each evaluation block._\n\n"
        md_supp = split_eval_blocks_markdown(df_pair, header)
        save_text(args.out_supp, md_supp)
    else:
        # Original behavior
        df_pair = build_pairwise_table_for_evaltypes(df_long, eval_types=[manual_mean_label, "LLM"], holm_within_eval=False)
        md_supp = "# Table 4 (Supplementary)\n\n" + df_to_markdown(df_pair) + "\n"
        save_text(args.out_supp, md_supp)


if __name__ == "__main__":
    main()