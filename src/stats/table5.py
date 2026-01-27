#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
table5.py

Generate Table 5 (category-level aggregation) from:
  - manual evaluation JSON (list of dicts; has "ID", "QuestionNumber", "..._Total", "..._<metric>")
  - llm evaluation JSON (list of dicts; list order is treated as ID=1..N; has "QuestionNumber", "...", and scores)
  - q->category mapping JSON (e.g., {"q1": "Basic information", ...})

Outputs TWO markdown files:
  - table5.md               : ONE table (per Evaluation x Category) with mean Total score (95% bootstrap CI)
                              for each method, plus Friedman p and Kendall's W
  - table5_supplementary.md : ONE table with pairwise Wilcoxon tests on Total score per Evaluation x Category
                              (with Holm adjustment)

Formatting:
  - p-values are formatted as "<0.001" when p < 0.001 (to match table4.py style).

Assumptions:
  - Methods: ChatTogoVar, GPT-4o, VarChat
  - "Total" column key patterns:
      manual: "<Method>_Total"
      llm   : "<Method>"   (overall total score)
  - Question identifier: "QuestionNumber" like "q1"..."q50"
  - IDs:
      manual -> uses record["ID"]
      llm    -> uses array index (1-based)

Dependencies:
  - pandas, numpy, scipy, tabulate (for DataFrame.to_markdown)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

CANON_METHODS = ["ChatTogoVar", "GPT-4o", "VarChat"]

def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)

def save_text(path: str, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def format_p(p: float) -> str:
    if p is None or (isinstance(p, float) and (math.isnan(p) or math.isinf(p))):
        return ""
    if p < 0.001:
        return "<0.001"
    return f"{p:.6g}"

def bootstrap_mean_ci(
    x: np.ndarray, alpha: float = 0.05, n_boot: int = 10000, seed: int = 1
) -> Tuple[float, float]:
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


# ---- utilities ----

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def extract_total_scores_manual(rec: Dict[str, Any]) -> Dict[str, float]:
    # manual keys: "<Method>_Total"
    out: Dict[str, float] = {}
    for m in CANON_METHODS:
        k = f"{m}_Total"
        if k in rec and is_number(rec[k]):
            out[m] = to_float(rec[k])
    return out


def extract_total_scores_llm(rec: Dict[str, Any]) -> Dict[str, float]:
    # llm keys: "<Method>" (overall total)
    out: Dict[str, float] = {}
    for m in CANON_METHODS:
        if m in rec and is_number(rec[m]):
            out[m] = to_float(rec[m])
    return out


def build_long_df_manual(manual: List[Dict[str, Any]], q2cat: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for rec in manual:
        q = str(rec.get("QuestionNumber", "")).strip()
        if not q:
            continue
        cat = q2cat.get(q, "Unknown")
        rid = rec.get("ID", None)
        if rid is None:
            continue
        scores = extract_total_scores_manual(rec)
        for m, sc in scores.items():
            rows.append(
                {
                    "eval": "Manual",
                    "ID": int(rid),
                    "q": q,
                    "category": cat,
                    "method": m,
                    "score": float(sc),
                }
            )
    return pd.DataFrame(rows)


def build_long_df_llm(llm: List[Dict[str, Any]], q2cat: Dict[str, str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for i, rec in enumerate(llm, start=1):
        q = str(rec.get("QuestionNumber", "")).strip()
        if not q:
            continue
        cat = q2cat.get(q, "Unknown")
        scores = extract_total_scores_llm(rec)
        for m, sc in scores.items():
            rows.append(
                {
                    "eval": "LLM",
                    "ID": int(i),  # array index as ID
                    "q": q,
                    "category": cat,
                    "method": m,
                    "score": float(sc),
                }
            )
    return pd.DataFrame(rows)


@dataclass
class FriedmanResult:
    chi2: float
    p: float
    w: float
    n_complete: int


def friedman_test_by_group(long_df: pd.DataFrame) -> FriedmanResult:
    """
    Friedman test (paired) across the three methods within one group (Evaluation x Category).
    Pairing is by ID; requires all three methods present for each ID.
    """
    piv = long_df.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=CANON_METHODS, how="any")
    n = len(piv)
    if n < 2:
        return FriedmanResult(float("nan"), float("nan"), float("nan"), int(n))

    a = piv["ChatTogoVar"].to_numpy()
    b = piv["GPT-4o"].to_numpy()
    c = piv["VarChat"].to_numpy()
    chi2, p = friedmanchisquare(a, b, c)

    # Kendall's W: effect size for Friedman
    k = 3
    w = float(chi2) / float(n * (k - 1)) if n > 0 else float("nan")
    return FriedmanResult(float(chi2), float(p), float(w), int(n))


def wilcoxon_pair(long_df: pd.DataFrame, a: str, b: str) -> Tuple[float, float, int]:
    """
    Paired Wilcoxon signed-rank test between methods a and b within one group (Evaluation x Category).
    Returns (W_stat, p_value, n_pairs).
    """
    piv = long_df.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=[a, b], how="any")
    n = len(piv)
    if n < 1:
        return (float("nan"), float("nan"), 0)

    x = piv[a].to_numpy()
    y = piv[b].to_numpy()
    diff = x - y
    if np.allclose(diff, 0):
        return (0.0, 1.0, int(n))

    stat, p = wilcoxon(
        x,
        y,
        zero_method="wilcox",
        correction=False,
        alternative="two-sided",
        mode="auto",
    )
    return (float(stat), float(p), int(n))


def mean_ci_str(x: np.ndarray, alpha: float, n_boot: int, seed: int) -> str:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return ""
    mu = float(np.mean(x))
    lo, hi = bootstrap_mean_ci(x, alpha=alpha, n_boot=n_boot, seed=seed)
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{mu:.2f}"
    return f"{mu:.2f} [{lo:.2f}, {hi:.2f}]"


def make_table5_main(
    long_all: pd.DataFrame, alpha: float, n_boot: int, seed: int
) -> pd.DataFrame:
    """
    Main table: one row per (Evaluation, Category).
    Columns: methods (mean [CI]), n_complete, friedman_p, kendalls_W
    """
    out_rows: List[Dict[str, Any]] = []

    for (ev, cat), sub in long_all.groupby(["eval", "category"], sort=True):
        row: Dict[str, Any] = {"Evaluation": ev, "Category": cat}

        for m in CANON_METHODS:
            xs = sub.loc[sub["method"] == m, "score"].to_numpy(dtype=float)
            row[m] = mean_ci_str(xs, alpha=alpha, n_boot=n_boot, seed=seed)

        fr = friedman_test_by_group(sub)
        row["n_complete"] = fr.n_complete
        row["friedman_p"] = format_p(fr.p)
        row["kendalls_W"] = f"{fr.w:.3f}" if math.isfinite(fr.w) else ""
        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    cols = ["Evaluation", "Category"] + CANON_METHODS + ["n_complete", "friedman_p", "kendalls_W"]
    df = df[cols]
    return df


def make_table5_pairwise(long_all: pd.DataFrame) -> pd.DataFrame:
    """
    Supplementary table: pairwise Wilcoxon tests per (Evaluation, Category), Holm-adjusted.
    """
    pairs = [("ChatTogoVar", "GPT-4o"), ("ChatTogoVar", "VarChat"), ("GPT-4o", "VarChat")]
    rows: List[Dict[str, Any]] = []

    for (ev, cat), sub in long_all.groupby(["eval", "category"], sort=True):
        raw_pvals: List[float] = []
        tmp_rows: List[Dict[str, Any]] = []

        for a, b in pairs:
            stat, p, n = wilcoxon_pair(sub, a, b)
            raw_pvals.append(p if math.isfinite(p) else 1.0)
            tmp_rows.append(
                {
                    "Evaluation": ev,
                    "Category": cat,
                    "Comparison": f"{a} vs {b}",
                    "n_pairs": n,
                    "wilcoxon_W": f"{stat:.3f}" if math.isfinite(stat) else "",
                    "p": p,
                }
            )

        padj = holm_adjust(raw_pvals)
        for r, adj in zip(tmp_rows, padj):
            r["p"] = format_p(r["p"])
            r["p_holm"] = format_p(float(adj))
            rows.append(r)

    df = pd.DataFrame(rows)
    df = df[["Evaluation", "Category", "Comparison", "n_pairs", "wilcoxon_W", "p", "p_holm"]]
    return df


def df_to_markdown_tabulate(df: pd.DataFrame) -> str:
    # Requires tabulate installed (pandas uses it for to_markdown)
    return df.to_markdown(index=False)


def write_markdown_main(df_main: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Table 5 (Category-level summary; Total score)\n")
    lines.append(
        "Mean Total score is shown as mean [95% bootstrap CI]. "
        "For each Evaluation × Category, a Friedman test (paired across methods within the same questions) is performed; "
        "Kendall’s W is reported as an effect size. "
        "p-values are formatted as “<0.001” when p < 0.001.\n"
    )
    lines.append(df_to_markdown_tabulate(df_main))
    lines.append("")
    return "\n".join(lines)


def write_markdown_supp(df_pair: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Table 5 Supplementary (Pairwise Wilcoxon tests; Total score)\n")
    lines.append(
        "Pairwise Wilcoxon signed-rank tests are computed per Evaluation × Category (paired by ID). "
        "Holm adjustment is applied within each Evaluation × Category across the three pairwise comparisons. "
        "p-values are formatted as “<0.001” when p < 0.001.\n"
    )
    lines.append(df_to_markdown_tabulate(df_pair))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True, help="Manual evaluation JSON (aggregate_human.json)")
    ap.add_argument("--llm", required=True, help="LLM evaluation JSON (aggregate_aqes.json)")
    ap.add_argument("--q2cat", required=True, help="QuestionNumber (q1..q50) -> category JSON")
    ap.add_argument("--out_main", required=True, help="Output markdown for main table (table5.md)")
    ap.add_argument("--out_supp", required=True, help="Output markdown for supplementary table (table5_supplementary.md)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for CI (default 0.05 -> 95% CI)")
    ap.add_argument("--n_boot", type=int, default=10000, help="Bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed (default 1)")
    args = ap.parse_args()

    manual = load_json(args.manual)
    llm = load_json(args.llm)
    q2cat = load_json(args.q2cat)

    df_m = build_long_df_manual(manual, q2cat)
    df_l = build_long_df_llm(llm, q2cat)
    long_all = pd.concat([df_m, df_l], ignore_index=True)

    df_main = make_table5_main(long_all, alpha=args.alpha, n_boot=args.n_boot, seed=args.seed)
    md_main = write_markdown_main(df_main)
    save_text(args.out_main, md_main)

    df_pair = make_table5_pairwise(long_all)
    md_supp = write_markdown_supp(df_pair)
    save_text(args.out_supp, md_supp)


if __name__ == "__main__":
    main()
