#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
table5.py

Generate Table 5 (category-level aggregation) from:
  - manual evaluation JSON (list of dicts; has "ID", "QuestionNumber", "..._Total")
  - llm evaluation JSON (list of dicts; list order is treated as ID=1..N; has "QuestionNumber", "<Method>" totals)
  - q->category mapping JSON (e.g., {"q1": "Basic information", ...})

Outputs TWO markdown files:
  - table5.md               : MAIN table (fixed to Manual(mean) + LLM only)
  - table5_supplementary.md : Supplementary pairwise Wilcoxon tables
                              - always includes Manual(mean) and LLM
                              - additionally includes Manual(primary) and Manual(secondary) ONLY if provided

Sensitivity analysis policy (as requested):
  - Main: Manual(mean) + LLM
  - Supp: Manual(mean) + Manual(primary) + Manual(secondary) + LLM (when primary/secondary are provided)

Dependencies:
  - pandas, numpy, scipy, tabulate
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


# ----------------------------
# IO / helpers
# ----------------------------
def ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def save_text(path: str, text: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


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


def q_key_to_int(q: str) -> int:
    s = str(q).strip()
    if len(s) >= 2 and (s[0].lower() == "q"):
        try:
            return int(s[1:])
        except Exception:
            pass
    return 10**9


def build_category_order(q2cat: Dict[str, str]) -> List[str]:
    cats: List[str] = []
    seen = set()
    for q, cat in sorted(q2cat.items(), key=lambda kv: q_key_to_int(kv[0])):
        if cat not in seen:
            cats.append(cat)
            seen.add(cat)
    return cats


# ----------------------------
# Score extraction
# ----------------------------
def extract_total_scores_manual(rec: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in CANON_METHODS:
        k = f"{m}_Total"
        if k in rec and is_number(rec[k]):
            out[m] = to_float(rec[k])
    return out


def extract_total_scores_llm(rec: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in CANON_METHODS:
        if m in rec and is_number(rec[m]):
            out[m] = to_float(rec[m])
    return out


# ----------------------------
# Build long dataframes
# ----------------------------
def build_long_df_manual(manual: List[Dict[str, Any]], q2cat: Dict[str, str], eval_label: str) -> pd.DataFrame:
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
                    "eval": eval_label,
                    "ID": int(rid),
                    "q": q,
                    "category": cat,
                    "method": m,
                    "score": float(sc),
                }
            )
    return pd.DataFrame(rows)


def build_long_df_llm(llm: List[Dict[str, Any]], q2cat: Dict[str, str], eval_label: str = "LLM") -> pd.DataFrame:
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
                    "eval": eval_label,
                    "ID": int(i),
                    "q": q,
                    "category": cat,
                    "method": m,
                    "score": float(sc),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------
# Tests / summaries
# ----------------------------
@dataclass
class FriedmanResult:
    chi2: float
    p: float
    w: float
    n_complete: int


def friedman_test_by_group(long_df: pd.DataFrame) -> FriedmanResult:
    piv = long_df.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=CANON_METHODS, how="any")
    n = len(piv)
    if n < 2:
        return FriedmanResult(float("nan"), float("nan"), float("nan"), int(n))

    a = piv["ChatTogoVar"].to_numpy()
    b = piv["GPT-4o"].to_numpy()
    c = piv["VarChat"].to_numpy()
    chi2, p = friedmanchisquare(a, b, c)

    k = 3
    w = float(chi2) / float(n * (k - 1)) if n > 0 else float("nan")
    return FriedmanResult(float(chi2), float(p), float(w), int(n))


def count_highest_scoring_answers(long_df: pd.DataFrame) -> Tuple[Dict[str, int], int, int]:
    piv = long_df.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=CANON_METHODS, how="any")
    n = int(len(piv))

    counts = {m: 0 for m in CANON_METHODS}
    if n == 0:
        return counts, 0, 0

    vals = piv[CANON_METHODS].to_numpy(dtype=float)
    row_max = np.max(vals, axis=1, keepdims=True)
    is_best = np.isclose(vals, row_max)
    n_best = is_best.sum(axis=1)

    n_ties = int((n_best >= 2).sum())

    unique_mask = (n_best == 1)
    if unique_mask.any():
        winner_idx = np.argmax(is_best[unique_mask, :], axis=1)
        for j, m in enumerate(CANON_METHODS):
            counts[m] = int((winner_idx == j).sum())

    return counts, n, n_ties


def wilcoxon_pair(long_df: pd.DataFrame, a: str, b: str) -> Tuple[float, float, int]:
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


def median_diff_pair(long_df: pd.DataFrame, a: str, b: str) -> Tuple[float, int]:
    piv = long_df.pivot_table(index="ID", columns="method", values="score", aggfunc="first")
    piv = piv.dropna(subset=[a, b], how="any")
    n = int(len(piv))
    if n < 1:
        return (float("nan"), 0)
    d = piv[a].to_numpy(dtype=float) - piv[b].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return (float("nan"), n)
    return (float(np.median(d)), n)


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


# ----------------------------
# Table builders
# ----------------------------
def make_table5_main(
    long_all: pd.DataFrame,
    alpha: float,
    n_boot: int,
    seed: int,
    category_order: List[str],
    eval_order: List[str],
) -> pd.DataFrame:
    out_rows: List[Dict[str, Any]] = []

    for ev in eval_order:
        for cat in category_order:
            sub = long_all[(long_all["eval"] == ev) & (long_all["category"] == cat)]
            if sub.empty:
                continue

            row: Dict[str, Any] = {"Evaluation": ev, "Category": cat}

            best_counts, _, n_ties_best = count_highest_scoring_answers(sub)
            row["N_highest_ChatTogoVar"] = best_counts["ChatTogoVar"]
            row["N_highest_GPT-4o"] = best_counts["GPT-4o"]
            row["N_highest_VarChat"] = best_counts["VarChat"]
            row["Tie_no_unique_winner"] = n_ties_best

            for m in CANON_METHODS:
                xs = sub.loc[sub["method"] == m, "score"].to_numpy(dtype=float)
                row[m] = mean_ci_str(xs, alpha=alpha, n_boot=n_boot, seed=seed)

            fr = friedman_test_by_group(sub)
            row["n_complete"] = fr.n_complete
            row["friedman_p"] = format_p(fr.p)
            row["kendalls_W"] = f"{fr.w:.3f}" if math.isfinite(fr.w) else ""

            out_rows.append(row)

    df = pd.DataFrame(out_rows)
    cols = (
        ["Evaluation", "Category"]
        + ["N_highest_ChatTogoVar", "N_highest_GPT-4o", "N_highest_VarChat", "Tie_no_unique_winner"]
        + CANON_METHODS
        + ["n_complete", "friedman_p", "kendalls_W"]
    )
    return df[cols]


def make_table5_pairwise(long_all: pd.DataFrame, category_order: List[str], eval_order: List[str]) -> pd.DataFrame:
    pairs = [("ChatTogoVar", "GPT-4o"), ("ChatTogoVar", "VarChat"), ("GPT-4o", "VarChat")]
    rows: List[Dict[str, Any]] = []

    for ev in eval_order:
        for cat in category_order:
            sub = long_all[(long_all["eval"] == ev) & (long_all["category"] == cat)]
            if sub.empty:
                continue

            raw_pvals: List[float] = []
            tmp_rows: List[Dict[str, Any]] = []

            for a, b in pairs:
                stat, p, n = wilcoxon_pair(sub, a, b)
                med, _n_med = median_diff_pair(sub, a, b)

                raw_pvals.append(p if math.isfinite(p) else 1.0)
                tmp_rows.append(
                    {
                        "Evaluation": ev,
                        "Category": cat,
                        "Comparison": f"{a} vs {b}",
                        "n_pairs": n,
                        "Median diff": "" if not math.isfinite(med) else f"{med:.3g}",
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
    return df[["Evaluation", "Category", "Comparison", "n_pairs", "Median diff", "wilcoxon_W", "p", "p_holm"]]


def df_to_markdown_tabulate(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def write_markdown_main(df_main: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Table 5 (Category-level summary; Total score)\n")
    lines.append(
        "Mean Total score is shown as mean [95% bootstrap CI]. "
        "For each Evaluation × Category, a Friedman test (paired across methods within the same questions) is performed; "
        "Kendall’s W is reported as an effect size. "
        "`N_highest_*` indicates the number of answers where the method achieved a unique highest score (paired by ID). "
        "`Tie_no_unique_winner` is the number of IDs where the highest score was shared by two or more methods (ties are not counted toward any method’s highest-score count). "
        "p-values are formatted as “<0.001” when p < 0.001.\n"
    )
    lines.append(df_to_markdown_tabulate(df_main))
    lines.append("")
    return "\n".join(lines)


def write_markdown_supp_blocks(df_pair: pd.DataFrame) -> str:
    lines: List[str] = []
    lines.append("# Table 5 Supplementary (Pairwise Wilcoxon tests; Total score)\n")
    lines.append(
        "Pairwise Wilcoxon signed-rank tests are computed per Evaluation × Category (paired by ID). "
        "Holm adjustment is applied within each Evaluation × Category across the three pairwise comparisons. "
        "Median paired differences are computed as (A−B), where A is the first method and B is the second method in the “Comparison” column; "
        "positive values indicate higher scores for A than B (negative values indicate the opposite). "
        "p-values are formatted as “<0.001” when p < 0.001.\n"
    )

    for ev in df_pair["Evaluation"].drop_duplicates().tolist():
        sub = df_pair[df_pair["Evaluation"] == ev]
        lines.append(f"## {ev}\n")
        lines.append(df_to_markdown_tabulate(sub))
        lines.append("")

    return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manual", required=True, help="Manual evaluation JSON (aggregate_human.json; human-mean)")
    ap.add_argument("--manual_primary", default=None, help="Primary rater manual JSON (optional)")
    ap.add_argument("--manual_secondary", default=None, help="Secondary rater manual JSON (optional)")
    ap.add_argument("--llm", required=True, help="LLM evaluation JSON (aggregate_llm.json)")
    ap.add_argument("--q2cat", required=True, help="QuestionNumber (q1..q50) -> category JSON")
    ap.add_argument("--out_main", required=True, help="Output markdown for main table (table5.md)")
    ap.add_argument("--out_supp", required=True, help="Output markdown for supplementary table (table5_supplementary.md)")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for CI (default 0.05 -> 95% CI)")
    ap.add_argument("--n_boot", type=int, default=10000, help="Bootstrap resamples (default 10000)")
    ap.add_argument("--seed", type=int, default=1, help="RNG seed (default 1)")
    args = ap.parse_args()

    manual_mean = load_json(args.manual)
    llm = load_json(args.llm)
    q2cat = load_json(args.q2cat)
    category_order = build_category_order(q2cat)

    # Build long dfs
    df_parts: List[pd.DataFrame] = []
    df_parts.append(build_long_df_manual(manual_mean, q2cat, eval_label="Manual (mean)"))
    df_parts.append(build_long_df_llm(llm, q2cat, eval_label="LLM"))

    have_raters = bool(args.manual_primary) and bool(args.manual_secondary)
    if have_raters:
        manual_p = load_json(args.manual_primary)
        manual_s = load_json(args.manual_secondary)
        df_parts.append(build_long_df_manual(manual_p, q2cat, eval_label="Manual (primary)"))
        df_parts.append(build_long_df_manual(manual_s, q2cat, eval_label="Manual (secondary)"))

    long_all = pd.concat(df_parts, ignore_index=True)

    # MAIN is fixed to Manual(mean) + LLM only
    eval_order_main: List[str] = ["Manual (mean)", "LLM"]
    df_main = make_table5_main(
        long_all,
        alpha=args.alpha,
        n_boot=args.n_boot,
        seed=args.seed,
        category_order=category_order,
        eval_order=eval_order_main,
    )
    md_main = write_markdown_main(df_main)
    save_text(args.out_main, md_main)

    # SUP includes primary/secondary only if provided
    eval_order_supp: List[str] = ["Manual (mean)"]
    if have_raters:
        eval_order_supp += ["Manual (primary)", "Manual (secondary)"]
    eval_order_supp += ["LLM"]

    df_pair = make_table5_pairwise(long_all, category_order=category_order, eval_order=eval_order_supp)
    md_supp = write_markdown_supp_blocks(df_pair)
    save_text(args.out_supp, md_supp)


if __name__ == "__main__":
    main()