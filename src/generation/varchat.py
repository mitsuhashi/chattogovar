#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
varchat.py

Fetch answers from VarChat for a list of rs numbers and save them as Markdown files.

Changes from the original script:
  1) Input/output paths are provided via command-line options (argparse).
  2) All comments/docstrings are in English.
  3) Added --rs-limit to process only the first N rs numbers.
  4) Added basic retry/backoff and request timeout controls.
  5) Added --sleep to control request interval.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Import common utilities
# -----------------------------------------------------------------------------
# Prefer: src/common/utils.py  ->  from common.utils import ...
# Fallback: legacy utils.py in PYTHONPATH or repo root.
try:
    from common.utils import read_rs_numbers_from_file, save_answer_to_markdown  # type: ignore
except Exception:
    # Add ../common to sys.path (relative to this file) as a fallback
    _this = Path(__file__).resolve()
    _common = _this.parents[1] / "common"  # .../src/common
    if _common.exists():
        sys.path.insert(0, str(_common))
    try:
        from utils import read_rs_numbers_from_file, save_answer_to_markdown  # type: ignore
    except Exception as e:
        raise ImportError(
            "Could not import read_rs_numbers_from_file / save_answer_to_markdown. "
            "Please ensure they exist in src/common/utils.py as common.utils, "
            "or are importable as utils.py."
        ) from e


# -----------------------------------------------------------------------------
# VarChat helpers
# -----------------------------------------------------------------------------
def build_varchat_url(
    rs_number: str,
    user: str,
    user_input: str,
    country: str,
    source: str = "",
) -> str:
    """
    Build VarChat API URL (stream=true).
    Note: VarChat query parameters are kept compatible with your original script.
    """
    base = "https://varchat.engenome.com/varchat/api/request/"
    params = (
        f"?gene=&rs={rs_number}&hgvs_c=&hgvs_p=&hgvs_m=&transcript=&genomic_coord="
        f"&stream=true&user={user}&user_input={user_input}&country={country}&source={source}"
    )
    return base + params


def parse_sse_text_to_output(text: str) -> str:
    """
    Parse VarChat SSE-like response text.
    Your original logic extracted the first JSON line (if present) and concatenated the remaining data lines.
    This function is slightly more defensive:
      - Collects all lines beginning with "data:"
      - Uses the first "data: {...}" as json_line if present
      - Concatenates the remaining "data:" payloads as answer text
    """
    json_line: Optional[str] = None
    chunks: List[str] = []

    for raw in text.splitlines():
        raw = raw.strip()
        if not raw.startswith("data:"):
            continue
        payload = raw[len("data:") :].lstrip()

        if json_line is None and payload.startswith("{") and payload.endswith("}"):
            json_line = payload
            continue

        # For non-JSON payloads, keep as text
        chunks.append(payload)

    combined_text = "".join(chunks).strip()
    if json_line and combined_text:
        return f"{json_line}\n{combined_text}\n"
    if json_line:
        return f"{json_line}\n"
    return f"{combined_text}\n" if combined_text else ""


def fetch_varchat(
    rs_number: str,
    user: str,
    user_input: str,
    country: str,
    timeout: int,
    retries: int,
    backoff: float,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Fetch VarChat response for an rs number.
    Returns (output_text, error_message).
    """
    url = build_varchat_url(rs_number, user=user, user_input=user_input, country=country)

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            out = parse_sse_text_to_output(resp.text)
            if not out.strip():
                return None, "Empty response after parsing SSE data lines."
            return out, None
        except requests.exceptions.RequestException as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(backoff * (2**attempt))
                continue
            return None, last_err

    return None, last_err


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Fetch VarChat answers for rs numbers and save as Markdown.")
    ap.add_argument(
        "--rs-list",
        default="pubtator3/rs.txt",
        help="Path to rs list file (default: pubtator3/rs.txt)",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Output directory. If omitted, uses $VARCHAT_RESULT_DIR",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=5.0,
        help="Seconds to sleep between requests (default: 5)",
    )
    ap.add_argument(
        "--rs-limit",
        type=int,
        default=0,
        help="Process only the first N rs numbers (0 = no limit)",
    )
    ap.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout seconds (default: 60)",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retries on request failure (default: 2)",
    )
    ap.add_argument(
        "--backoff",
        type=float,
        default=1.0,
        help="Base backoff seconds for retries (default: 1.0)",
    )
    ap.add_argument(
        "--user",
        default=os.getenv("VARCHAT_USER", "mitsuhashi"),
        help='VarChat "user" parameter (default: $VARCHAT_USER or "mitsuhashi")',
    )
    ap.add_argument(
        "--country",
        default=os.getenv("VARCHAT_COUNTRY", "Japan"),
        help='VarChat "country" parameter (default: $VARCHAT_COUNTRY or "Japan")',
    )
    ap.add_argument(
        "--user-input",
        default=os.getenv("VARCHAT_USER_INPUT", "rs671"),
        help='VarChat "user_input" parameter (default: $VARCHAT_USER_INPUT or "rs671")',
    )
    return ap.parse_args()


def main() -> None:
    load_dotenv()

    args = parse_args()

    out_dir = args.outdir
    if not out_dir:
        raise SystemExit("ERROR: --outdir is not set.")

    out_dir_path = Path(out_dir).resolve()
    original_dir = out_dir_path / "original"
    original_dir.mkdir(parents=True, exist_ok=True)

    rs_numbers: List[str] = read_rs_numbers_from_file(args.rs_list)
    if args.rs_limit and args.rs_limit > 0:
        rs_numbers = rs_numbers[: args.rs_limit]

    for rs in rs_numbers:
        rs = rs.strip()
        if not rs:
            continue

        print(f"[INFO] Processing: {rs}")

        text, err = fetch_varchat(
            rs_number=rs,
            user=args.user,
            user_input=args.user_input,
            country=args.country,
            timeout=args.timeout,
            retries=args.retries,
            backoff=args.backoff,
        )

        if text:
            out_path = original_dir / f"{rs}.md"
            save_answer_to_markdown(str(out_path), text)
            print("[INFO] done.")
        else:
            print(f"[WARN] VarChat failed for {rs}: {err}")

        time.sleep(args.sleep)


if __name__ == "__main__":
    main()