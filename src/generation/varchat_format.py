#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
format_varchat_markdown.py

Post-process VarChat outputs saved as Markdown files that include:
  1) a JSON block (first {...} in the file), and
  2) a free-text summary (rest of the content)

This script:
  - Extracts the JSON block (if present)
  - Extracts the remaining text as "summary"
  - (Optional) Adds Japanese translation of the summary via Translator (Azure OpenAI)
  - Appends "References" (from JSON["cits"]) and "ClinVar Submissions" (from JSON["rcv_cits"])
  - Writes a cleaned Markdown file per input file

Usage examples:
  python3 format_varchat_markdown.py \
    --in-dir  /path/to/varchat/original \
    --out-dir /path/to/varchat/processed \
    --translate-ja
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

from dotenv import load_dotenv

# Ensure repo src/common is importable regardless of current working directory
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Translator is assumed to be your existing module/class.
# It should provide:
#   - Translator().generate_prompt(text: str) -> str
#   - Translator().query_azure_openai(prompt: str) -> Optional[str]
from common.translator import Translator

def read_text(path: Path) -> str:
    """Read a text file as UTF-8; ignore undecodable bytes."""
    return path.read_text(encoding="utf-8", errors="ignore")


def find_first_json_block(text: str) -> Optional[str]:
    """
    Return the first {...} block in the text (greedy across lines),
    or None if not found.
    """
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    return m.group(1) if m else None


def extract_json_and_summary(text: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extract JSON dict and the remaining text as summary.
    If JSON parsing fails, returns (None, original_text).
    """
    json_block = find_first_json_block(text)
    if not json_block:
        return None, text.strip()

    try:
        data = json.loads(json_block)
    except Exception:
        # JSON block exists but parsing failed; fall back to full text
        return None, text.strip()

    summary = text.replace(json_block, "").strip()
    return data, summary


def translate_to_ja(summary: str) -> str:
    """
    Translate summary to Japanese using Translator.
    If translation fails, return summary unchanged.
    """
    if not summary.strip():
        return summary

    load_dotenv()
    translator = Translator(deployment_name=os.getenv("deployment_name"))
    summary_ja = translator.translate_to_japanese(summary)

    if summary_ja:
        return summary + "\n\n" + summary_ja.strip()
    return summary


def build_references_section(data: Dict[str, Any]) -> str:
    """
    Build References section from data["cits"] if available.
    Each cit element is expected to have "cit" and optionally "pmid".
    """
    cits = data.get("cits", [])
    if not isinstance(cits, list) or len(cits) == 0:
        return ""

    lines: List[str] = []
    lines.append("## References")
    for cit in cits:
        if not isinstance(cit, dict):
            continue
        cit_text = str(cit.get("cit", "")).strip()
        pmid = str(cit.get("pmid", "")).strip()

        if not cit_text:
            continue

        # Keep your existing behavior: link to PubMed using pmid string if present.
        if pmid:
            lines.append(f"- {cit_text} [PubMed]({pmid})")
        else:
            lines.append(f"- {cit_text}")

    return "\n".join(lines).strip() + "\n"


def build_clinvar_section(data: Dict[str, Any]) -> str:
    """
    Build ClinVar Submissions section from data["rcv_cits"] if available.
    Each element is expected to have "rcv", "link", "condition".
    """
    rcv_cits = data.get("rcv_cits", [])
    if not isinstance(rcv_cits, list) or len(rcv_cits) == 0:
        return ""

    lines: List[str] = []
    lines.append("## ClinVar Submissions")
    for rcv in rcv_cits:
        if not isinstance(rcv, dict):
            continue
        rcv_id = str(rcv.get("rcv", "")).strip()
        link = str(rcv.get("link", "")).strip()
        condition = str(rcv.get("condition", "")).strip()

        if not rcv_id and not condition:
            continue

        if link and rcv_id:
            lines.append(f"- **[{rcv_id}]({link})**: {condition}".rstrip())
        elif rcv_id:
            lines.append(f"- **{rcv_id}**: {condition}".rstrip())
        else:
            lines.append(f"- {condition}".rstrip())

    return "\n".join(lines).strip() + "\n"


def process_one_file(
    in_path: Path,
    out_path: Path,
    do_translate_ja: bool,
) -> None:
    """Process one VarChat markdown file and write the cleaned output."""
    text = read_text(in_path)
    data, summary = extract_json_and_summary(text)

    if do_translate_ja:
        summary = translate_to_ja(summary)

    sections: List[str] = []
    if summary.strip():
        sections.append(summary.strip())

    if data:
        ref_sec = build_references_section(data)
        if ref_sec:
            sections.append(ref_sec.strip())

        clinvar_sec = build_clinvar_section(data)
        if clinvar_sec:
            sections.append(clinvar_sec.strip())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")


def main() -> None:
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in-dir",
        default=None,
        help="Input directory containing VarChat raw markdown files. ",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for cleaned markdown files.",
    )
    ap.add_argument(
        "--glob",
        default="*.md",
        help="Glob pattern to select input files (default: *.md)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N files (0 = no limit)",
    )
    ap.add_argument(
        "--translate-ja",
        action="store_true",
        help="Append Japanese translation of the summary via Translator",
    )
    args = ap.parse_args()

    base = os.getenv("VARCHAT_RESULT_DIR", "").strip()

    if args.in_dir:
        in_dir = Path(args.in_dir)
    else:
        if not base:
            raise SystemExit("ERROR: --in-dir not given and VARCHAT_RESULT_DIR is not set.")
        in_dir = Path(base) / "original"

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if not base:
            raise SystemExit("ERROR: --out-dir not given and VARCHAT_RESULT_DIR is not set.")
        out_dir = Path(base)

    if not in_dir.exists():
        raise SystemExit(f"ERROR: input directory does not exist: {in_dir}")

    files = sorted(in_dir.glob(args.glob))
    if args.limit and args.limit > 0:
        files = files[: args.limit]

    for p in files:
        out_path = out_dir / p.name
        print(f"[INFO] Processing: {p.name} -> {out_path}")
        process_one_file(p, out_path, do_translate_ja=args.translate_ja)

    print(f"[INFO] Done. Processed {len(files)} files.")


if __name__ == "__main__":
    main()
    
