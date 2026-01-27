#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
translator.py

A small translation helper for varchat_format.py.

- Importable usage (recommended):
    from translator import Translator

    tr = Translator()  # uses OpenAIAzure (expects env vars handled there)
    ja = tr.translate_to_japanese(text)

- CLI usage (optional):
    python3 translator.py --in input.txt --out output.txt
    echo "hello" | python3 translator.py

Notes:
- Keeps prompts and comments in English.
- Designed to be called from varchat_format.py as a library (no side effects on import).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from dotenv import load_dotenv

from common.open_ai_azure import OpenAIAzure

class Translator(OpenAIAzure):
    def __init__(self, deployment_name: Optional[str] = None) -> None:
        """
        deployment_name:
          If provided, overrides the deployment/model name.
          Otherwise, uses env var "deployment_name" if set; if not, OpenAIAzure's default.
        """
        super().__init__()
        self._deployment_name = deployment_name or os.getenv("deployment_name")

    @staticmethod
    def build_prompt(text: str) -> str:
        """
        Build an instruction prompt for Japanese translation.

        We request natural Japanese and keep technical terms (rs numbers, HGVS, gene names) unchanged.
        """
        return (
            "Translate the following text into natural Japanese. "
            "Keep technical terms (e.g., rs numbers, HGVS, gene symbols) as-is. "
            "Do not add new information.\n\n"
            f"{text}"
        )

    def translate_to_japanese(
        self,
        text: str,
        *,
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> Optional[str]:
        """
        Translate text to Japanese using Azure OpenAI.

        Returns:
          Translated text (str) or None on error.
        """
        prompt = self.build_prompt(text)

        try:
            # Prefer using a "user" message for the actual content; keep a short system message.
            model = self._deployment_name  # may be None; OpenAIAzure may still handle default
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a professional Japanese translator."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"[ERROR] Azure OpenAI translation failed: {e}", file=sys.stderr)
            return None


def _read_text(path: Optional[str]) -> str:
    if path:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    return sys.stdin.read()


def _write_text(path: Optional[str], text: str) -> None:
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
        if not text.endswith("\n"):
            sys.stdout.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Translate text into Japanese via Azure OpenAI.")
    ap.add_argument("--in", dest="in_path", default=None, help="Input text file (default: stdin)")
    ap.add_argument("--out", dest="out_path", default=None, help="Output text file (default: stdout)")
    ap.add_argument("--deployment-name", default=None, help='Azure deployment/model name (default: env "deployment_name")')
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--no-dotenv", action="store_true", help="Do not call load_dotenv()")
    args = ap.parse_args()

    if not args.no_dotenv:
        load_dotenv()

    text = _read_text(args.in_path).strip()
    if not text:
        _write_text(args.out_path, "")
        return

    tr = Translator(deployment_name=args.deployment_name)
    ja = tr.translate_to_japanese(text, max_tokens=args.max_tokens, temperature=args.temperature)
    _write_text(args.out_path, ja or "")


if __name__ == "__main__":
    main()