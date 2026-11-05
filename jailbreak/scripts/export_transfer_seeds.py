#!/usr/bin/env python3
"""Export transfer-evaluation datasets into the common seeds JSONL format."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


def export_jailbreakhub(data_root: Path, out_path: Path, *, max_rows: int | None = None) -> int:
    src = data_root / "jailbreakhub" / "data" / "forbidden_question" / "forbidden_question_set.csv"
    if not src.exists():
        raise FileNotFoundError(f"JailbreakHub forbidden question CSV not found: {src}")
    rows: list[dict[str, Any]] = []
    with src.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if max_rows is not None and len(rows) >= max_rows:
                break
            text = str(row.get("question") or "").strip()
            if not text:
                continue
            category = str(row.get("content_policy_name") or "JailbreakHub").strip() or "JailbreakHub"
            rows.append({
                "question_id": f"jailbreakhub::{row.get('content_policy_id', 'na')}::{row.get('q_id', i)}",
                "category": category,
                "text": text,
                "source": "jailbreakhub_forbidden_question_set",
            })
    if not rows:
        raise RuntimeError(f"No usable JailbreakHub rows exported from {src}")
    _write_jsonl(out_path, rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=["jailbreakhub"], default="jailbreakhub")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-rows", type=int, default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset == "jailbreakhub":
        n = export_jailbreakhub(args.data_root, args.out, max_rows=args.max_rows)
    else:  # pragma: no cover
        raise AssertionError(args.dataset)
    print(f"[transfer-seeds] wrote {n} rows -> {args.out}")


if __name__ == "__main__":
    main()
