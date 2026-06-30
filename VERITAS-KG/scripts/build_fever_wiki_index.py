from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def iter_claim_files(data_root: Path, splits: Iterable[str], limit: Optional[int]) -> Iterable[Dict[str, Any]]:
    for split in splits:
        path = data_root / "fever" / f"paper_{split}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle):
                if limit is not None and idx >= limit:
                    break
                if line.strip():
                    yield json.loads(line)


def collect_targets(rows: Iterable[Dict[str, Any]]) -> Tuple[Set[str], Set[Tuple[str, int]]]:
    pages: Set[str] = set()
    lines: Set[Tuple[str, int]] = set()
    for item in rows:
        for group in item.get("evidence") or []:
            if not isinstance(group, list):
                continue
            for evidence in group:
                if not isinstance(evidence, list) or len(evidence) < 4:
                    continue
                page = evidence[2]
                line_no = evidence[3]
                if page is None or line_no is None:
                    continue
                try:
                    line_int = int(line_no)
                except (TypeError, ValueError):
                    continue
                pages.add(str(page))
                lines.add((str(page), line_int))
    return pages, lines


def parse_lines(raw: str) -> Dict[int, str]:
    parsed: Dict[int, str] = {}
    for row in raw.splitlines():
        parts = row.split("\t")
        if len(parts) < 2:
            continue
        try:
            line_no = int(parts[0])
        except ValueError:
            continue
        parsed[line_no] = parts[1]
    return parsed


def build_index(zip_path: Path, target_pages: Set[str], target_lines: Set[Tuple[str, int]]) -> Dict[str, Any]:
    pages: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("wiki-pages/wiki-") and name.endswith(".jsonl")
        ]
        for name in sorted(names):
            with archive.open(name) as handle:
                for raw_line in handle:
                    if not raw_line.strip():
                        continue
                    item = json.loads(raw_line.decode("utf-8"))
                    page_id = str(item.get("id", ""))
                    if page_id not in target_pages:
                        continue
                    line_map = parse_lines(str(item.get("lines", "")))
                    wanted = {
                        str(line_no): line_map.get(line_no, "")
                        for page, line_no in target_lines
                        if page == page_id
                    }
                    pages[page_id] = {
                        "id": page_id,
                        "text": item.get("text", ""),
                        "lines": wanted,
                    }
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact FEVER wiki evidence index from wiki-pages.zip.")
    parser.add_argument("--data-root", default="../data", type=Path)
    parser.add_argument("--zip-path", default=None, type=Path)
    parser.add_argument("--output", default=Path("data/processed/fever_wiki_index.json"), type=Path)
    parser.add_argument("--split", action="append", choices=["dev", "test"], default=[])
    parser.add_argument("--limit", type=int, default=-1, help="Claims per split; use -1 for all.")
    args = parser.parse_args()

    splits = args.split or ["dev", "test"]
    limit = None if args.limit < 0 else args.limit
    zip_path = args.zip_path or args.data_root / "fever" / "wiki-pages.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Missing FEVER wiki zip: {zip_path}")

    target_pages, target_lines = collect_targets(iter_claim_files(args.data_root, splits, limit))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pages = build_index(zip_path, target_pages, target_lines)
    payload = {
        "source": str(zip_path),
        "splits": splits,
        "target_pages": len(target_pages),
        "target_lines": len(target_lines),
        "indexed_pages": len(pages),
        "pages": pages,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "pages"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
