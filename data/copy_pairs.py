"""
copy_pairs.py
=============

Copy every document listed in a *pairs_to_annotate.tsv* file
(`qid<TAB>did`) from an original corpus directory to a new
destination folder.

The source files follow the naming pattern ``{did}.*`` where the
extension can be HTML, PDF, DOCX, etc.  The script looks for *exactly
one* match per document ID; if none (or more than one) are found it
reports the problem and continues.

Example
-------
>>> python copy_pairs.py \
        --pairs pairs_to_annotate.tsv \
        --src   orig_docs \
        --dst   docs_to_annotate

Notes
-----
* Requires Python 3.8+.
* Uses `pathlib` for portability and `shutil.copy2` to preserve
  timestamps/metadata.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Copy selected documents to a separate folder."
    )
    parser.add_argument(
        "--pairs",
        required=True,
        type=Path,
        help="TSV file with columns: qid<TAB>did",
    )
    parser.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Folder containing the 5 k original documents (orig_docs/).",
    )
    parser.add_argument(
        "--dst",
        required=True,
        type=Path,
        help="Destination folder for copies (docs_to_annotate/).",
    )
    return parser.parse_args()


def read_dids(pairs_path: Path) -> List[str]:
    """
    Read the second column (did) from *pairs_to_annotate.tsv*.

    Parameters
    ----------
    pairs_path : pathlib.Path
        Path to the TSV file.

    Returns
    -------
    list[str]
        Sorted list of unique document IDs to copy.
    """
    dids = set()
    with pairs_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 2:  # skip malformed lines
                continue
            _, did = parts
            dids.add(did)
    return sorted(dids)


def copy_docs(dids: List[str], src_dir: Path, dst_dir: Path) -> None:
    """
    Copy every ``did.*`` from *src_dir* to *dst_dir*.

    Parameters
    ----------
    dids : list[str]
        Document IDs to copy.
    src_dir : pathlib.Path
        Source directory containing original files.
    dst_dir : pathlib.Path
        Destination directory (will be created if missing).
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_missing, n_multi = 0, 0, 0

    for did in dids:
        matches = list(src_dir.glob(f"{did}.*"))
        if len(matches) == 1:
            shutil.copy2(matches[0], dst_dir / matches[0].name)
            n_ok += 1
        elif len(matches) == 0:
            print(f"[WARN] No file found for DID {did}", file=sys.stderr)
            n_missing += 1
        else:
            print(
                f"[WARN] Multiple files for DID {did}: {[m.name for m in matches]}",
                file=sys.stderr,
            )
            n_multi += 1

    print(
        f"✅  Copied {n_ok} files | ⚠️  {n_missing} missing | ⚠️  {n_multi} duplicated"
    )


def main() -> None:
    """Entry point when run as a script."""
    args = parse_args()
    did_list = read_dids(args.pairs)
    copy_docs(did_list, args.src, args.dst)


if __name__ == "__main__":
    main()