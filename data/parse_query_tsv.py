"""
Convert a TSV of validated queries into a JSON mapping.

Reads a TSV file containing at least the columns
`topic_id` and `Query final`, and outputs a JSON
file where each key is the integer topic_id and
each value is the corresponding final query string.
"""

import csv
import json
import sys
sys.path.append("home/leon/tesis/messirve-ir")
from config.config import STORAGE_DIR
# import Path
from pathlib import Path


def tsv_to_json(tsv_path, json_path):
    """
    Read TSV and write JSON mapping topic_id → final query.

    Parameters
    ----------
    tsv_path : str
        Path to the input TSV file.
    json_path : str
        Path where the output JSON will be saved.

    Returns
    -------
    None
    """
    mapping = {}
    with open(tsv_path, mode="r", encoding="utf-8", newline="") as tsv_in:
        reader = csv.DictReader(tsv_in, delimiter="\t")
        for row in reader:
            # Extract and normalize fields
            tid = row.get("topic_id")
            final_q = row.get("Query final")
            if tid is None or final_q is None or final_q.strip() == "":
                # Skip rows with missing or empty topic_id or final query
                print(f"Skipping row with missing or empty fields: {row}")
                continue
            try:
                mapping[int(tid)] = final_q.strip()
            except ValueError:
                # Skip rows where topic_id is not an integer
                print(f"Skipping row with invalid topic_id: {row}")
                continue

    with open(json_path, mode="w", encoding="utf-8") as json_out:
        json.dump(mapping, json_out, ensure_ascii=False, indent=2)


def main():
    """
    Main entry point.

    Parses arguments and converts the TSV to JSON.
    """
    in_file = Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "anotacion_consultas_sinteticas.tsv"
    out_file = Path(STORAGE_DIR) / "legal_ir" / "data" / "corpus" / "consultas_sinteticas_393.json"
    tsv_to_json(in_file, out_file)
    print(f"Converted {in_file} to {out_file}")


if __name__ == "__main__":
    main()
