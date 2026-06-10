"""
build_complete_jsons.py

Reads the existing splits/train and splits/test JSON files to get the
commit hashes that were already selected, then looks up each commit in
the five repo CSVs and writes new, more complete JSON files.

Output columns kept per commit:
    fix, ns, nd, nf, entropy, la, ld, lt, ndev, age, nuc, exp, rexp, sexp, contains_bug

Usage:
    python build_complete_jsons.py

Expected layout (relative to this script or adjust BASE_DIR / CSV_DIR):
    splits/
        train/
            clean.json
            defect_inducing.json
        test/
            clean.json
            defect_inducing.json
    csvs/
        camel_rot_gpt.csv
        fabric8_rot_gpt.csv
        gimp_rot_gpt.csv
        neutron_rot_gpt.csv
        postgresql_rot_gpt.csv
"""

import json
import csv
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(".")
SPLITS_DIR = BASE_DIR / "splits"
CSV_DIR    = BASE_DIR / "data/csvs"
OUT_DIR    = BASE_DIR / "splits"

WANTED_COLS = [
    "fix", "ns", "nd", "nf", "entropy",
    "la", "ld", "lt", "ndev", "age",
    "nuc", "exp", "rexp", "sexp",
    "contains_bug",
]

CSV_FILES = [
    CSV_DIR / "camel_rot_gpt.csv",
    CSV_DIR / "fabric8_rot_gpt.csv",
    CSV_DIR / "gimp_rot_gpt.csv",
    CSV_DIR / "neutron_rot_gpt.csv",
    CSV_DIR / "postgresql_rot_gpt.csv",
]

# ── Load all CSVs into one dict keyed by commit_hash ──────────────────────────

def load_all_csvs(csv_files):
    """
    Returns a dict:  commit_hash -> {col: value, ...}
    CSVs are semicolon-delimited.
    """
    master = {}
    for path in csv_files:
        if not path.exists():
            print(f"  [WARN] CSV not found, skipping: {path}")
            continue
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=";")
            before = len(master)
            for row in reader:
                # Normalise: strip whitespace from keys and string values
                clean_row = {}
                for k, v in row.items():
                    k = k.strip() if isinstance(k, str) else k
                    if isinstance(v, str):
                        v = v.strip()
                    elif v is None:
                        v = ""
                    else:
                        v = str(v)
                    clean_row[k] = v
                h = clean_row.get("commit_hash", "")
                if h:
                    master[h] = clean_row
        print(f"  Loaded {path.name}  →  added {len(master) - before}  (total: {len(master)})")
    return master


# ── Value casting ──────────────────────────────────────────────────────────────

def cast_value(value: str):
    """Try int, then float, then bool strings, else keep as string."""
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    return value


# ── Build one output record ────────────────────────────────────────────────────

def build_record(csv_row: dict, wanted_cols: list, contains_bug_override=None):
    record = {}
    for col in wanted_cols:
        if col == "contains_bug" and contains_bug_override is not None:
            record["contains_bug"] = contains_bug_override
        elif col in csv_row:
            record[col] = cast_value(csv_row[col])
        else:
            record[col] = None
    return record


# ── Process one JSON file ──────────────────────────────────────────────────────

def process_json(json_path: Path, csv_master: dict, contains_bug: bool, split: str):
    with open(json_path, encoding="utf-8") as f:
        existing = json.load(f)

    records = []
    missing = []

    for item in existing:
        h = item.get("commit_hash", "").strip()
        if not h:
            print(f"    [WARN] item without commit_hash in {json_path}, skipping")
            continue

        csv_row = csv_master.get(h)
        if csv_row is None:
            missing.append(h)
            continue

        record = {"commit_hash": h, "split": split}
        if "repo" in item:
            record["repo"] = item["repo"]
        record.update(build_record(csv_row, WANTED_COLS, contains_bug_override=contains_bug))
        records.append(record)

    if missing:
        print(f"    [WARN] {len(missing)} commits not found in any CSV:")
        for h in missing[:5]:
            print(f"           {h}")
        if len(missing) > 5:
            print(f"           … and {len(missing) - 5} more")

    print(f"    {json_path}  →  {len(records)} matched  ({len(missing)} missed)")
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading CSVs …")
    csv_master = load_all_csvs(CSV_FILES)
    print(f"Total unique commits in CSVs: {len(csv_master)}\n")

    all_clean           = []
    all_defect_inducing = []

    for split in ("train", "test"):
        split_dir = SPLITS_DIR / split
        print(f"── Processing split: {split} ──")

        clean_path  = split_dir / "clean.json"
        defect_path = split_dir / "defect_inducing.json"

        if clean_path.exists():
            all_clean.extend(process_json(clean_path, csv_master, contains_bug=False, split=split))
        else:
            print(f"  [SKIP] {clean_path} not found")

        if defect_path.exists():
            all_defect_inducing.extend(process_json(defect_path, csv_master, contains_bug=True, split=split))
        else:
            print(f"  [SKIP] {defect_path} not found")

        print()

    for split in ("train", "test"):
        out_dir = OUT_DIR / split
        out_dir.mkdir(parents=True, exist_ok=True)

        split_clean  = [r for r in all_clean           if r.get("split") == split]
        split_defect = [r for r in all_defect_inducing if r.get("split") == split]

        clean_out  = out_dir / "complete_clean.json"
        defect_out = out_dir / "complete_defect_inducing.json"

        with open(clean_out, "w", encoding="utf-8") as f:
            json.dump(split_clean, f, indent=2, ensure_ascii=False)
        print(f"Wrote {clean_out}  ({len(split_clean)} records)")

        with open(defect_out, "w", encoding="utf-8") as f:
            json.dump(split_defect, f, indent=2, ensure_ascii=False)
        print(f"Wrote {defect_out}  ({len(split_defect)} records)")

    print("\nDone ✓")


if __name__ == "__main__":
    main()