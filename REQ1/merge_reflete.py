"""
merge_reflete.py

Reads the reflete_mudanca annotations from the two .jsonl sidecar files
and stitches the field into the complete test JSON files by matching on
commit_hash.

Only the TEST split needs this — the stratified evaluation is test-only.

Expected layout:
    splits/test/
        complete_clean.json
        complete_defect_inducing.json
        reflection_labeled_clean_commits.jsonl
        reflection_labeled_defec_ind_commits.jsonl

Output (files are overwritten in-place):
    splits/test/complete_clean.json          ← now includes "reflete_mudanca"
    splits/test/complete_defect_inducing.json ← now includes "reflete_mudanca"

Commits that have no matching annotation get  "reflete_mudanca": null
so the field is always present and the training script can filter cleanly.

Usage:
    python merge_reflete.py
"""

import json
from pathlib import Path

# ── Configuration ─────────────────────────────────────────────────────────────

TEST_DIR = Path("splits/test")

JSONL_FILES = [
    TEST_DIR / "reflection_labeled_clean_commits.jsonl",
    TEST_DIR / "reflection_labeled_defec_ind_commits.jsonl",
]

JSON_FILES = [
    TEST_DIR / "complete_clean.json",
    TEST_DIR / "complete_defect_inducing.json",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_json(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Build annotation lookup ───────────────────────────────────────────────────

def build_annotation_map(jsonl_files: list[Path]) -> dict[str, str]:
    """
    Returns  commit_hash -> reflete_mudanca  from all jsonl sidecar files.
    If a hash appears in both files the last value wins (shouldn't happen).
    """
    mapping: dict[str, str] = {}
    for path in jsonl_files:
        if not path.exists():
            print(f"  [WARN] JSONL not found, skipping: {path}")
            continue
        records = load_jsonl(path)
        found = 0
        for r in records:
            h = r.get("commit_hash", "").strip()
            v = r.get("reflete_mudanca", "")
            if h and v:
                mapping[h] = v
                found += 1
        print(f"  {path.name}  →  {found} annotations loaded  (total so far: {len(mapping)})")
    return mapping

# ── Merge into JSON files ─────────────────────────────────────────────────────

def merge(json_path: Path, annotation_map: dict[str, str]):
    if not json_path.exists():
        print(f"  [SKIP] {json_path} not found")
        return

    records = load_json(json_path)
    matched = 0
    missing = 0

    for record in records:
        h = record.get("commit_hash", "").strip()
        reflete = annotation_map.get(h)   # None if not found
        record["reflete_mudanca"] = reflete
        if reflete is not None:
            matched += 1
        else:
            missing += 1

    save_json(json_path, records)
    print(f"  {json_path.name}  →  {matched} annotated  |  {missing} without annotation (set to null)")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Building annotation map from JSONL sidecars …")
    annotation_map = build_annotation_map(JSONL_FILES)
    print(f"Total unique annotations: {len(annotation_map)}\n")

    print("Merging into complete JSON files …")
    for json_path in JSON_FILES:
        merge(json_path, annotation_map)

    print("\nDone ✓")
    print("The field 'reflete_mudanca' is now present on every record in the test JSONs.")
    print("Records with no matching annotation have  \"reflete_mudanca\": null")


if __name__ == "__main__":
    main()