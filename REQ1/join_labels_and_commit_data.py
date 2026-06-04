import json
from pathlib import Path

LABELS_FILE = "./splits/train/clean_labeled.json"
COMMITS_FILE = "./splits/train/clean.json"
OUTPUT_FILE = "reflection_labeled_clean_commits.jsonl"


def repo_to_project(repo: str) -> str:
    if "/" in repo:
        return repo.split("/", 1)[1]
    return repo


labels = json.loads(Path(LABELS_FILE).read_text(encoding="utf-8"))
commits = json.loads(Path(COMMITS_FILE).read_text(encoding="utf-8"))

commit_map = {
    c["commit_hash"]: c
    for c in commits
}

written = 0

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for item in labels:

        commit_hash = item["commit_hash"]

        if commit_hash not in commit_map:
            print(f"WARNING: {commit_hash} not found")
            continue

        commit = commit_map[commit_hash]

        row = {
            "commit_hash": commit_hash,
            "project": repo_to_project(commit["repo"]),
            "commit_message": commit["commit_message"],
            "diff": commit["diff"],
            "reflete_mudanca": item["reflete_mudanca"],
            "qualidade": item["qualidade"],
        }

        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            )
            + "\n"
        )

        written += 1

print(f"Wrote {written} rows")