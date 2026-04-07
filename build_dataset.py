import json
import random
from pathlib import Path
from tqdm import tqdm

from search_prs import search_pull_requests, parse_pr_info
from fetch_comments import fetch_review_comments, is_valid_comment
from code_utils import fetch_file_content, extract_window

human_reviews = []
code_only = []

MAX_PRS = 20    # Max de prs para buscar
WINDOW_SIZE = 15 # Janela de linhas ao redor do comentario
MAX_SAMPLES_PER_PR = 1  # Limita quantos comentarios pegar por PR (melhora diversidade)

pr_items = search_pull_requests(per_page=MAX_PRS)

for pr in tqdm(pr_items):
    repo, pr_number = parse_pr_info(pr)

    try:
        comments = fetch_review_comments(repo, pr_number)
    except Exception:
        continue

    valid_comments = [c for c in comments if is_valid_comment(c)]
    # Em PRs com muitos comentarios, embaralha para evitar sempre pegar o mesmo trecho
    random.shuffle(valid_comments)
    valid_comments = valid_comments[:MAX_SAMPLES_PER_PR]

    for c in valid_comments:
        if not is_valid_comment(c):
            continue

        line = c.get("line") or c.get("original_line")
        commit = c["commit_id"]
        path = c["path"]

        try:
            lines = fetch_file_content(repo, path, commit)
            snippet, start, end = extract_window(lines, line, WINDOW_SIZE)
        except Exception:
            continue

        uid = f"{repo}#{pr_number}#{c['id']}"

        human_reviews.append({
            "id": uid,
            "repo": repo,
            "pull_request": pr_number,
            "commit": commit,
            "file": path,
            "start_line": start,
            "end_line": end,
            "code_snippet": snippet,
            "human_review": c["body"]
        })

        code_only.append({
            "id": uid,
            "code_snippet": snippet
        })

data_dir = Path(__file__).resolve().parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

with open(data_dir / "human_reviews.json", "w", encoding="utf-8") as f:
    json.dump(human_reviews, f, indent=2, ensure_ascii=False)

with open(data_dir / "code_only.json", "w", encoding="utf-8") as f:
    json.dump(code_only, f, indent=2, ensure_ascii=False)
