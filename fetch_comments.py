import requests
from config import HEADERS

def fetch_review_comments(repo, pr_number):
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()

def is_valid_comment(comment):
    if not comment.get("body"):
        return False

    if len(comment["body"].strip()) < 15:
        return False

    if not comment.get("path", "").endswith(".py"):
        return False

    if comment.get("line") is None and comment.get("original_line") is None:
        return False

    user = comment.get("user", {}).get("login", "").lower()
    if "bot" in user:
        return False

    return True
