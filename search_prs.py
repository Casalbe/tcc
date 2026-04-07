import requests
from config import HEADERS

SEARCH_URL = "https://api.github.com/search/issues"

def search_pull_requests(page=1, per_page=10):
    query = (
        "is:pr is:merged language:Python comments:>0"
    )

    params = {
        "q": query,
        "sort": "updated",
        "order": "desc",
        "page": page,
        "per_page": per_page
    }

    response = requests.get(SEARCH_URL, headers=HEADERS, params=params)
    if response.status_code == 401:
        raise RuntimeError(
            "GitHub API returned 401 Unauthorized. "
            "Set a valid token in GITHUB_TOKEN (or GITHUB_TOKEN_ENV) in your environment/.env."
        )
    response.raise_for_status()
    return response.json()["items"]

def parse_pr_info(pr_item):
    repo_url = pr_item["repository_url"]
    repo_full_name = repo_url.replace("https://api.github.com/repos/", "")
    pr_number = pr_item["number"]
    return repo_full_name, pr_number
