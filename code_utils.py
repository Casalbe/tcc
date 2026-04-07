import requests
import base64
from config import HEADERS

def fetch_file_content(repo, path, commit_sha):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    params = {"ref": commit_sha}

    response = requests.get(url, headers=HEADERS, params=params)
    response.raise_for_status()

    content = response.json()["content"]
    decoded = base64.b64decode(content).decode("utf-8", errors="ignore")
    return decoded.splitlines()

def extract_window(lines, line_number, window=15):
    start = max(0, line_number - window - 1)
    end = min(len(lines), line_number + window)
    snippet = "\n".join(lines[start:end])
    return snippet, start + 1, end
