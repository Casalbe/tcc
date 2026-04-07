import json
from llm.prompt import build_review_prompt
from llm.chatgpt_client import generate_chatgpt_review
from llm.claude_client import generate_claude_review

INPUT_PATH = "data/code_only.json"
OUTPUT_PATH = "data/enriched_reviews.json"

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

results = []

for i, sample in enumerate(dataset):
    code = sample["code_snippet"]
    prompt = build_review_prompt(code)

    print(f"Processing sample {i+1}/{len(dataset)}")

    chatgpt_review = generate_chatgpt_review(prompt)
    claude_review = generate_claude_review(prompt)

    results.append({
        "id": sample["id"],
        "code_snippet": code,
        "llm_reviews": {
            "chatgpt": {
                "text": chatgpt_review,
                "model": "gpt-4.1",
                "temperature": 0.2
            },
            "claude": {
                "text": claude_review,
                "model": "claude-3.7-sonnet",
                "temperature": 0.2
            }
        }
    })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
