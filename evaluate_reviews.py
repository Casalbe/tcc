import json
import re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# dicionarios de palavras-chave

SUGGESTION_KEYWORDS = [
    "you should", "please consider", "you could", "i suggest", "why not",
    "it would be better", "i recommend", "i propose", "i advise", "it might be good",
    "consider changing", "think about", "might want to", "it may be useful",
    "you may", "it could be beneficial", "verify that", "ensure that", "check that",
    "suggestion", "consider either", "consider"
]

ACTION_VERBS = [
    "rename", "remove", "add", "extract", "check", "validate",
    "handle", "refactor", "simplify", "split", "merge", "avoid"
]

POLITENESS_MARKERS = [
    "please", "maybe", "might", "could", "i think", "perhaps"
]

ISSUE_KEYWORDS = {
    "bug": ["bug", "error", "incorrect", "wrong", "edge case", "fail"],
    "performance": ["performance", "slow", "inefficient", "optimize"],
    "security": ["security", "vulnerability", "unsafe", "sanitize"],
    "readability": ["readability", "clear", "confusing", "hard to read"],
    "design": ["design", "architecture", "responsibility", "abstraction"],
    "style": ["style", "naming", "format", "pep8"],
    "testing": ["test", "coverage", "unit test", "assert"]
}

CODE_TOKEN_REGEX = re.compile(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b")

# funcoes de metrica

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

def comment_length(comment):
    return len(tokenize(comment))

def code_reference_density(comment, code_snippet):
    comment_tokens = set(CODE_TOKEN_REGEX.findall(comment))
    code_tokens = set(CODE_TOKEN_REGEX.findall(code_snippet))
    if not comment_tokens:
        return 0.0
    overlap = comment_tokens & code_tokens
    return len(overlap) / len(comment_tokens)

def has_suggestion(comment):
    text = comment.lower()
    return any(k in text for k in SUGGESTION_KEYWORDS)

def recommendation_score(comment):
    text = comment.lower()
    return sum(text.count(v) for v in ACTION_VERBS)

def politeness_score(comment):
    text = comment.lower()
    return sum(text.count(p) for p in POLITENESS_MARKERS)

def classify_issues(comment):
    text = comment.lower()
    found = []
    for issue, keywords in ISSUE_KEYWORDS.items():
        if any(k in text for k in keywords):
            found.append(issue)
    return found

def semantic_similarity(comment, code):
    documents = [comment, code]

    vectorizer = TfidfVectorizer(
        token_pattern=r"\b[a-zA-Z_][a-zA-Z0-9_]*\b",
        lowercase=True,
        stop_words=None
    )

    tfidf = vectorizer.fit_transform(documents)

    sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(float(sim), 3)

def evaluate_single_review(comment, code, entry_id):
    issues = classify_issues(comment)

    return {
        "id": entry_id,
        "length": comment_length(comment),
        "code_ref_density": round(code_reference_density(comment, code), 3),
        "has_suggestion": has_suggestion(comment),
        "recommendation_score": recommendation_score(comment),
        "issue_types": issues,
        "politeness_score": politeness_score(comment),
        "semantic_similarity": semantic_similarity(comment, code)
    }


# funcao principal

def evaluate_all_reviews(human_path, llm_path):
    with open(human_path, "r", encoding="utf-8") as f:
        human_data = json.load(f)

    with open(llm_path, "r", encoding="utf-8") as f:
        llm_data = json.load(f)

    human_by_id = {e["id"]: e for e in human_data}
    llm_by_id = {e["id"]: e for e in llm_data}

    evaluations = []
    table_rows = []

    issue_distribution = defaultdict(Counter)

    for entry_id, human_entry in human_by_id.items():
        if entry_id not in llm_by_id:
            continue  # PELOAMORDEDEUSNAOTIRAISSO

        llm_entry = llm_by_id[entry_id]
        code = human_entry["code_snippet"]

        # --- HUMANO ---
        human_metrics = evaluate_single_review(
            human_entry["human_review"], code, entry_id
        )
        issue_distribution["human"].update(human_metrics["issue_types"])

        # --- CHATGPT ---
        gpt_metrics = evaluate_single_review(
            llm_entry["llm_reviews"]["chatgpt"]["text"], code, entry_id
        )
        issue_distribution["chatgpt"].update(gpt_metrics["issue_types"])

        # --- CLAUDE ---
        claude_metrics = evaluate_single_review(
            llm_entry["llm_reviews"]["claude"]["text"], code, entry_id
        )
        issue_distribution["claude"].update(claude_metrics["issue_types"])

        evaluations.append({
            "id": entry_id,
            "metadata": {
                "repo": human_entry["repo"],
                "pull_request": human_entry["pull_request"],
                "commit": human_entry["commit"],
                "file": human_entry["file"],
                "start_line": human_entry["start_line"],
                "end_line": human_entry["end_line"]
            },
            "evaluations": {
                "human": human_metrics,
                "chatgpt": gpt_metrics,
                "claude": claude_metrics
            }
        })

        for source, metrics in [
            ("human", human_metrics),
            ("chatgpt", gpt_metrics),
            ("claude", claude_metrics)
        ]:
            row = {
                "id": entry_id,
                "source": source
            }
            for k, v in metrics.items():
                if k not in ["id", "issue_types"]:
                    row[k] = v
            table_rows.append(row)

    return evaluations, table_rows, issue_distribution

# rodar

if __name__ == "__main__":
    evaluations, table_rows, issue_dist = evaluate_all_reviews(
        "data/human_reviews.json",
        "data/enriched_reviews.json"
    )

    # gerar um json com as avaliacoes
    with open("data/review_evaluations.json", "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)

    # gerar um csv para comparação
    import csv
    with open("data/review_metrics_table.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
        writer.writeheader()
        writer.writerows(table_rows)

    print("Evaluation complete.\n")

    for source, counter in issue_dist.items():
        print(f"Issue distribution ({source}):")
        for issue, count in counter.items():
            print(f"  {issue}: {count}")
        print()

