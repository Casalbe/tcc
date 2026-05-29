import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import requests

GITHUB_API_BASE = "https://api.github.com"


def load_dotenv(dotenv_path: Path) -> None:
    if not dotenv_path.exists() or not dotenv_path.is_file():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class CommitRow:
    commit_hash: str
    contains_bug: bool
    commit_message: str
    la: float
    ld: float

    @property
    def churn(self) -> float:
        return float(self.la + self.ld)


@dataclass(frozen=True)
class RepoSpec:
    slug: str  # owner/repo
    csv_path: Path

    @property
    def owner_repo(self) -> Tuple[str, str]:
        owner, repo = self.slug.split("/", 1)
        return owner, repo


def _parse_bool(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"true", "1", "t", "yes", "y"}


def _parse_float(value: str) -> float:
    try:
        return float((value or "").strip())
    except Exception:
        return 0.0


def _normalize_header(name: str) -> str:
    return (name or "").replace("\ufeff", "").strip().casefold()


def _pick_delimiter_from_header_line(header_line: str) -> Optional[str]:
    if not header_line:
        return None
    candidates = [";", ",", "\t", "|"]
    counts = {d: header_line.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None


def _iter_dialects(sample: str, header_line: str) -> Iterable[csv.Dialect]:
    try:
        yield csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return
    except Exception:
        pass

    delim = _pick_delimiter_from_header_line(header_line)
    if delim:

        class _D(csv.Dialect):
            delimiter = delim
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        yield _D

    for d in [";", ",", "\t", "|"]:

        class _D2(csv.Dialect):
            delimiter = d
            quotechar = '"'
            doublequote = True
            skipinitialspace = False
            lineterminator = "\n"
            quoting = csv.QUOTE_MINIMAL

        yield _D2


def read_rows(csv_path: Path) -> List[CommitRow]:
    required = {"commit_hash", "contains_bug", "commit_message", "la", "ld"}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(65536)
        f.seek(0)
        header_line = f.readline()
        f.seek(0)

        last_headers: List[Optional[str]] = []
        last_missing: List[str] = []

        for dialect in _iter_dialects(sample, header_line):
            reader = csv.DictReader(f, dialect=dialect)
            raw_fieldnames = list(reader.fieldnames or [])
            normalized_to_raw: Dict[str, str] = {
                _normalize_header(h): h for h in raw_fieldnames if h is not None
            }

            missing = sorted(required - set(normalized_to_raw.keys()))
            if not missing:
                rows: List[CommitRow] = []
                for r in reader:
                    rr = {(_normalize_header(k) if k is not None else ""): v for k, v in r.items()}
                    rows.append(
                        CommitRow(
                            commit_hash=(rr.get("commit_hash") or "").strip(),
                            contains_bug=_parse_bool(rr.get("contains_bug") or ""),
                            commit_message=(rr.get("commit_message") or "").strip(),
                            la=_parse_float(rr.get("la") or "0"),
                            ld=_parse_float(rr.get("ld") or "0"),
                        )
                    )
                return rows

            last_headers = raw_fieldnames
            last_missing = missing
            f.seek(0)

        raise ValueError(
            "CSV missing required columns: "
            f"{last_missing}. First line: {header_line.rstrip()} Detected headers (last attempt): {last_headers}"
        )


def filter_by_max_churn(rows: Iterable[CommitRow], max_churn: float) -> List[CommitRow]:
    out: List[CommitRow] = []
    for r in rows:
        if not r.commit_hash:
            continue
        if r.churn >= max_churn:
            continue
        out.append(r)
    return out


def build_headers(token: Optional[str]) -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "tcc-req1-split-scraper",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _now_ts() -> str:
    return time.strftime("%H:%M:%S")


def log(msg: str) -> None:
    print(f"[{_now_ts()}] {msg}", flush=True)


def _sleep_until_reset(resp: requests.Response, *, extra_s: int = 2) -> None:
    reset = resp.headers.get("X-RateLimit-Reset")
    remaining = resp.headers.get("X-RateLimit-Remaining")
    if reset and reset.isdigit():
        reset_epoch = int(reset)
        sleep_s = max(0, reset_epoch - int(time.time()) + extra_s)
        log(
            f"Rate limit hit (remaining={remaining}). Sleeping for {sleep_s}s until reset..."
        )
        time.sleep(sleep_s)
    else:
        log("Rate limit hit and reset time unknown. Sleeping 60s...")
        time.sleep(60)


def fetch_commit_diff(
    owner: str,
    repo: str,
    sha: str,
    headers: Dict[str, str],
    *,
    timeout_s: int = 30,
    min_delay_s: float = 0.0,
) -> str:
    if min_delay_s > 0:
        time.sleep(min_delay_s)

    url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/commits/{sha}"
    diff_headers = dict(headers)
    diff_headers["Accept"] = "application/vnd.github.v3.diff"

    resp = requests.get(url, headers=diff_headers, timeout=timeout_s)

    # Basic rate-limit handling.
    if resp.status_code == 403 and (
        "rate limit" in (resp.text or "").lower() or resp.headers.get("X-RateLimit-Remaining") == "0"
    ):
        _sleep_until_reset(resp)
        resp = requests.get(url, headers=diff_headers, timeout=timeout_s)

    if resp.status_code == 404:
        raise RuntimeError(f"Commit not found: {owner}/{repo}@{sha}")
    if resp.status_code == 401:
        raise RuntimeError("GitHub API returned 401 Unauthorized. Check your GITHUB_TOKEN.")

    resp.raise_for_status()
    return resp.text


def load_excluded_commit_hashes(msgs_and_diffs_dir: Path) -> Set[str]:
    excluded: Set[str] = set()
    if not msgs_and_diffs_dir.exists():
        return excluded

    for p in sorted(msgs_and_diffs_dir.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            continue

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    sha = str(item.get("commit_hash") or "").strip()
                    if sha:
                        excluded.add(sha)
        elif isinstance(data, dict):
            sha = str(data.get("commit_hash") or "").strip()
            if sha:
                excluded.add(sha)

    return excluded


def _shuffle(rows: Sequence[CommitRow], rng: random.Random) -> List[CommitRow]:
    out = list(rows)
    rng.shuffle(out)
    return out


def _take_successes(
    *,
    repo: RepoSpec,
    candidates: Sequence[CommitRow],
    target: int,
    label_contains_bug: bool,
    split_name: str,
    headers: Dict[str, str],
    excluded_hashes: Set[str],
    used_hashes: Set[str],
    min_delay_s: float,
) -> List[Dict[str, Any]]:
    owner, repo_name = repo.owner_repo

    out: List[Dict[str, Any]] = []
    attempted = 0
    for r in candidates:
        if len(out) >= target:
            break

        sha = r.commit_hash
        if not sha:
            continue
        if sha in excluded_hashes or sha in used_hashes:
            continue

        attempted += 1
        try:
            diff = fetch_commit_diff(owner, repo_name, sha, headers, min_delay_s=min_delay_s)
        except Exception as e:
            log(f"Skip (error) {repo.slug}@{sha}: {e}")
            continue

        used_hashes.add(sha)
        out.append(
            {
                "commit_hash": sha,
                "repo": repo.slug,
                "commit_message": r.commit_message,
                "diff": diff,
                "contains_bug": bool(label_contains_bug),
                "la": r.la,
                "ld": r.ld,
                "churn": r.churn,
                "split": split_name,
            }
        )

        if len(out) % 25 == 0:
            log(f"{repo.slug} {split_name} {label_contains_bug=} collected {len(out)}/{target} (attempted {attempted})")

    return out


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Build train/test splits by scraping GitHub commit diffs from per-repo CSVs, "
            "with strict de-duplication across outputs and against existing msgs-and-diffs JSONs."
        )
    )

    p.add_argument(
        "--csv-dir",
        default=str(Path(__file__).resolve().parent / "data" / "csvs"),
        help="Folder containing *_rot_gpt.csv files (default: REQ1/data/csvs)",
    )
    p.add_argument(
        "--msgs-and-diffs-dir",
        default=str(Path(__file__).resolve().parent / "data" / "msgs and diffs"),
        help="Folder with existing JSONs to exclude duplicates from (default: REQ1/data/msgs and diffs)",
    )
    p.add_argument(
        "--splits-dir",
        default=str(Path(__file__).resolve().parent / "splits"),
        help="Root folder for output splits (default: REQ1/splits)",
    )
    p.add_argument(
        "--max-churn",
        type=float,
        default=50.0,
        help="Only include commits where la+ld < max-churn (default: 50)",
    )
    p.add_argument(
        "--per-repo",
        type=int,
        default=500,
        help="Target count per repo per group (default: 500)",
    )
    p.add_argument(
        "--strict-per-repo-defect",
        action="store_true",
        help=(
            "Require train/defect_inducing to reach --per-repo for every repo. "
            "If not set (default), missing defect-inducing samples for a repo are compensated by sampling more "
            "defect-inducing commits from other repos so the overall total still reaches per-repo*#repos."
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling candidates (default: 42)",
    )
    p.add_argument(
        "--token",
        default=None,
        help="GitHub token (default: env GITHUB_TOKEN, also loads repo-root .env if present)",
    )
    p.add_argument(
        "--min-delay",
        type=float,
        default=0.0,
        help="Minimum delay between requests in seconds (default: 0)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Only compute eligibility and duplicate-exclusion stats; do not call GitHub.",
    )

    args = p.parse_args()

    # Load .env from repo root (../.env)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    load_dotenv(repo_root / ".env")

    token = args.token or os.getenv("GITHUB_TOKEN")
    if token:
        log("GitHub token detected (authenticated requests)")
    else:
        log("No GitHub token found. This will not work for 7500 commits due to strict rate limits.")

    headers = build_headers(token)
    rng = random.Random(args.seed)

    csv_dir = Path(args.csv_dir)
    splits_dir = Path(args.splits_dir)

    repo_specs: List[RepoSpec] = [
        RepoSpec("ceejay66/camel", csv_dir / "camel_rot_gpt.csv"),
        RepoSpec("jboss-fuse/fabric8", csv_dir / "fabric8_rot_gpt.csv"),
        RepoSpec("bolabola/gimp", csv_dir / "gimp_rot_gpt.csv"),
        RepoSpec("subramani95/neutron", csv_dir / "neutron_rot_gpt.csv"),
        RepoSpec("krhancoc/postgres", csv_dir / "postgresql_rot_gpt.csv"),
    ]

    missing_csvs = [str(r.csv_path) for r in repo_specs if not r.csv_path.exists()]
    if missing_csvs:
        raise FileNotFoundError(
            "Missing required CSV files:\n" + "\n".join(missing_csvs)
        )

    excluded_hashes = load_excluded_commit_hashes(Path(args.msgs_and_diffs_dir))
    log(f"Loaded {len(excluded_hashes)} excluded commit hashes from msgs-and-diffs")

    # We ensure no duplicates across ANY output JSON.
    used_hashes: Set[str] = set()

    train_clean: List[Dict[str, Any]] = []
    train_defect: List[Dict[str, Any]] = []
    test_clean: List[Dict[str, Any]] = []

    # Pre-load & shuffle candidates per repo once so we can do a multi-pass selection strategy.
    repo_clean: Dict[str, List[CommitRow]] = {}
    repo_buggy: Dict[str, List[CommitRow]] = {}
    repo_stats: Dict[str, Dict[str, int]] = {}

    total_buggy_available = 0
    for repo in repo_specs:
        rows = read_rows(repo.csv_path)
        eligible = filter_by_max_churn(rows, args.max_churn)
        clean_rows = _shuffle([r for r in eligible if not r.contains_bug], rng)
        buggy_rows = _shuffle([r for r in eligible if r.contains_bug], rng)

        clean_available = sum(1 for r in clean_rows if r.commit_hash and r.commit_hash not in excluded_hashes)
        buggy_available = sum(1 for r in buggy_rows if r.commit_hash and r.commit_hash not in excluded_hashes)
        total_buggy_available += buggy_available

        repo_clean[repo.slug] = clean_rows
        repo_buggy[repo.slug] = buggy_rows
        repo_stats[repo.slug] = {
            "rows": len(rows),
            "eligible": len(eligible),
            "clean": len(clean_rows),
            "buggy": len(buggy_rows),
            "clean_available": clean_available,
            "buggy_available": buggy_available,
        }

        log(
            f"{repo.slug}: rows={len(rows)} eligible(churn<{args.max_churn})={len(eligible)} "
            f"clean={len(clean_rows)} buggy={len(buggy_rows)} "
            f"available_after_exclusion clean={clean_available} buggy={buggy_available}"
        )

    expected_total = args.per_repo * len(repo_specs)
    if args.dry_run:
        log(f"Total buggy available after exclusion (all repos): {total_buggy_available}")
        if total_buggy_available < expected_total:
            log(
                "WARNING: Not enough defect-inducing commits overall to reach the requested total "
                f"({total_buggy_available} < {expected_total})."
            )
        log("Dry-run complete (no scraping performed).")
        return

    # 1) Train clean (per-repo)
    for repo in repo_specs:
        log(f"{repo.slug}: collecting train/clean ({args.per_repo})")
        got_train_clean = _take_successes(
            repo=repo,
            candidates=repo_clean[repo.slug],
            target=args.per_repo,
            label_contains_bug=False,
            split_name="train",
            headers=headers,
            excluded_hashes=excluded_hashes,
            used_hashes=used_hashes,
            min_delay_s=args.min_delay,
        )
        if len(got_train_clean) < args.per_repo:
            raise RuntimeError(
                f"Not enough successful train clean commits for {repo.slug}: {len(got_train_clean)}/{args.per_repo}"
            )
        train_clean.extend(got_train_clean)

    # 2) Train defect_inducing
    defect_by_repo: Dict[str, int] = {}
    for repo in repo_specs:
        log(f"{repo.slug}: collecting train/defect_inducing (up to {args.per_repo})")
        got_train_defect = _take_successes(
            repo=repo,
            candidates=repo_buggy[repo.slug],
            target=args.per_repo,
            label_contains_bug=True,
            split_name="train",
            headers=headers,
            excluded_hashes=excluded_hashes,
            used_hashes=used_hashes,
            min_delay_s=args.min_delay,
        )
        defect_by_repo[repo.slug] = len(got_train_defect)
        if args.strict_per_repo_defect and len(got_train_defect) < args.per_repo:
            raise RuntimeError(
                f"Not enough successful train defect commits for {repo.slug}: {len(got_train_defect)}/{args.per_repo}"
            )
        train_defect.extend(got_train_defect)

    if len(train_defect) < expected_total:
        missing = expected_total - len(train_defect)
        if args.strict_per_repo_defect:
            raise RuntimeError(
                f"train_defect has {len(train_defect)}/{expected_total} and strict mode is enabled."
            )

        log(
            "Defect-inducing shortfall detected. "
            f"Collected {len(train_defect)}/{expected_total}; compensating remaining {missing} from other repos..."
        )

        remaining = missing
        # Round-robin across repos to avoid dumping everything into a single repo.
        while remaining > 0:
            progress = 0
            for repo in repo_specs:
                if remaining <= 0:
                    break
                got_extra = _take_successes(
                    repo=repo,
                    candidates=repo_buggy[repo.slug],
                    target=remaining,
                    label_contains_bug=True,
                    split_name="train",
                    headers=headers,
                    excluded_hashes=excluded_hashes,
                    used_hashes=used_hashes,
                    min_delay_s=args.min_delay,
                )
                if got_extra:
                    progress += len(got_extra)
                    defect_by_repo[repo.slug] = defect_by_repo.get(repo.slug, 0) + len(got_extra)
                    train_defect.extend(got_extra)
                    remaining -= len(got_extra)

            if progress == 0:
                raise RuntimeError(
                    "Unable to compensate remaining defect-inducing samples. "
                    f"Still missing {remaining} to reach {expected_total}."
                )

        log(
            "Compensation complete. Train defect-inducing per-repo counts: "
            + ", ".join(f"{k}={v}" for k, v in defect_by_repo.items())
        )

    # 3) Test clean (per-repo)
    for repo in repo_specs:
        log(f"{repo.slug}: collecting test/clean ({args.per_repo})")
        got_test_clean = _take_successes(
            repo=repo,
            candidates=repo_clean[repo.slug],
            target=args.per_repo,
            label_contains_bug=False,
            split_name="test",
            headers=headers,
            excluded_hashes=excluded_hashes,
            used_hashes=used_hashes,
            min_delay_s=args.min_delay,
        )
        if len(got_test_clean) < args.per_repo:
            raise RuntimeError(
                f"Not enough successful test clean commits for {repo.slug}: {len(got_test_clean)}/{args.per_repo}"
            )
        test_clean.extend(got_test_clean)

    # Final sanity checks.
    all_hashes = [
        *[x["commit_hash"] for x in train_clean],
        *[x["commit_hash"] for x in train_defect],
        *[x["commit_hash"] for x in test_clean],
    ]

    if len(all_hashes) != len(set(all_hashes)):
        raise RuntimeError("Duplicate commit_hash detected across output splits (should be impossible).")

    overlap_with_excluded = set(all_hashes) & excluded_hashes
    if overlap_with_excluded:
        raise RuntimeError(
            f"Output contains {len(overlap_with_excluded)} commit_hash values that exist in msgs-and-diffs exclusion set."
        )

    if len(train_clean) != expected_total:
        raise RuntimeError(f"train_clean wrong size: {len(train_clean)} expected {expected_total}")
    if len(train_defect) != expected_total:
        raise RuntimeError(f"train_defect wrong size: {len(train_defect)} expected {expected_total}")
    if len(test_clean) != expected_total:
        raise RuntimeError(f"test_clean wrong size: {len(test_clean)} expected {expected_total}")

    # Write outputs.
    train_dir = splits_dir / "train"
    test_dir = splits_dir / "test"

    _write_json(train_dir / "clean.json", train_clean)
    _write_json(train_dir / "defect_inducing.json", train_defect)
    _write_json(test_dir / "clean.json", test_clean)

    log(
        "Done. Wrote: "
        f"{train_dir / 'clean.json'} ({len(train_clean)}), "
        f"{train_dir / 'defect_inducing.json'} ({len(train_defect)}), "
        f"{test_dir / 'clean.json'} ({len(test_clean)})"
    )


if __name__ == "__main__":
    main()
