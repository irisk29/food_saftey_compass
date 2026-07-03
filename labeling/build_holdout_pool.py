"""
Build a pool of Yelp reviews that were NEVER used anywhere in this project's
pipeline, pre-filtered to be allergy/food-safety-adjacent so the gold set drawn
from them has a real chance of containing hazards (see labeling/create_gold_dataset.py
for the labeling step). This is the "unseen holdout" data source discussed with
the user: the enriched 7,500-row dataset was built by scanning the raw Yelp
review dump and stopping early once its MAX_HAZARDS=1500 / MAX_BENIGN=6000 quotas
were full (see preprocessing/final_project_preprocessing.ipynb) — so the vast
majority of matching reviews in the raw dump were never even looked at.

How "never seen" is guaranteed
-------------------------------
The pipeline keeps no Yelp review_id anywhere downstream, so text is the only
join key available. The preprocessing notebook normalizes text before saving it
(`text.replace("\\n"," ").replace("\\t"," ")` then whitespace-collapsed + stripped)
— this script applies the *exact same normalization* to every raw candidate
before checking it against the set of every text already in
postprocessing/enriched_allergy_hazard_dataset.csv (which is a strict superset of
whatever ended up in the train/test split, since that split is carved out of this
same CSV). Anything that normalizes to an existing entry is dropped.

Usage:
    python labeling/build_holdout_pool.py --target 3000

Output: labeling/holdout_candidate_pool.csv — unlabeled candidates (no `is_hazard`
column; these were never scored by the keyword+stars heuristic in the first
place). Feed this into create_gold_dataset.py with --source holdout to sample
and LLM-label them.
"""

import argparse
import json
import os
import re
import sys
import zipfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as cfg

LABELING_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(LABELING_DIR, "holdout_candidate_pool.csv")

REVIEW_JSON_PATH = os.path.join(cfg.PROJECT_ROOT, "pre-choosing-dataset", "yelp_academic_dataset_review.json")
BUSINESS_ZIP_PATH = os.path.join(cfg.PROJECT_ROOT, "..", "yelp-dataset.zip")
BUSINESS_JSON_NAME = "yelp_academic_dataset_business.json"

# Two-tier keyword filter, tuned for "food allergy concern" specifically.
#
# A first pass using every allergy-adjacent word as one flat OR (allergy, celiac,
# gluten-free, sick, hospital, "er", ...) against a 100K-line sample pulled ~1%
# of reviews, but ~45% of those were just "they have a gluten-free menu" — a
# neutral factual note, not a hazard — and "sick"/"hospital"/"er" alone pulled in
# unrelated reviews (dark humor, "sick of the wait", "hospital-themed decor").
#
# STRONG terms describe an actual adverse event and are kept regardless of the
# star rating (someone can still give 3-4 stars after describing a reaction).
# WEAK terms are contextual/neutral on their own (a bare "allergy" or
# "gluten-free" mention) and are only kept alongside a low rating, matching the
# same keyword+stars logic the original heuristic label uses.
STRONG_TERMS = re.compile(
    r"\b(?:anaphylact\w*|epi.?pen|cross.?contaminat\w*|glutened|hives|swelling|"
    r"swollen throat|threw up|vomit\w*|food.?poison\w*|contaminat\w*|"
    r"allergic reaction|sent (?:me|us) to the hospital|went to the hospital|"
    r"ended up in (?:the )?hospital)\b",
    re.IGNORECASE,
)
WEAK_TERMS = re.compile(
    r"\b(?:allerg\w*|celiac|coeliac|gluten.?free|lactose|"
    r"nut allergy|shellfish allergy|dairy allergy)\b",
    re.IGNORECASE,
)
WEAK_TERM_MAX_STARS = 3  # same severity gate the original heuristic label uses


def normalize_text(text):
    """Must match preprocessing/final_project_preprocessing.ipynb exactly."""
    cleaned = text.replace("\n", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", cleaned).strip()


def load_eligible_business_ids():
    """Restaurants/Food businesses — same rule as the original preprocessing notebook."""
    print("Loading eligible business IDs from business.json (inside yelp-dataset.zip)...")
    eligible = set()
    zip_path = os.path.abspath(BUSINESS_ZIP_PATH)
    if not os.path.exists(zip_path):
        sys.exit(f"Could not find {zip_path} — needed to filter to restaurant/food businesses.")

    with zipfile.ZipFile(zip_path) as zf, zf.open(BUSINESS_JSON_NAME) as f:
        for line in f:
            biz = json.loads(line)
            categories = biz.get("categories")
            if categories and ("Restaurants" in categories or "Food" in categories):
                eligible.add(biz["business_id"])
    print(f"  {len(eligible):,} eligible business IDs.")
    return eligible


def load_seen_text_set():
    """Every normalized text already present in the enriched dataset (superset of train+test)."""
    df = pd.read_csv(cfg.INPUT_DATA_PATH, usecols=[cfg.TEXT_COLUMN])
    seen = set(df[cfg.TEXT_COLUMN].fillna("").map(normalize_text))
    print(f"Loaded {len(seen):,} already-seen review texts to exclude.")
    return seen


def build_pool(target, chunk_size, max_lines):
    eligible_business_ids = load_eligible_business_ids()
    seen_texts = load_seen_text_set()

    if not os.path.exists(REVIEW_JSON_PATH):
        sys.exit(f"Could not find {REVIEW_JSON_PATH}")

    candidates = []
    lines_scanned = 0
    print(f"Scanning {REVIEW_JSON_PATH} for fresh allergy/hazard-adjacent candidates "
          f"(target={target})...")

    with open(REVIEW_JSON_PATH, "r", encoding="utf-8") as f:
        for chunk in pd.read_json(f, lines=True, chunksize=chunk_size):
            lines_scanned += len(chunk)

            chunk = chunk[chunk["business_id"].isin(eligible_business_ids)]
            if chunk.empty:
                _progress(lines_scanned, len(candidates))
                if max_lines and lines_scanned >= max_lines:
                    break
                continue

            chunk = chunk.copy()
            chunk["text"] = chunk["text"].fillna("").map(normalize_text)
            word_count = chunk["text"].str.split().str.len()
            chunk = chunk[(word_count >= 5) & (word_count <= 800)]

            # The core "never seen" guarantee: drop anything already in the enriched CSV.
            chunk = chunk[~chunk["text"].isin(seen_texts)]

            # Bias toward hazard-plausible reviews: a STRONG term at any rating,
            # or a WEAK/contextual term combined with a low rating.
            strong_mask = chunk["text"].str.contains(STRONG_TERMS, na=False)
            weak_mask = chunk["text"].str.contains(WEAK_TERMS, na=False) & (chunk["stars"] <= WEAK_TERM_MAX_STARS)
            chunk = chunk[strong_mask | weak_mask]

            if not chunk.empty:
                out = pd.DataFrame({
                    "review_id": chunk["review_id"],
                    "business_id": chunk["business_id"],
                    "stars": chunk["stars"],
                    "useful": chunk["useful"],
                    "funny": chunk["funny"],
                    "cool": chunk["cool"],
                    "text": chunk["text"],
                    "word_count": chunk["text"].str.split().str.len(),
                    "char_count": chunk["text"].str.len(),
                    "exclamation_count": chunk["text"].str.count("!"),
                })
                candidates.append(out)

            _progress(lines_scanned, sum(len(c) for c in candidates))

            if len(candidates) and sum(len(c) for c in candidates) >= target:
                print(f"\nReached target of {target} candidates after {lines_scanned:,} lines scanned.")
                break
            if max_lines and lines_scanned >= max_lines:
                print(f"\nHit --max-lines={max_lines} cap.")
                break

    if not candidates:
        sys.exit("No candidates found — try loosening ALLERGY_KEYWORDS or raising --max-lines.")

    return pd.concat(candidates, ignore_index=True)


def _progress(lines_scanned, n_found):
    print(f"  scanned {lines_scanned:,} lines | fresh candidates found: {n_found:,}", end="\r")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", type=int, default=3000,
                         help="Stop once this many fresh candidates are collected (default 3000)")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-lines", type=int, default=0,
                         help="Optional hard cap on lines scanned (0 = scan until --target is hit)")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args()

    pool = build_pool(args.target, args.chunk_size, args.max_lines)
    pool = pool.drop_duplicates(subset="review_id")
    pool.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print(f"Holdout candidate pool written to: {args.out}")
    print(f"Candidates: {len(pool):,} (none overlap with the {cfg.INPUT_DATA_PATH} dataset)")
    print(f"Star rating distribution:\n{pool['stars'].value_counts().sort_index()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
