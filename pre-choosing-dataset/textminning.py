import io
import os
import subprocess
import sys
import zipfile
import pandas as pd

DATASET_DIR = r"e:\textminning"
ZIP_PATH = os.path.join(DATASET_DIR, "yelp-dataset.zip")
JSON_PATH = os.path.join(DATASET_DIR, "yelp_academic_dataset_review.json")
KAGGLE_DATASET = "yelp-dataset/yelp-dataset"
TARGET_FILE = "yelp_academic_dataset_review.json"

def ensure_download():
    if os.path.exists(ZIP_PATH) or os.path.exists(JSON_PATH):
        return
    print("Dataset not found. Attempting to download from Kaggle...")
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

    kaggle_json = os.path.join(os.path.expanduser("~"), ".kaggle", "kaggle.json")
    if not os.path.exists(kaggle_json):
        print(f"\n❌ Kaggle API key not found at {kaggle_json}")
        sys.exit(1)

    print(f"Downloading '{KAGGLE_DATASET}' (~5GB, may take several minutes)...")
    subprocess.check_call([
        sys.executable, "-m", "kaggle", "datasets", "download",
        "-d", KAGGLE_DATASET, "-p", DATASET_DIR
    ])

def open_reviews():
    """Return a file-like object for the reviews JSON, from zip or extracted file."""
    if os.path.exists(JSON_PATH):
        print(f"Reading from extracted file: {JSON_PATH}")
        return open(JSON_PATH, "r", encoding="utf-8"), None

    if os.path.exists(ZIP_PATH):
        print(f"Reading directly from zip (no extraction needed): {ZIP_PATH}")
        zf = zipfile.ZipFile(ZIP_PATH, "r")
        names = zf.namelist()
        match = next((n for n in names if TARGET_FILE in n), None)
        if not match:
            print(f"❌ '{TARGET_FILE}' not found in zip. Files: {names}")
            sys.exit(1)
        return io.TextIOWrapper(zf.open(match), encoding="utf-8"), zf

    print("❌ No dataset file found.")
    sys.exit(1)

ensure_download()

allergy_keywords = [
    r"\ballerg(y|ies|ic)\b",
    r"\bceliac\b",
    r"\bgluten.free\b",
    r"\bcross.contaminat\w*",
    r"\bcontaminat\w*\b",
    r"\banaphylact\w*\b",
    r"\bepi.?pen\b",
    r"\bfood.?poison\w*",
    r"\bgot sick\b",
    r"\bsent me to the hospital\b",
    r"\bwent to the hospital\b",
    r"\bended up in the hospital\b",
]
keyword_pattern = "|".join(allergy_keywords)

harm_keywords = [
    r"\bgot sick\b", r"\bfelt sick\b", r"\bstomach.?ache\b",
    r"\ballergic reaction\b", r"\bbroke out\b",
    r"\bsent me to the hospital\b", r"\bended up in the hospital\b",
    r"\bepi.?pen\b", r"\banaphylact\w*\b", r"\bcross.contaminat\w*",
    r"\bfood.?poison\w*",
]
harm_pattern = "|".join(harm_keywords)

total_reviews = 0
total_allergy_reviews = 0
dangerous_allergy_reviews = 0
sample_reviews = []

print("Scanning Yelp reviews... This might take a few minutes.")

fh, zf = open_reviews()
try:
    chunk_size = 100_000
    for chunk in pd.read_json(fh, lines=True, chunksize=chunk_size):
        total_reviews += len(chunk)
        text_lower = chunk["text"].str.lower()

        allergy_mask = text_lower.str.contains(keyword_pattern, na=False)
        allergy_chunk = chunk[allergy_mask]
        total_allergy_reviews += len(allergy_chunk)

        harm_mask = text_lower[allergy_mask].str.contains(harm_pattern, na=False)
        dangerous_chunk = allergy_chunk[
            (allergy_chunk["stars"] <= 2) | harm_mask
        ]
        dangerous_allergy_reviews += len(dangerous_chunk)

        if len(sample_reviews) < 50 and not dangerous_chunk.empty:
            sample_reviews.extend(dangerous_chunk["text"].head(10).tolist())

        print(f"  Scanned {total_reviews:,} reviews so far | allergy: {total_allergy_reviews:,} | dangerous: {dangerous_allergy_reviews:,}", end="\r")
finally:
    fh.close()
    if zf:
        zf.close()

print("\n\n" + "=" * 50)
print("--- FEASIBILITY RESULTS ---")
print(f"Total reviews scanned:                {total_reviews:>10,}")
print(f"Reviews mentioning allergy keywords:  {total_allergy_reviews:>10,}  ({100*total_allergy_reviews/total_reviews:.2f}%)")
print(f"High-risk reviews (neg stars + harm): {dangerous_allergy_reviews:>10,}")
print("=" * 50)

if dangerous_allergy_reviews >= 2000:
    print(f"✅ SUFFICIENT DATA — {dangerous_allergy_reviews:,} high-risk reviews. Project 1 is viable.")
else:
    print(f"❌ TOO SPARSE — only {dangerous_allergy_reviews:,} high-risk reviews. Consider Project 2.")

print("\n--- SAMPLE HIGH-RISK REVIEWS ---")
for i, sample in enumerate(sample_reviews[:3], 1):
    print(f"\n[Sample #{i}]:\n{sample[:500]}\n" + "-" * 40)
