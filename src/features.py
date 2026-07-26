"""
Feature enrichment, extracted from postprocessing/final_project_postprocessing.ipynb.

The notebook was the only place these three features were defined, which meant the
gold holdout set — scraped fresh from the raw Yelp dump — could not be scored by the
baseline pipeline at all. Lifting the definitions into an importable module keeps the
holdout enrichment byte-identical to the training enrichment; any drift between the
two would silently corrupt the very comparison the holdout set exists to make.

Kept deliberately identical to the notebook, including its quirks:
  - `medical_lexicon_density` splits on whitespace without stripping punctuation, so
    "hospital." does not match "hospital". Reproducing the quirk matters more than
    fixing it, since the training data carries the same behaviour.
  - `negation_window_flag` includes "free" among its negation tokens, which is what
    catches "gluten free" as a negated allergy mention.

Note both of these are excluded from cfg.TABULAR_FEATURES as label-leaking (they are
built from the same keyword list as the heuristic label); they are computed here only
so the enriched schema matches end to end.
"""

import re

import pandas as pd

MEDICAL_TERMS = {
    "hospital", "er", "doctor", "ambulance", "hives",
    "swelled", "vomit", "breathing", "emergency", "epipen",
}

NEGATION_TOKENS = {"no", "not", "didnt", "didnot", "never", "wasnt", "without", "free"}
ALLERGY_TOKENS = {"allergy", "allergic", "celiac", "nuts", "peanuts", "gluten", "dairy"}

_vader = None


def _get_vader():
    """Lazily construct VADER so importing this module doesn't require the lexicon."""
    global _vader
    if _vader is None:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        try:
            _vader = SentimentIntensityAnalyzer()
        except LookupError:
            nltk.download("vader_lexicon")
            _vader = SentimentIntensityAnalyzer()
    return _vader


def medical_lexicon_density(text):
    if not isinstance(text, str):
        return 0.0
    tokens = text.lower().split()
    if not tokens:
        return 0.0
    return float(sum(1 for t in tokens if t in MEDICAL_TERMS) / len(tokens))


def vader_neg_intensity(text):
    if not isinstance(text, str):
        return 0.0
    return float(_get_vader().polarity_scores(text)["neg"])


def negation_window_flag(text):
    """1 if a negation token is followed within 3 tokens by an allergy token."""
    if not isinstance(text, str):
        return 0
    tokens = re.sub(r"[^a-zA-Z\s]", "", text.lower()).split()
    for i, tok in enumerate(tokens):
        if tok in NEGATION_TOKENS:
            if any(t in ALLERGY_TOKENS for t in tokens[i + 1: i + 4]):
                return 1
    return 0


def enrich(df, text_column="text"):
    """
    Adds every derived column the modelling code expects, in place on a copy.
    Existing columns are left untouched so already-enriched frames pass through.
    """
    out = df.copy()
    text = out[text_column].fillna("")

    if "word_count" not in out:
        out["word_count"] = text.str.split().str.len()
    if "char_count" not in out:
        out["char_count"] = text.str.len()
    if "exclamation_count" not in out:
        out["exclamation_count"] = text.str.count("!")
    if "medical_lexicon_density" not in out:
        out["medical_lexicon_density"] = text.map(medical_lexicon_density)
    if "vader_neg_intensity" not in out:
        out["vader_neg_intensity"] = text.map(vader_neg_intensity)
    if "negation_window_flag" not in out:
        out["negation_window_flag"] = text.map(negation_window_flag)

    for col in ("useful", "funny", "cool"):
        if col not in out:
            out[col] = 0

    return out


def verify_against_training_data(enriched_path, sample=500, text_column="text"):
    """
    Sanity check: recomputing the features on the training CSV must reproduce the
    stored values. Guards against this module drifting from the notebook.
    """
    df = pd.read_csv(enriched_path).head(sample)
    recomputed = enrich(df.drop(columns=[
        "medical_lexicon_density", "vader_neg_intensity", "negation_window_flag",
        "word_count", "char_count", "exclamation_count",
    ]), text_column=text_column)

    report = {}
    for col in ("word_count", "char_count", "exclamation_count",
                "medical_lexicon_density", "vader_neg_intensity", "negation_window_flag"):
        report[col] = bool((recomputed[col].round(6) == df[col].round(6)).all())
    return report
