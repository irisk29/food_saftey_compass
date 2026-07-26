"""
Categorised failure-mode analysis.

The previous error analysis printed two false positives and two false negatives with
no synthesis, which shows *that* the model errs but never *why*. This module assigns
every error to a named linguistic failure mode, so the output is a distribution over
causes rather than an anecdote.

Rule-based rather than hand-coded, for three reasons: it scales to every error instead
of a sample of 30, it is reproducible when the model is retrained, and the rules
themselves are inspectable — a reader can disagree with a bucket definition, which
they cannot do with a human's private judgement. The intended workflow is to run this,
then manually verify a sample per bucket and report the agreement.

The same machinery runs against the *labels* (heuristic vs LLM judge), which is where
it is most valuable: it explains the 201 reviews the keyword rule over-flagged, and
that analysis needs no trained model at all.
"""

import os
import re

import pandas as pd

import config.settings as cfg

# --- False-positive modes: model (or heuristic) cried hazard, ground truth says no ---

NEUTRAL_ALLERGEN = re.compile(
    r"\b(?:gluten[\s-]?free|gf\s+(?:menu|option)|dairy[\s-]?free|nut[\s-]?free|"
    r"allergy[\s-]?friendly|allergen\s+menu|accommodat\w*\s+(?:my|our|his|her|their)?\s*allerg\w*|"
    r"(?:has|have|had|offers?|great)\s+(?:a\s+)?(?:gluten[\s-]?free|vegan|allergy)|"
    # "I am allergic to X" as a standing dietary fact, with no reaction described.
    r"(?:i'?m|i\s+am|she\s+is|he\s+is|we'?re|being)\s+allergic\s+to|"
    r"(?:my|her|his|their)\s+(?:\w+\s+)?allerg(?:y|ies))\b",
    re.IGNORECASE)

NEGATED_HAZARD = re.compile(
    r"\b(?:no|not|never|didn'?t|did\s+not|wasn'?t|weren'?t|without|avoided|luckily|thankfully)\b"
    r"[^.!?]{0,40}?\b(?:sick|ill|allergic|reaction|poison\w*|vomit\w*|hospital|contaminat\w*)\b",
    re.IGNORECASE)

HYPERBOLE = re.compile(
    r"\b(?:to\s+die\s+for|killer|deadly|sinful|dangerously\s+good|food\s+coma|"
    r"sick(?:ly)?\s+(?:good|delicious|amazing)|so\s+good\s+it'?s?\s+(?:criminal|illegal)|"
    r"heart\s+attack\s+on\s+a\s+plate|guilty\s+pleasure)\b",
    re.IGNORECASE)

# Deliberately narrow: an earlier version matched a bare "if you", which fires in most
# restaurant reviews and swallowed a third of all false positives into a meaningless
# bucket. Modality only counts here when it governs hazard vocabulary.
HYPOTHETICAL = re.compile(
    r"\b(?:i'?m\s+surprised\s+(?:i|we)\s+didn'?t|risk\s+of|afraid\s+(?:i|we)\s+(?:would|might|could)|"
    r"worried\s+(?:i|we)\s+(?:would|might|could)|hope\s+(?:i|we)\s+don'?t|"
    r"(?:would|could|might|may)\s+(?:have\s+)?(?:get|got|gotten|make|made)\s+"
    r"(?:me|us|you|someone)?\s*sick|"
    r"(?:probably|surely|bound\s+to)\s+(?:get|make)\s+\w*\s*sick|"
    r"waiting\s+to\s+(?:get\s+sick|happen))\b",
    re.IGNORECASE)

# Hazard vocabulary appears, but nothing ties it causally to eating at this restaurant:
# "drove across town to get food for a sick friend", "when I was sick I craved a burrito".
CAUSAL_LINK = re.compile(
    r"\b(?:after\s+(?:eating|the\s+meal|dinner|lunch|i\s+ate|we\s+ate)|"
    r"(?:got|gotten|made|make|makes|making)\s+(?:me|us|my|her|his|them|everyone)\s+\w*\s*sick|"
    r"food\s*poison\w*|from\s+(?:eating|the\s+food|this\s+place|their)|"
    r"gave\s+(?:me|us|him|her)\s+\w*\s*(?:food\s*poisoning|diarrhea)|"
    r"(?:sick|ill|vomit\w*|threw\s+up)\s+(?:all\s+)?(?:that\s+night|the\s+next\s+day|later|"
    r"within|hours?\s+(?:later|after))|"
    r"ended\s+up\s+in\s+(?:the\s+)?(?:hospital|er))\b",
    re.IGNORECASE)

ILLNESS_WORD = re.compile(r"\b(?:sick|ill|nausea\w*|vomit\w*|allerg\w*|poison\w*|hospital)\b",
                          re.IGNORECASE)

HEARSAY = re.compile(
    r"\b(?:(?:i|we)\s+(?:have\s+)?heard|other\s+reviews?|reviews?\s+(?:say|said|mention)|"
    r"someone\s+(?:said|told)|my\s+friend\s+said|apparently|rumou?r|health\s+(?:department|inspect\w*)\s+"
    r"(?:report|score|grade))\b",
    re.IGNORECASE)

DIRTY_NOT_ILL = re.compile(
    r"\b(?:dirty|filthy|gross|nasty|unsanitary|disgusting|grimy|sticky|smell\w*\s+bad)\b",
    re.IGNORECASE)

# --- False-negative modes: ground truth says hazard, model missed it ---

IMPLICIT_ILLNESS = re.compile(
    r"\b(?:spent\s+(?:the\s+)?(?:night|day|hours?)\s+in\s+the\s+bathroom|"
    r"(?:regretted|paid\s+for)\s+it\s+(?:later|all\s+night|the\s+next\s+day)|"
    r"couldn'?t\s+keep\s+(?:it|anything)\s+down|running\s+to\s+the\s+(?:bathroom|toilet)|"
    r"stomach\s+(?:was\s+)?(?:in\s+knots|turning|churning)|up\s+all\s+night|"
    r"worst\s+night\s+of\s+my\s+life|never\s+been\s+so\s+sick)\b",
    re.IGNORECASE)

MILD_WORDING = re.compile(
    r"\b(?:didn'?t\s+(?:agree|sit\s+right)\s+with\s+(?:me|us)|upset\s+stomach|"
    r"queasy|nauseous|nausea|indigestion|stomach\s+(?:ache|issues?|problems?|pain)|"
    r"felt\s+(?:off|weird|funny|unwell)|tummy|bathroom\s+(?:issues|trips))\b",
    re.IGNORECASE)

EXPLICIT_HAZARD = re.compile(
    r"\b(?:food\s*poison\w*|vomit\w*|threw\s+up|throwing\s+up|diarrhea|"
    r"anaphyla\w*|epi[\s-]?pen|hives|hospital|emergency\s+room|\ber\b|ambulance|"
    r"allergic\s+reaction|salmonella|e\.?\s?coli|norovirus|maggot|roach|"
    r"hair\s+in\s+(?:my|the)\s+food|glass\s+in\s+(?:my|the)\s+food)\b",
    re.IGNORECASE)

# Unsafe handling described directly, with no illness vocabulary — the keyword rule
# was built around allergy/illness terms, so these can slip past it.
UNSAFE_HANDLING = re.compile(
    r"\b(?:raw|undercooked|under\s+cooked|uncooked|spoiled|rotten|rancid|expired|"
    r"mold\w*|slimy|room\s+temperature|left\s+out|not\s+refrigerated|"
    r"(?:no|without|didn'?t\s+(?:wear|change|wash))\s+(?:gloves|hands)|"
    r"(?:proper\s+)?food\s+handling|hygien\w*|sanitation|bare\s+hands|"
    r"touch\w*\s+(?:the\s+)?food\s+with|same\s+(?:knife|board|gloves)|cross[\s-]?contaminat\w*)\b",
    re.IGNORECASE)

# Physical contamination, including the "someone else handled/ate it" cases.
CONTAMINATION = re.compile(
    r"\b(?:hair\s+in|bug|insect|roach|cockroach|fly\s+in|maggot|worm|"
    r"piece\s+of\s+(?:glass|plastic|metal)|foreign\s+object|"
    r"partly\s+eaten|bite\s+(?:taken|out\s+of)|someone\s+(?:had\s+)?(?:eaten|bitten)|"
    r"finger\s*nail|band[\s-]?aid|spit\s+in)\b",
    re.IGNORECASE)

_LATE_MENTION_FRACTION = 0.60   # hazard first appears in the final 40% of the review
_LONG_REVIEW_WORDS = 150
_SHORT_REVIEW_WORDS = 25
_HIGH_STARS = 4


def _first_hazard_position(text):
    m = EXPLICIT_HAZARD.search(text)
    return m.start() / max(len(text), 1) if m else None


def categorize_false_positive(text, stars=None, vader_neg=None):
    """
    All matching FP failure modes, ordered most-specific first.

    Ordering matters: the first entry becomes the reported primary mode, so the
    diagnostic buckets must precede the broad ones. `generic_complaint_no_hazard`
    is last on purpose — it is the "nothing hazard-like here at all" residual, and
    keeping it distinct from `unexplained_fp` matters, because the two demand
    different responses (expected behaviour vs. go read these by hand).
    """
    text = str(text)
    has_explicit = bool(EXPLICIT_HAZARD.search(text))
    modes = []

    if NEUTRAL_ALLERGEN.search(text) and not has_explicit:
        modes.append("neutral_allergen_mention")
    if NEGATED_HAZARD.search(text):
        modes.append("negated_hazard")
    if HYPERBOLE.search(text):
        modes.append("hyperbole_or_slang")
    if HYPOTHETICAL.search(text) and not IMPLICIT_ILLNESS.search(text):
        modes.append("hypothetical_or_speculative")
    if HEARSAY.search(text):
        modes.append("secondhand_or_hearsay")
    # Illness vocabulary with no causal tie to this meal: "food for a sick friend".
    if ILLNESS_WORD.search(text) and not CAUSAL_LINK.search(text) and not has_explicit:
        modes.append("illness_mentioned_not_caused_here")
    if DIRTY_NOT_ILL.search(text) and not has_explicit:
        modes.append("unpleasant_not_unsafe")
    if vader_neg is not None and vader_neg >= 0.15 and not has_explicit:
        modes.append("strong_negative_sentiment_only")
    if not ILLNESS_WORD.search(text) and not has_explicit and not CONTAMINATION.search(text):
        modes.append("generic_complaint_no_hazard")

    return modes or ["unexplained_fp"]


def categorize_false_negative(text, stars=None, vader_neg=None):
    """All matching FN failure modes."""
    text = str(text)
    words = len(text.split())
    has_explicit = bool(EXPLICIT_HAZARD.search(text))
    modes = []

    if IMPLICIT_ILLNESS.search(text) and not has_explicit:
        modes.append("implicit_illness_no_keyword")
    if MILD_WORDING.search(text) and not has_explicit:
        modes.append("mild_understated_wording")
    # Hazard types the allergy/illness keyword list was never designed to catch.
    if UNSAFE_HANDLING.search(text) and not ILLNESS_WORD.search(text):
        modes.append("unsafe_handling_no_illness")
    if CONTAMINATION.search(text) and not ILLNESS_WORD.search(text):
        modes.append("contamination_no_illness")

    pos = _first_hazard_position(text)
    if pos is not None and pos >= _LATE_MENTION_FRACTION and words >= _LONG_REVIEW_WORDS:
        modes.append("buried_in_long_review")

    if stars is not None and stars >= _HIGH_STARS:
        modes.append("positive_review_with_hazard")
    if words <= _SHORT_REVIEW_WORDS:
        modes.append("too_short_weak_signal")
    if NEGATED_HAZARD.search(text):
        # A negation the model over-trusted: "not the first time I got sick here".
        modes.append("negation_misread")

    return modes or ["unexplained_fn"]


def _bucket_frame(df, kind, text_col, stars_col, vader_col):
    """One row per error with its primary and all matched modes."""
    fn = categorize_false_positive if kind == "FP" else categorize_false_negative
    rows = []
    for idx, r in df.iterrows():
        modes = fn(
            r[text_col],
            stars=r[stars_col] if stars_col and stars_col in r else None,
            vader_neg=r[vader_col] if vader_col and vader_col in r else None,
        )
        rows.append({
            "error_type": kind,
            "index": idx,
            "primary_mode": modes[0],
            "all_modes": "|".join(modes),
            "n_modes": len(modes),
            "stars": r.get(stars_col) if stars_col else None,
            "word_count": len(str(r[text_col]).split()),
            "prob": r.get("prob"),
            "text": str(r[text_col])[:400],
        })
    return pd.DataFrame(rows)


def analyze_errors(df, y_true, y_pred, probs=None, text_col=None,
                   stars_col="stars", vader_col="vader_neg_intensity",
                   label="model", output_dir=None, samples_per_bucket=3):
    """
    Buckets every FP and FN, writes a per-error CSV and a readable markdown summary.

    Returns (summary_df, per_error_df).
    """
    text_col = text_col or cfg.TEXT_COLUMN
    output_dir = output_dir or cfg.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    work = df.copy()
    work["_true"], work["_pred"] = list(y_true), list(y_pred)
    if probs is not None:
        work["prob"] = list(probs)

    fps = work[(work["_true"] == 0) & (work["_pred"] == 1)]
    fns = work[(work["_true"] == 1) & (work["_pred"] == 0)]

    frames = []
    if len(fps):
        frames.append(_bucket_frame(fps, "FP", text_col, stars_col, vader_col))
    if len(fns):
        frames.append(_bucket_frame(fns, "FN", text_col, stars_col, vader_col))

    if not frames:
        print(f"[{label}] no errors to analyse")
        return pd.DataFrame(), pd.DataFrame()

    per_error = pd.concat(frames, ignore_index=True)

    summary = (per_error.groupby(["error_type", "primary_mode"])
               .agg(count=("index", "size"),
                    mean_words=("word_count", "mean"),
                    mean_prob=("prob", "mean"))
               .reset_index())
    totals = summary.groupby("error_type")["count"].transform("sum")
    summary["share_of_error_type"] = (summary["count"] / totals).round(3)
    summary = summary.sort_values(["error_type", "count"], ascending=[True, False])

    per_error.to_csv(os.path.join(output_dir, f"error_analysis_{label}_detail.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, f"error_analysis_{label}_summary.csv"), index=False)

    _write_markdown(summary, per_error, label, output_dir, samples_per_bucket,
                    n_fp=len(fps), n_fn=len(fns), n_total=len(work))

    print(f"\n[{label}] {len(fps)} false positives, {len(fns)} false negatives")
    print(summary.to_string(index=False))

    return summary, per_error


_MODE_EXPLANATIONS = {
    "neutral_allergen_mention":
        "Allergen vocabulary used as a neutral factual note ('they have a gluten-free menu'). "
        "The model has learned allergen words predict hazards because the label was built from "
        "an allergen keyword list — this is the labelling rule leaking into the model.",
    "negated_hazard":
        "A hazard term inside a negation scope ('never got sick here'). Bag-of-words baselines "
        "cannot represent negation at all; a transformer can in principle, so residual errors "
        "here indicate the fine-tune did not have enough negated examples to learn it.",
    "hyperbole_or_slang":
        "Figurative language reusing hazard vocabulary ('to die for', 'killer tacos'). Purely "
        "lexical signal with inverted sentiment — the clearest case for contextual embeddings "
        "over TF-IDF.",
    "hypothetical_or_speculative":
        "Hazard raised as a possibility, not an event ('I'm surprised I didn't get sick'). "
        "Requires modality/irrealis detection, which neither model is trained for.",
    "secondhand_or_hearsay":
        "Hazard attributed to someone else or to other reviews. Needs source attribution, "
        "not just topic detection.",
    "unpleasant_not_unsafe":
        "Describes filth or disgust without an adverse event. The boundary is a genuine "
        "definitional question — arguably these deserve flagging in a real deployment.",
    "strong_negative_sentiment_only":
        "Highly negative review with no hazard content. The model is partly reading sentiment "
        "as hazard, unsurprising given the label used a star-rating gate.",
    "implicit_illness_no_keyword":
        "Illness described euphemistically ('spent the night in the bathroom') with no hazard "
        "keyword. These are invisible to the keyword heuristic by construction, so the training "
        "label taught the model to miss them — a ceiling imposed by the labelling method.",
    "mild_understated_wording":
        "Understated symptoms ('didn't sit right with me'). Same ceiling as above, milder.",
    "buried_in_long_review":
        "Hazard mentioned late in a long review, past the 256-token truncation window or "
        "diluted by surrounding content. Directly actionable: raise max_length.",
    "positive_review_with_hazard":
        "4-5 star review reporting a hazard. The star gate in the labelling rule means the "
        "training data barely contains these, so the model associates hazards with low ratings.",
    "too_short_weak_signal":
        "Very short review; little evidence either way.",
    "negation_misread":
        "Negation cue present but the review is genuinely a hazard ('not the first time I got "
        "sick here'). Over-application of the negation pattern.",
    "illness_mentioned_not_caused_here":
        "Illness vocabulary with no causal link to this meal — 'picking up food for a sick "
        "friend', 'I was sick that week so I craved soup'. The keyword rule cannot represent "
        "causation at all, only co-occurrence, so every one of these is guaranteed to be "
        "mislabelled. A contextual model should beat the label here, which means these cases "
        "are where the model looks *wrong* while actually being right.",
    "generic_complaint_no_hazard":
        "An ordinary bad review with no hazard vocabulary whatsoever. If the model flags these, "
        "it is reading general negativity as danger — check whether the star gate in the label "
        "taught it that.",
    "unsafe_handling_no_illness":
        "Unsafe practice described (raw, spoiled, bare hands, hygiene) without anyone falling "
        "ill. Genuinely a hazard, but the keyword list was built from allergy/illness terms, so "
        "the label misses a share of these — the model inherits the blind spot.",
    "contamination_no_illness":
        "A foreign object or tampering, without illness vocabulary. Same inherited blind spot "
        "as above; this is where the heuristic's recall is weakest (88.5% on contamination).",
    "unexplained_fp": "No rule matched — requires manual review.",
    "unexplained_fn": "No rule matched — requires manual review.",
}


def _write_markdown(summary, per_error, label, output_dir, samples_per_bucket,
                    n_fp, n_fn, n_total):
    lines = [
        f"# Error Analysis — {label}", "",
        f"{n_fp} false positives and {n_fn} false negatives out of {n_total} evaluated reviews.",
        "",
        "Failure modes are assigned by the rule set in `analysis/error_analysis.py`. Each error "
        "may match several modes; the table counts the highest-priority one. Buckets are "
        "reproducible by construction — verify a sample per bucket by hand and report the "
        "agreement rate rather than trusting them blind.",
        "",
    ]

    for kind, title in (("FP", "False positives — flagged a hazard that is not there"),
                        ("FN", "False negatives — missed a real hazard")):
        sub = summary[summary.error_type == kind]
        if sub.empty:
            continue
        lines += [f"## {title}", "", "| Failure mode | Count | Share | Mean words |", "|---|---:|---:|---:|"]
        for _, r in sub.iterrows():
            lines.append(f"| `{r['primary_mode']}` | {int(r['count'])} | "
                         f"{r['share_of_error_type']:.0%} | {r['mean_words']:.0f} |")
        lines.append("")

        for _, r in sub.iterrows():
            mode = r["primary_mode"]
            lines += [f"### `{mode}` — {int(r['count'])} cases", "",
                      _MODE_EXPLANATIONS.get(mode, ""), ""]
            examples = per_error[(per_error.error_type == kind) &
                                 (per_error.primary_mode == mode)].head(samples_per_bucket)
            for _, ex in examples.iterrows():
                prob = f" (p={ex['prob']:.3f})" if pd.notna(ex.get("prob")) else ""
                snippet = ex["text"].replace("\n", " ")[:280]
                lines.append(f"- {snippet}…{prob}")
            lines.append("")

    with open(os.path.join(output_dir, f"error_analysis_{label}.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def analyze_label_disagreement(output_dir=None):
    """
    Applies the same failure-mode taxonomy to the *labelling rule*, using the LLM judge
    as ground truth. Explains the 201 reviews the keyword+stars heuristic over-flagged
    and the 12 it missed.

    Needs no trained model, and the conclusions transfer directly: every systematic
    error in the label becomes a systematic error the model is trained to reproduce.
    """
    output_dir = output_dir or cfg.RESULTS_DIR
    gold = pd.read_csv(cfg.GOLD_INSIDE_PATH)

    summary, per_error = analyze_errors(
        gold,
        y_true=gold["llm_is_hazard"].astype(int),
        y_pred=gold[cfg.TARGET_COLUMN].astype(int),
        label="heuristic_label",
        output_dir=output_dir,
    )
    return summary, per_error


if __name__ == "__main__":
    analyze_label_disagreement()
