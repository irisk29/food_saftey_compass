# Food Safety Compass — Instructor Review

*Reviewed against `TM25_6B - Final Project Instructions.pdf` (Shay Palachy Affek, Text Mining 2026, TAU). Submission deadline: 2026-08-02 midnight; presentations 2026-08-03.*

## Bottom line

**If you submitted today, this lands around 70–75/100** — a solid, well-engineered "Ciyun 70" project with real engineering maturity in places, but it is not yet demonstrating the **≥2 course text-mining techniques, compared, with variant experiments** that the 80–89 band requires. The good news: the two biggest gaps (technique count, leakage-safe evaluation) are things you've already started building infrastructure for — you just haven't closed the loop. Realistic target with the fixes below: **85–90**.

What's unusual — in a good way — for a course project: you *noticed* your own label-leakage risk, wrote it into `CLAUDE.md`, and then built dedicated tooling (`labeling/build_holdout_pool.py`, `labeling/create_gold_dataset.py`) to fix it with an LLM-as-judge gold set. That instinct — "is my ground truth actually trustworthy?" — is exactly what separates the 90s from the 70s in Appendix B. Right now it's unfinished plumbing; finish it and *foreground it* in your presentation, because most groups never think to do this at all.

---

## Scoring against the rubric (Appendix B / grade structure)

| Component | Weight | Current state | Est. |
|---|---|---|---|
| Problem definition & approach | 20% | Real problem, correctly framed as classification with a topic-modeling extension path. Weak spot: no written problem-framing doc (README is one line). | ~75% |
| Text-mining design & implementation | 40% | Only **2 techniques exist in code**, and one of them (TF-IDF+XGBoost) isn't really one of the 5 course categories — see below. This is the single biggest risk to your grade. | ~65% |
| Results analysis & evaluation | 25% | `analysis/evaluation_pipeline.py` is genuinely strong (PR-AUC, business cost curves, qualitative FP/FN). But it's currently evaluating against the same **noisy heuristic labels** used to train, and no results are saved to the repo yet. | ~70% |
| Presentation & summary | 15% | Not yet assessable — no slides in the repo. | n/a |

---

## 1. Problem definition & approach (20%)

**Working well:**
- Food-safety hazard detection in Yelp reviews is a legitimate, non-trivial applied problem — not an off-the-shelf Kaggle sentiment task (avoids the 55–65 trap explicitly called out in Appendix B).
- Framing as binary classification (`is_hazard`) with a stated extension into hazard-*type* discovery (topic modeling) is exactly the "which text-mining task is this" reasoning the instructions ask for in step 2.

**Gaps:**
- There is no written problem statement anywhere in the repo (`README.md` is literally `# food_saftey_compass`). The optional written document is worth points ("יכול לשפר את הציון") — even a one-page problem framing (business motivation: why false negatives are costly, why this needs *text* analysis and not just star ratings) would strengthen this section and can be reused almost verbatim in your presentation's opening.
- The dataset construction stops early once `MAX_HAZARDS=1500` / `MAX_BENIGN=6000` are filled (`preprocessing/final_project_preprocessing.ipynb`). Since Yelp's review JSON isn't randomly ordered, "first N matches" is a mild sampling bias worth one sentence of acknowledgment in the write-up — not a big deal, but graders notice when limitations are and aren't named.

---

## 2. Text-mining design & implementation (40%) — biggest risk

This is where the grade is actually won or lost, and it's currently thin. The instructions list five course techniques:

1. Word Embeddings
2. Topic Models
3. Document Embedding
4. Fitting & Fine-Tuning Supervised Classification Models
5. Generative Language Models

**What you have:**
- `src/baseline_model.py` — TF‑IDF (2,500 features) + tabular features → XGBoost. This is a fine, necessary *baseline*, but TF‑IDF‑as‑features‑into‑a‑classifier is closer to "standard tabular processing" than to any of the five listed techniques — and the instructions explicitly say not to invest depth there ("אין צורך להיכנס לעומק... עיבוד טבלאות"). Don't cut it — keep it as a baseline — but don't count on it as one of your "≥2 techniques."
- `src/sota_model.py` — DeBERTa‑v3‑base fine‑tuning with a **custom `AsymmetricSafetyLoss`** (50× weight on hazard-class false negatives) and a tuned 0.20 decision threshold. This *is* a legitimate, well-executed instance of technique #4, and the custom loss is the kind of "not just used it, but adapted it to the business problem" work that Appendix B rewards at the 80+ level — **if you explain it well in the write-up** (why 50×? why 0.20, not 0.5? what's the recall/precision trade-off you're accepting?).

**That's 1 real course technique, deeply done, plus 1 justified classical baseline.** The rubric wants **two techniques from the course list, compared, with variant experiments on each** (e.g., different LDA topic counts, different embedding window sizes). You don't have that yet.

**Recommendation — close this gap with Topic Modeling (LDA/NMF), not another embedding model:**
- It's the natural extension already named in your own `CLAUDE.md` ("Word/document embeddings and topic modeling... needed to hit the '≥2 techniques' band").
- You already have the ingredients to make it *rigorous* rather than decorative: `labeling/create_gold_dataset.py` has the LLM judge assign `hazard_type` ∈ {allergic_reaction, food_poisoning, contamination, unsafe_handling, none}. Run LDA/NMF on the hazard-flagged reviews and **check whether the discovered topics line up with those LLM-assigned hazard types.** That's a real, gradeable "does the technique recover a meaningful structure" experiment — exactly what page 6 of the instructions asks for ("check if discovered topics correspond to known categories").
- Run at least 2 topic-count (K) variants and justify the choice via coherence score — this single addition covers "technique #2" *and* the "variant experiments" rubric line in one move.

A cheaper alternative if time is short: Sentence-BERT / Doc2Vec document embeddings compared against TF-IDF for a "what does the classic representation miss vs. the semantic one" analysis (this is literally suggested project idea #1 in Appendix A) — faster to implement than topic modeling but less connected to your specific business problem (hazard-type discovery). Topic modeling is the stronger choice given what you've already built.

**A real methodological risk to fix regardless:** both your Optuna objective (`main.py`, `target_score = eval_metrics["eval_recall"]`) and the Trainer's `metric_for_best_model="recall"` (`src/sota_model.py`) optimize for **recall alone, with no precision floor**. A model that predicts "hazard" for every input scores perfect recall. Your asymmetric loss makes this degenerate outcome plausible, not just theoretical. Switch the selection metric to something that can't be gamed by a constant-positive model — F2-score (weights recall higher than precision, matching your stated business priority, but still penalizes an all-positive model), or PR-AUC. This is a one-line fix that removes a real "why does this number look too good" question in Q&A.

---

## 3. Results analysis & evaluation (25%)

**Working well:**
- `analysis/evaluation_pipeline.py` is the strongest part of the codebase: side-by-side baseline vs. SOTA comparison, PR-AUC, a business-cost curve (translating FN/FP into dollar liability — directly answers "מה המשמעות העסקית" from step 5), and pulls concrete FP/FN text samples for qualitative review. This is 80+-band work *if it makes it into the final write-up with actual numbers*.

**Gaps:**
- Nothing has been persisted back into the repo yet — no saved metrics, no committed run output, no filled-in `results/` artifacts. Right now this is all print-to-stdout / logged to a private W&B dashboard. The grader needs to see numbers in the submission.
- **This is the important one:** your evaluation currently runs against the same keyword+stars heuristic label used to train both models. Given the explicit leakage concern already documented in `CLAUDE.md`, a precision/recall number computed against that same noisy label doesn't tell you (or the grader) how good the model actually is — it partly tells you how well the model learned to reproduce the labeling *rule*. You are one step away from fixing this: finish running `labeling/create_gold_dataset.py` to completion, and **re-run `run_production_evaluation` with the gold-set labels as `y_true`** for at least the overlapping rows. Reporting "heuristic-label metrics vs. gold-label metrics" side by side would itself be a compelling piece of error analysis — showing the disagreement rate directly measures how much you were previously trusting a flawed proxy.
- Qualitative error analysis currently prints 2 FP + 2 FN samples with no synthesis. Bump this to a real "error analysis" pass: categorize a handful of FNs/FPs by cause (sarcasm, mixed sentiment, "gluten-free menu" false triggers, negation handled wrong, etc.) and connect each failure mode back to *why* the technique missed it — this is explicitly what separates 80–89 from 90–100 in Appendix B ("הבנה עמוקה של למה הטכניקות עובדות או נכשלות").

---

## 4. Presentation & summary (15%)

No slides exist yet in the repo, so this can't be scored. Structural reminder for when you build it (≤15 min, per the instructions):
1. Problem (with the business cost framing you already have in the cost-curve chart)
2. Text-mining solution (lead with the two techniques + why each is suited to the problem)
3. Results (baseline vs. SOTA vs. gold-label-validated numbers)
4. Challenges & conclusions — **explicitly narrate the label-leakage discovery and fix.** That story (found a flaw in your own ground truth, built tooling to independently validate it) is a stronger "conclusions" slide than any accuracy number, and directly demonstrates the "understanding, not effort" standard the instructions close on.

---

## Priority action list before 2026-08-02

1. **Finish the gold dataset** (`labeling/create_gold_dataset.py`) and re-run evaluation against it, not the heuristic label. This single step de-risks your entire Results section.
2. **Add topic modeling (LDA/NMF)** over hazard-type text, validated against the LLM-assigned `hazard_type` categories, with 2+ K variants compared. This gets you to "≥2 real course techniques, compared, with variants."
3. **Fix the recall-only tuning objective** → F2 or PR-AUC, in both `main.py`'s Optuna objective and `sota_model.py`'s `metric_for_best_model`.
4. **Persist results to the repo**: commit `results/model_performance_profile.csv` and the three generated figures from a real run, plus a short markdown results summary.
5. **Write the one-page problem/approach doc** (optional per the instructions, but cheap and directly improves the 20% problem-definition score).
6. Deepen the qualitative error analysis section with categorized failure modes, not just raw samples.

Everything else — the pipeline structure, the config-level leakage documentation, the custom asymmetric loss, the cost-based evaluation — is already above the median course project. The remaining gap is closing the technique count and making sure your reported numbers are measuring the real problem, not the labeling heuristic.
