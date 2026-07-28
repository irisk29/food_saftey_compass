> ## ⚠️ SUPERSEDED — build from [`DECK.md`](DECK.md)
>
> **2026-07-28:** the full deck is written. `DECK.md` is a complete, speaker-ready
> 16-slide Marp deck with every number verified against a committed artifact, presenter
> notes, an 11-question Q&A back-pocket, a figure-export checklist and rehearsal timings.
>
> ```
> npx @marp-team/marp-cli@latest slides/DECK.md -o deck.pdf     # or -o deck.pptx
> npx @marp-team/marp-cli@latest -s slides/                     # live preview
> ```
>
> This skeleton is kept for its planning rationale only. Where the two disagree,
> **`DECK.md` wins** — notably Slide 8, whose guidance below is now out of date (see the
> note inline). Do not build slides from this file.

---

# Slide Skeleton — Food Safety Compass (≤15 min, presentations 2026-08-03)

*One slide ≈ 60–70 seconds. 13 content slides ≈ 14 min with a fast open and close.
Slides marked 🔶 PENDING wait on a data run — everything else can be built today.
Every number on a slide must trace to a file in `results/` or `labeling/` — the "Evidence" line says which.*

**Spine of the talk (say it in one breath):** *We built a hazard detector — and then spent the project proving our own ground truth wrong, twice, before trusting a single number.*

---

## Slide 1 — Title
- **Allergen & Food Safety Hazard Compass** — detecting health hazards hidden in Yelp reviews
- Iris Kronfeld · Tal Edrehy — Text Mining 2026 (Prof. Shay Palachy Affek, TA: Tal Cordova)
- Visual: one 4.5★ restaurant card with a buried 1★ "allergic reaction" review peeking out

## Slide 2 — The problem: aggregate rating masking
- A 4.5★ restaurant can hide a systemic kitchen failure — 98% love the ambiance, 2% go to the ER
- For allergy/celiac diners, star averages make dining an invisible gamble
- **Cost asymmetry drives everything downstream:** a missed hazard (FN) ≫ a false alarm (FP) → we optimize recall-weighted metrics, tuned thresholds, cost-sensitive loss
- Two research questions: (1) can we **detect** hazard reviews from raw text? (2) can we **discover hazard types** unsupervised?
- Source: reuse the Background prose from `final_work.docx` — it's the strongest writing in the project

## Slide 3 — Data & the heuristic label
- Yelp Academic Dataset → restaurants only, 5–800 words → streamed until 1,500 keyword-flagged + 6,000 benign = **7,500 rows** (natural hazard sparsity ~0.12%: 24 in the first 20,000)
- Heuristic label: safety keyword (allergy, celiac, contamination, …) **AND** ≤3★ → `is_hazard = 1`
- Engineered features: medical-lexicon density, VADER negativity, negation-window flag
- Closing beat (sets up the whole talk): *"a rule-based label is a hypothesis, not ground truth — so we tested it"*
- Evidence: preprocessing/postprocessing notebooks

## Slide 4 — Act I: we audited our own label
- Independent LLM judge re-labeled 1,500 rows blind → **85.8% agreement** — and the disagreement is one-sided
- 2×2 confusion table (heuristic vs LLM): **201/750 heuristic positives are benign (27% over-flagging)**; only 12/750 missed
- The rule's 97.9% "recall" is circular — measured inside its own keyword filter
- Evidence: `labeling/gold_dataset.csv`, agreement numbers in CLAUDE.md / IMPLEMENTATION_NOTES

## Slide 5 — *Why* the rule over-flags (error taxonomy)
- Bucketed all 201 FPs + 12 FNs by cause — top buckets with one real quote each:
  - **48% — illness mentioned, not caused here** ("haven't gotten sick from them in 10 years") → keyword ≠ causation
  - **12% — negated hazard** ("nothing to trigger my allergy")
  - remaining buckets: menu-mention-only ("gluten-free options"), general negativity, …
- Takeaway line: *co-occurrence is not causation — the exact failure bag-of-words methods inherit*
- Evidence: `results/error_analysis_heuristic_label.md` + summary CSV

## Slide 6 — Act II: our validation set was contaminated too
- First gold set was sampled from the training pool → **1,334 / 1,500 rows (89%) sat in the train split** — evaluating on it = measuring memorization
- Fix: built a candidate pool straight from raw Yelp, identical normalization, **verified 0/772 text overlap** with all training data (check runs in `verify_setup.py` preflight)
- Fresh LLM-labeled holdout: **772 rows, 355 hazards (46.0%)** — *within the keyword-screened funnel* (say this caveat out loud before someone asks; population rate ~2–5%)
- Evidence: `labeling/gold_dataset_holdout.csv`, `labeling/build_holdout_pool.py`

## Slide 7 — Technique 1: fine-tuned DeBERTa-v3 with cost-sensitive loss
- Text-only transformer (metadata deliberately withheld — no `stars` shortcut back to the label)
- Cost-sensitive loss **family**, compared honestly:
  - `pos_weight` — label-keyed ≡ `BCE(pos_weight=w)` (standard class weighting — we say so)
  - `focal_asymmetric` — weight grows with error: `1 + (w−1)(1−p)^γ`
  - `fn_gated` — penalty only when a true hazard scores below the deployed threshold τ
- Deployed threshold 0.20 (chosen from the FN≫FP cost model) + 9 unit tests incl. a degenerate-model regression guard
- Evidence: `src/losses.py`, `src/sota_model.py`, `tests/test_losses.py`

## Slide 8 — Act III: the metric that selects a broken model
- 🔄 **THIS SLIDE'S GUIDANCE IS OUT OF DATE — see `DECK.md` slide 9.** The Optuna sweep committed
  on 2026-07-28 (`results/optuna_trials.csv`) **produced a real collapse**: trial 1
  (lr=3.80e-05, bs=4) hit flag rate 1.000, recall 1.000, precision 0.200, PR-AUC 0.196. And bare
  recall would have ranked it **first of three**, while PR-AUC ranked it last by 5×. The slide is
  now a measurement, not an a-priori argument.
- ⚠️ Still true and still essential: **the collapse is NOT caused by w=50.** All three trials used
  `focal_asymmetric@50`, and the grid shows `collapsed=False` at w=50 in both formulations
  (`pos_weight@50` flag rate 0.197, recall 0.921, precision 0.936). Attribute it to the learning
  rate and batch size. The old "100% recall / 37.5% precision" figures remain unsourced — deleted.
- The a-priori argument still opens the slide: **bare recall is maximised by a constant all-positive
  answer**, so it cannot be a selection metric — and now we can show what that costs.
- Fix: checkpoint by **F2**, hyperparameter search by **PR-AUC**, log flag-rate as a collapse alarm — and `tests/test_losses.py` has a regression guard asserting an all-positive model is caught by F2/PR-AUC/flag-rate
- One-liner: *"any metric you can maximize with a constant answer is not a safety metric"*
- Evidence: `config/settings.py` (CHECKPOINT_METRIC/HPO_METRIC), `tests/test_losses.py`, `results/grid_search_loss_variants.csv`

## Slide 9 — 🔶 PENDING `analysis.py` — Results: same models, two ground truths
- THE payoff slide. Table: Baseline (TF-IDF+XGB, class-balanced) vs DeBERTa × {heuristic test labels, gold holdout} — PR-AUC leads, then F2 / precision / recall @ deployed threshold
- Expected shape: scores drop from heuristic → gold; baseline drops *more* (TF-IDF partially re-learns the keyword rule); **the delta quantifies how much the flawed label inflated results**
- Footnote on slide: heuristic test split was used for model selection → in-selection; the holdout is untouched
- Evidence (after run): `results/performance_*.csv`, `results/ground_truth_comparison.csv`
- ⚠️ Do not fill from `final_work.docx`'s old table — those are pre-fix numbers (see overview doc)

## Slide 10 — Technique 2: topic modeling for hazard-type discovery
- LDA + NMF over 1,500 flagged reviews, shared vocab, domain stopwords, K ∈ {2…10}, degenerate fits excluded
- Validated *externally* against LLM hazard types (545 docs) — purity vs majority baseline, NMI vs shuffle null
- 🔶 PENDING (1–2h): re-run after the NPMI co-occurrence fix — corrected story: **LDA coherence and validation independently agree on K=4**
- Disclose in one breath: K selected by the validation metric (mild circularity); ~27% of the fit corpus is benign per our own audit
- Evidence: `src/topic_model.py`, `results/topic_model_*.csv` (regenerate before quoting coherence)

## Slide 11 — What topic modeling can and cannot find
- ✅ **Lead with NMF** (deterministic `nndsvda` init, reproduces exactly). NMF K=6 isolates a crisp allergen/celiac topic — "gluten, celiac, cross contamination, gf" — **lift 5.28**, and **per-topic NPMI +0.43**, by far the most coherent topic found
- ⚠️ LDA is **environment-sensitive** (different fits across machines at the same seed) — quote LDA only from the committed artifact (K=4, lift 2.52), never the old 4.18, and say the reproducibility caveat out loud
- ❌ Neither subdivides the dominant food-poisoning mass (69% of validated docs); overall purity **0.697** (NMF K=6) vs 0.690 majority baseline — the old 0.716 has no surviving artifact
- The honest finding: *topic models find the **lexically distinct** rare hazard and fail on the **lexically diffuse** common one* — that's a property of the technique, not a tuning failure
- Evidence: `results/topic_model_type_lift.csv`, `topic_model_crosstabs.txt`

## Slide 12 — 🔶 PENDING `analysis.py` — Where the *model* still fails
- Same taxonomy from Slide 5 applied to DeBERTa's FP/FN on the gold holdout: counts per bucket + one quote each
- Compare failure profiles: rule fails on negation/causation; where does the transformer fail? (expect: subtle/implicit reports, mixed sentiment)
- Tie each bucket to a mechanism → this is the "why techniques fail" the course rewards
- Evidence (after run): `results/error_analysis_deberta_*.md`

## Slide 13 — Conclusions
- The methodology arc in five beats: built a label → audited it (27% over-flagging) → caught our contaminated validation set → caught our degenerate selection metric → only then reported numbers
- Detection: [one sentence from Slide 9's delta]. Discovery: allergen hazards separable unsupervised; food-poisoning mass is not
- Business answer: deploy as a **screening funnel** — threshold tuned to the FN≫FP cost model, human triage downstream
- Future work (10 sec): token-attribution XAI, ordinal risk tiers, cross-platform transfer

---

## Q&A back-pocket (not presented — rehearse answers)
1. "Is your asymmetric loss just class weighting?" → the `pos_weight` variant is, exactly, and we say so; `focal_asymmetric` is error-dependent. The grid compares those two across 4 weights (8 cells). **`fn_gated` is implemented and unit-tested but was never run** — say that, don't imply it was.
2. "46% hazard rate — really?" → funnel rate, 100% of holdout passes the keyword screen; strong-tier 71.5% vs weak-tier 21.4%; population ~2–5%.
3. "Why w=50 when your cost ratio implies 100:1?" → the weight was fixed before the grid ran. The grid then ranked two configs ahead of it on gold, by 0.008 (`pos_weight@1`) and 0.019 (`focal@15`) PR-AUC — both smaller than our measured run-to-run discrepancy (up to 0.054), so we report the grid rather than retrofit the deployment to it.
4. "Test set used for selection?" → no. Checkpoint selection ran on the **validation** split, so the heuristic test-split numbers are out-of-selection; the gold holdout never touched selection at all.
5. "Only 772 holdout rows?" → recall CI ±3–4 pts at n_pos=355; adequate, and labeling is resumable.
6. "Why no word/doc embeddings?" → chose topic modeling as technique #2 because it answers research question 2 (hazard-type discovery); embeddings answer neither question better than DeBERTa already does.
7. **"How stable are these numbers?"** → all runs use the HF default seed 42; re-training four configs under identical settings moved gold PR-AUC by 0.00005–0.054, so we treat gold gaps below ~0.05 as unresolved and report the weight-response *shape*, not a ranking. The deployed config was trained 3× across 2 code paths: 0.7961 / 0.8021 / 0.8045.
8. **"Why do your PR-AUC and cost columns disagree?"** → cost and F2 are computed at a fixed th=0.20 while PR-AUC is threshold-free, so a model that simply flags more (`focal@5` flags 75% of the holdout) looks cheap under a 100:1 ratio. We select on PR-AUC and would re-tune the threshold per model.
9. **"Your flag rate on gold is 66–75% but the base rate is 46% — isn't the model over-flagging?"** → yes, and knowably: the 0.20 threshold was tuned against the heuristic label and does not transfer to the gold distribution. Same models flag 19–21% on validation. One more instance of the central thesis.

## Build checklist
- [x] Run `analysis.py` → fills Slides 9, 12 (+ final numbers on 13)
- [x] NPMI fix + topic rerun → refresh coherence numbers on Slide 10
- [x] Full 8-cell loss grid, all gold-scored → rebuild Slide 8 and add the two new slides it earned
- [x] **Full deck written** → `DECK.md` (16 slides, all numbers artifact-verified, Q&A + notes)
- [x] **NEW Slide — the Optuna collapse:** bare recall would have selected the degenerate trial
- [ ] **NEW Slide — saturation:** validation spread 0.0030 across 8 configs vs gold spread 0.0992 (**33×**), and **Spearman ρ = −0.24** between the two ground truths. Line: *"we built an evaluation set because we could not tell our own models apart without one."*
- [ ] **NEW Slide — what the loss actually does:** gold PR-AUC vs w for both variants. `pos_weight` declines monotonically (0.8096 → 0.7956 → 0.7905 → 0.7219, range 0.088); `focal_asymmetric` is non-monotone in a band half as wide (range 0.043). Claim: **robustness to a hyperparameter, not peak performance.** Include the noise-floor caveat on the same slide.
- [ ] Figures to export: confusion 2×2 (S4), FP-bucket bar chart (S5), contamination diagram (S6), dual-ground-truth table (S9), lift table heat-strip (S11), weight-response curve (new)
- [ ] Rehearse to 14 min; Slides 4–6 are the differentiator — do not rush them
