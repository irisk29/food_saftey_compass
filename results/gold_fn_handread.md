# Hand-read of the 23 `unexplained_fn` gold false negatives

Source: `results/error_analysis_deberta_gold_llm_label_fresh_holdout_detail.csv` (DeBERTa-v3, deployed threshold 0.20, fresh LLM-labelled holdout of 772 reviews). That run has 39 false negatives; **23 of them (59%)** fell into the residual `unexplained_fn` bucket. The count is unchanged by the 2026-07-28 `NEGATED_HAZARD` widening — that fix moved four false *positives* only.

Every review below was read in full from `labeling/gold_dataset_holdout.csv` (the detail CSV stores only a 400-character excerpt). Modes were assigned by hand; each row carries a verbatim quote so the assignment is checkable.

## Why these were 'unexplained' in the first place

`unexplained_fn` does not mean 'no hazard vocabulary'. Reading them shows the opposite: 17 of 23 contain a word from `EXPLICIT_HAZARD` or `ILLNESS_WORD` (`vomit`, `threw up`, `roach`, `hair in`). The taxonomy's FN rules are all *conditioned on the absence* of such a word — `implicit_illness_no_keyword`, `mild_understated_wording`, `unsafe_handling_no_illness` and `contamination_no_illness` each require `not has_explicit` or `not ILLNESS_WORD.search(text)`. So a review with a clear hazard word that is short, low-starred and not negated matches nothing and falls through. The residual bucket is therefore a *taxonomy* gap, not a mystery in the data.

## (a) Named modes

| Mode | Count | Share of 23 | Share of all 39 FN | Mean model prob |
|---|---:|---:|---:|---:|
| `implicit_hazard` | 5 | 22% | 13% | 0.0019 |
| `second_hand_report` | 5 | 22% | 13% | 0.0292 |
| `label_questionable` | 5 | 22% | 13% | 0.0052 |
| `contamination_novel_phrasing` | 5 | 22% | 13% | 0.0195 |
| `explicit_hazard_missed` | 2 | 9% | 5% | 0.0010 |
| `mild_or_hedged` | 1 | 4% | 3% | 0.0014 |
| **total** | **23** | **100%** | **59%** | **0.0122** |

Secondary modes (a review may carry more than one):

| Secondary mode | Count |
|---|---:|
| `hazard_word_is_disgust_idiom` | 5 |
| `hazard_in_final_sentence` | 4 |
| `mild_or_hedged` | 1 |
| `exact_duplicate_of_24` | 1 |

- **`implicit_hazard`** — The hazard event is described with no hazard keyword the taxonomy or the labelling rule knows ('on the toilet for four hours', 'a little extra protein with 6 legs'). Invisible to a keyword label by construction.
- **`second_hand_report`** — The illness happened to a companion, not the reviewer ('my husband threw up'). Requires resolving who the experiencer is, not just detecting the topic.
- **`label_questionable`** — The LLM gold label is arguable: a slip-and-fall on a wet floor, a doneness mix-up, revulsion at a flavour, or someone else's vomit smelled on arrival. The model's low probability is defensible here.
- **`contamination_novel_phrasing`** — Genuine physical contamination, but phrased outside the CONTAMINATION / EXPLICIT_HAZARD vocabulary — 'hair baked into cheese topping', 'a rock from the pavement', 'a LONG hair attached'. The rule wants literally 'hair in the food'.
- **`explicit_hazard_missed`** — Unhedged, first-hand, early-in-review hazard language ('I threw up the whole night', 'Cockroaches crawling across the bar') that the model still scored near zero. No linguistic excuse — this is plain model error.
- **`mild_or_hedged`** — Hedged or explicitly non-realised symptom ('felt like I was gonna vomit but it didn't happen').
- **`hazard_word_is_disgust_idiom`** — The only hazard keyword present is figurative revulsion ('made me want to VOMIT'); the real hazard is described without keywords.
- **`hazard_in_final_sentence`** — The hazard statement is the last thing in the review, often an 'Update:'.

## Truncation check (`hazard_in_final_sentence`)

Tokenised with the model's own tokenizer, `microsoft/deberta-v3-base` (`DebertaV2TokenizerFast`, downloaded from the Hub — **not** a whitespace proxy), against the `max_length=256` used in `src/sota_model.py:107`. `hazard_cue_token_pos` is the token index of the first token of the quoted hazard cue, including the leading `[CLS]`.

- **1 of 23 (4%)** have their hazard cue at a token position beyond the 256-token window.
- 2 of 23 reviews are long enough to be truncated at all (>256 tokens); longest is 451 tokens (index 79).
- Median hazard-cue token position: 39; max 360.

**Truncation is not the story for this bucket.** Four reviews place the hazard in the literal final sentence (2 of them introduced by an explicit `Update:` — indices 104 and 463), which is a real *position* effect, but only 1 of the four sits past the truncation point. Raising `max_length` would recover at most 1 of these 23.

## (b) Per-review assignments

| idx | p(hazard) | ★ | words | tok | cue tok | >256 | mode | quote |
|---:|---:|---:|---:|---:|---:|:-:|---|---|
| 24 | 0.0032 | 1 | 130 | 153 | 36 | no | `implicit_hazard` | "Grease was floating on top of my water" |
| 56 | 0.0007 | 2 | 89 | 108 | 95 | no | `second_hand_report` | "my friend got a cake pop from here and threw up" |
| 79 | 0.0209 | 2 | 387 | 451 | 360 | yes | `label_questionable` | "water on the floor the entire time we were there, that my partner slipped on" |
| 104 | 0.0012 | 1 | 124 | 153 | 116 | no | `implicit_hazard` | "My stomach has been hurting me all night" |
| 159 | 0.0010 | 1 | 36 | 49 | 19 | no | `second_hand_report` | "my husband threw up before we left" |
| 249 | 0.0410 | 1 | 192 | 225 | 72 | no | `contamination_novel_phrasing` | "a 12" black hair baked into cheese topping" |
| 263 | 0.0059 | 2 | 181 | 204 | 54 | no | `contamination_novel_phrasing` | "the other was a rock from the pavement outside" |
| 264 | 0.0007 | 1 | 33 | 48 | 39 | no | `explicit_hazard_missed` | "Cockroaches crawling across the bar" |
| 319 | 0.0008 | 2 | 122 | 141 | 71 | no | `label_questionable` | "I almost threw up after tasting it" |
| 386 | 0.0010 | 1 | 47 | 59 | 32 | no | `implicit_hazard` | "my friends and I were on the toilet for approximately 4 hours after we left" |
| 394 | 0.0009 | 1 | 51 | 62 | 7 | no | `implicit_hazard` | "a little extra UNWANTED Protein that had 6 legs and antennas" |
| 405 | 0.0007 | 1 | 30 | 39 | 25 | no | `label_questionable` | "i took one lick of the soft serve adn wanted to vomit" |
| 414 | 0.0014 | 2 | 89 | 104 | 58 | no | `mild_or_hedged` | "I was having greasy burps and felt like i was gonna vomit" |
| 417 | 0.0024 | 1 | 61 | 79 | 7 | no | `contamination_novel_phrasing` | "there was hair in my taco" |
| 422 | 0.0026 | 3 | 222 | 277 | 130 | no | `label_questionable` | "We think the kitchen switched our meat" |
| 463 | 0.0883 | 1 | 98 | 138 | 120 | no | `second_hand_report` | "Son vomited his chicken sandwich two hours later" |
| 493 | 0.0032 | 1 | 130 | 153 | 36 | no | `implicit_hazard` | "Grease was floating on top of my water" |
| 496 | 0.0472 | 2 | 96 | 125 | 33 | no | `contamination_novel_phrasing` | "a curly strand of hair was right on top" |
| 588 | 0.0552 | 1 | 114 | 162 | 114 | no | `second_hand_report` | "My date actually did vomit" |
| 661 | 0.0008 | 2 | 58 | 71 | 20 | no | `label_questionable` | "the smell of fresh vomit greeted us" |
| 692 | 0.0006 | 1 | 47 | 57 | 1 | no | `second_hand_report` | "My gf threw up the pork chops" |
| 751 | 0.0007 | 1 | 57 | 78 | 43 | no | `contamination_novel_phrasing` | "with a LONG hair attached" |
| 766 | 0.0012 | 1 | 50 | 69 | 7 | no | `explicit_hazard_missed` | "I threw up the whole night" |

Machine-readable companion, including the LLM judge's `llm_rationale` for every row: `results/gold_fn_handread.csv`.

### `label_questionable` in the judge's own words

- **79** (p=0.0209, judge confidence *high*, type `contamination`) — judge rationale: "water on the floor caused a slip and fall incident" — review says: "water on the floor the entire time we were there, that my partner slipped on"
- **319** (p=0.0008, judge confidence *high*, type `food_poisoning`) — judge rationale: "Reviewer almost threw up after tasting the Ginza Roll." — review says: "I almost threw up after tasting it"
- **405** (p=0.0007, judge confidence *high*, type `food_poisoning`) — judge rationale: "Reviewer describes an immediate, severe reaction to eating the soft serve ice cream." — review says: "i took one lick of the soft serve adn wanted to vomit"
- **422** (p=0.0026, judge confidence *high*, type `unsafe_handling`) — judge rationale: "The reviewer suspects the kitchen switched their meat orders, indicating a potential issue with food handling." — review says: "We think the kitchen switched our meat"
- **661** (p=0.0008, judge confidence *high*, type `food_poisoning`) — judge rationale: "The reviewer mentions a smell of fresh vomit, indicating a possible foodborne illness." — review says: "the smell of fresh vomit greeted us"

## (c) Interpretation

All 23 previously `unexplained_fn` gold false negatives now carry a named mode, so the unexplained share of the 39 gold FNs drops from 59% to 0%. They split almost evenly four ways — 5 `implicit_hazard`, 5 `second_hand_report`, 5 `contamination_novel_phrasing` and 5 `label_questionable` — with 1 hedged case and 2 outright model misses. The dominant residual cause is *vocabulary*, not context length: 10 of 23 (43%) describe a real hazard in words the keyword labelling rule never contained — which is exactly the ceiling the weak label imposes on the model trained from it. The truncation hypothesis, tested with the model's own tokenizer, explains 1 of 23: raising `max_length` is not the fix. The genuinely encouraging finding is that 5 of 23 are arguable gold labels (a wet-floor slip hazard, a steak-doneness mix-up, revulsion at a flavour), so the model's true miss rate on the holdout is slightly better than the headline number, and only 2 of these 23 are inexcusable model errors (the remaining 16 of the 39 FNs keep the modes the rule set already assigned them and are not re-judged here).
