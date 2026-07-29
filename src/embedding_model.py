"""
Document-embedding classifiers — the project's third course technique.

WHY THIS EXISTS
---------------
Until now the comparison ran lexical (TF-IDF + XGBoost) against contextual
(fine-tuned DeBERTa-v3) with nothing in between, and the embedding family was
explicitly declined in CLAUDE.md. That leaves the most interesting question
unasked: how much of the transformer's advantage comes from *dense distributed
representations* versus from *fine-tuning on this task*? A frozen document
embedding with a linear head isolates exactly that. It is the middle rung:

    sparse lexical  ->  frozen dense embedding  ->  fine-tuned contextual

DESIGN DECISIONS WORTH DEFENDING
--------------------------------
1. **Text only.** No tabular or lexicon features, unlike `src/baseline_model.py`,
   which gets TF-IDF *plus* the seven columns in `cfg.TABULAR_FEATURES`. This is
   deliberate: DeBERTa is text-only, so a text-only embedding model is the variant
   that is directly comparable to it. The handicap runs against the embedding model,
   which makes any win it takes more credible, and it means the baseline column in
   our tables is the *stronger* of the two possible baselines.

2. **Four representations, one head.** Every variant feeds the identical
   `LogisticRegression` (same C, same class weighting, same seed), so differences
   in the table are differences in the *representation* and nothing else. The
   TF-IDF-sparse variant is included as a control for precisely this reason: it
   shares the head with the embeddings but not the geometry, which separates
   "dense embeddings help" from "a linear head helps".

   | variant        | representation                                        | fitted on |
   |----------------|-------------------------------------------------------|-----------|
   | `tfidf_lr`     | TF-IDF, 2500 features (control, sparse lexical)       | train     |
   | `tfidf_lsa`    | TF-IDF -> TruncatedSVD 300d (LSA / latent semantics)  | train     |
   | `doc2vec_dbow` | gensim PV-DBOW, 300d, trained on our own corpus       | train     |
   | `minilm`       | frozen `all-MiniLM-L6-v2` sentence embeddings, 384d   | nothing   |

   `minilm` is a *pretrained* document embedding: nothing in it is fitted on our
   data at all, only the 384->1 linear head is. That makes it the cleanest possible
   transfer-learning contrast with DeBERTa, which starts from comparable pretraining
   and then updates every weight.

3. **Class balancing matches the baseline.** `src/baseline_model.py` passes
   `scale_pos_weight = n_neg / n_pos`. sklearn's `class_weight="balanced"` sets
   `w_c = n / (k * n_c)`, so `w_pos / w_neg` is exactly `n_neg / n_pos` — the same
   relative weighting, pinned by a test in `tests/test_embedding_model.py`.

4. **Metrics are not re-implemented.** `score_variant`, `COST_FALSE_NEGATIVE` and
   `COST_FALSE_POSITIVE` are imported from `analysis/evaluation_pipeline.py`, and
   errors are bucketed by the same `analyze_errors` taxonomy the other two models
   are reported under. A parallel metric implementation would make the comparison
   worthless even if every formula happened to agree.

5. **Fitted on the 64% train split only** — the same `train_df` the reported
   baseline and DeBERTa saw (`load_and_split_data(with_validation=True)`), so the
   test-split and gold numbers are directly comparable to the committed CSVs.
   `verify_no_text_overlap` re-checks zero overlap between everything the model was
   fitted on and both evaluation sets, using the same normalisation as
   `verify_setup.py`.

REPRODUCIBILITY — READ BEFORE QUOTING A DOC2VEC NUMBER
------------------------------------------------------
Doc2Vec is the only genuinely seed-sensitive component here, and gensim has three
non-obvious sources of non-determinism. All three are pinned, and the third was found
by measurement after the first two had been "fixed":

  * `workers > 1` makes training order non-deterministic. We force `workers=1`.
  * gensim seeds each word vector from `hashfxn(word)`, whose default is Python's
    built-in `hash`, which is **salted per process** unless `PYTHONHASHSEED` is set.
    Without intervention two runs in the same script agree while two runs in
    separate processes do not. We pass a deterministic `hashfxn` (adler32 over the
    UTF-8 bytes) so no environment variable is required.
  * `infer_vector`'s negative sampling draws from `model.random`, so it is
    order-dependent. We reset `model.random` to a fresh `RandomState(seed)` before
    every single inference.
  * **The one that the first two did not fix.** `Doc2Vec.infer_vector` initialises the
    new document vector with `pseudorandom_weak_vector(size, seed_string=...)`, and
    that function's signature is `(size, seed_string=None, hashfxn=hash)` — it uses
    the *module default* `hash`, and ignores the `hashfxn` given to the model. So the
    constructor argument that makes *training* reproducible has no effect on
    *inference*. Measured: the same fit in two separate interpreters produced
    bit-identical word vectors and different inferred vectors, with gold PR-AUC at
    0.6825 / 0.6833 / 0.6836 across three processes. `_deterministic_gensim_init()`
    scopes a patch over that function during fit and inference.

Only after the fourth item is `doc2vec_dbow` reproducible **across** processes, which
is what a reproducibility claim has to mean. Guarded by a test. Across *seeds* it is
of course not reproducible, so `--seeds 42 43 44` re-fits it and reports the observed
spread, in the same spirit as the project's 0.054 gold-PR-AUC noise floor for the
transformer.

RUN
---
    python -m src.embedding_model                    # all four variants, seed 42
    python -m src.embedding_model --seeds 42 43 44   # + Doc2Vec seed-spread study
    python -m src.embedding_model --variants minilm doc2vec_dbow

If `sentence-transformers` cannot be installed or `all-MiniLM-L6-v2` cannot be
downloaded, drop `minilm` from `--variants`: `tfidf_lsa` is the documented dense
fallback and is a legitimate document embedding (LSA) in its own right.
"""

import argparse
import os
import re
import sys
import zlib
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config.settings as cfg
from analysis.error_analysis import analyze_errors
from analysis.evaluation_pipeline import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    _format_table,
    score_variant,
)
from src.data_pipeline import load_and_split_data, load_gold_holdout

# --- Representation hyperparameters -------------------------------------------
# Held fixed across variants wherever the concept is shared, so the table compares
# representations rather than budgets. 2500 TF-IDF features matches
# src/baseline_model.py exactly; 300 dense dimensions is the conventional Doc2Vec /
# LSA size and is close enough to MiniLM's fixed 384 that capacity is not the story.
TFIDF_MAX_FEATURES = 2500
DENSE_DIM = 300
D2V_EPOCHS = 40
D2V_WINDOW = 5
D2V_MIN_COUNT = 2
D2V_NEGATIVE = 5
SBERT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 64

# Logistic-regression head. One C for every variant — see design decision 2. C=1.0
# is sklearn's default L2 strength and is deliberately NOT tuned: the project's own
# loss-variant grid showed validation PR-AUC spans only 0.0030 across eight models
# (results/grid_ground_truth_agreement.csv), so validation cannot resolve a
# regularisation choice here either. Tuning it on gold would be cheating. Validation
# PR-AUC is still recorded per variant as a diagnostic.
HEAD_C = 1.0
HEAD_MAX_ITER = 2000

VARIANTS = ("tfidf_lr", "tfidf_lsa", "doc2vec_dbow", "minilm")
SEED_SENSITIVE_VARIANTS = ("doc2vec_dbow",)

# Display names for the results tables. They carry the threshold the way the
# committed performance CSVs do, so rows from both can be concatenated.
DISPLAY = {
    "tfidf_lr": "TF-IDF + LogReg (text-only control)",
    "tfidf_lsa": "LSA 300d + LogReg",
    "doc2vec_dbow": "Doc2Vec PV-DBOW 300d + LogReg",
    "minilm": "MiniLM-L6-v2 frozen 384d + LogReg",
}


# -----------------------------------------------------------------------------
# Leakage guard
# -----------------------------------------------------------------------------
def _norm(text):
    """Identical normalisation to verify_setup.py, so the two checks agree."""
    return re.sub(r"\s+", " ", str(text).replace("\n", " ").replace("\t", " ")).strip()


def verify_no_text_overlap(fit_texts, eval_sets, strict_sets=("gold-holdout",),
                           strict=True):
    """
    Checks whether anything the model was fitted on reappears in an evaluation set.

    `fit_texts` must be every text that touched *any* fitted component — the
    vectoriser, the SVD, the Doc2Vec vocabulary and weights, and the logistic head.
    For this module that is exactly the train split, because no variant is fitted on
    test, validation or gold text (Doc2Vec infers held-out vectors from a frozen
    model rather than re-training with them, which is the whole reason `infer_vector`
    exists and the most likely place a leak would hide).

    Only `strict_sets` abort the run, and by default that is the gold holdout alone —
    the set every reported number depends on, and the one `verify_setup.py` section 4
    already guarantees at zero. The in-corpus splits are *reported but not fatal*, for
    a reason found while writing this module and worth stating rather than silencing:

        `postprocessing/enriched_allergy_hazard_dataset.csv` contains 3 duplicate
        review texts among its 7,500 rows (7,497 unique after whitespace
        normalisation), and one of those duplicate pairs straddles the train/test
        boundary. So 1 of 1,500 test-split rows shares its text with a training row.

    That is a property of the dataset, not of this module: the committed baseline and
    DeBERTa were trained on the identical split and carry the identical single
    contaminated test row. It moves a test-split metric by at most 1/1500 = 0.07%,
    far below anything the project reports, and it does not touch the gold holdout at
    all. Aborting on it would be theatre; hiding it would be worse.
    """
    seen = {_norm(t) for t in fit_texts}
    report = {}
    for name, texts in eval_sets.items():
        n = sum(1 for t in texts if _norm(t) in seen)
        report[name] = n
        is_strict = name in strict_sets
        msg = (f"    leakage check: {n} of {len(texts)} {name} rows appear in the "
               f"fitted corpus")
        print(msg if n == 0 else f"  [!]{msg}"
              + ("" if is_strict else "  (pre-existing duplicate text in the source CSV;"
                                      " identical for the baseline and DeBERTa)"))
        if n and is_strict and strict:
            raise RuntimeError(
                f"{n} {name} rows also appear in the corpus the embedding model was "
                f"fitted on — every metric on that set is invalid.")
    return report


# -----------------------------------------------------------------------------
# Representations. Each exposes fit(texts) -> self and transform(texts) -> ndarray.
# -----------------------------------------------------------------------------
class TfidfRepresentation:
    """Sparse lexical control. Same vectoriser settings as src/baseline_model.py."""

    name = "tfidf_lr"
    is_sparse = True

    def __init__(self, seed=cfg.RANDOM_STATE):
        self.seed = seed
        self.vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                   stop_words="english", lowercase=True)

    def fit(self, texts):
        self.vec.fit(list(texts))
        return self

    def transform(self, texts):
        return self.vec.transform(list(texts))

    def describe(self):
        return f"TF-IDF, {len(self.vec.vocabulary_)} features (sparse)"


class LsaRepresentation:
    """
    TF-IDF -> TruncatedSVD. Latent Semantic Analysis: a document embedding obtained
    by linear projection rather than by learning. It is the honest floor for "dense
    helps" — if a rotation of the same TF-IDF matrix matches a learned embedding,
    the learned embedding bought nothing.

    TruncatedSVD's randomised solver is seeded, so this is deterministic.
    """

    name = "tfidf_lsa"
    is_sparse = False

    def __init__(self, seed=cfg.RANDOM_STATE, n_components=DENSE_DIM):
        self.seed = seed
        self.vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES,
                                   stop_words="english", lowercase=True)
        self.svd = TruncatedSVD(n_components=n_components, random_state=seed,
                                algorithm="randomized", n_iter=10)

    def fit(self, texts):
        self.svd.fit(self.vec.fit_transform(list(texts)))
        return self

    def transform(self, texts):
        return self.svd.transform(self.vec.transform(list(texts)))

    def describe(self):
        return (f"LSA {self.svd.n_components}d over {len(self.vec.vocabulary_)} TF-IDF "
                f"features, explained variance {self.svd.explained_variance_ratio_.sum():.3f}")


def _stable_hash(word):
    """
    Deterministic replacement for gensim's default `hashfxn=hash`.

    Python salts `hash(str)` per process unless PYTHONHASHSEED is set, and gensim
    uses it to seed each word's initial vector — so the default makes Doc2Vec
    irreproducible across processes in a way that is invisible inside one process.
    adler32 over the encoded bytes is stable everywhere.
    """
    return zlib.adler32(word.encode("utf-8"))


class Doc2VecRepresentation:
    """
    PV-DBOW (`dm=0`) — the distributed-memory-free paragraph vector. Chosen over
    PV-DM because DBOW is the variant that behaves better on short documents and is
    what the Le & Mikolov follow-up literature recommends; our median review is 78
    words.

    Trained on the project's own corpus, which is the point (and the limitation):
    4,800 training reviews is one to three orders of magnitude smaller than the
    corpora paragraph vectors are normally learned on.
    """

    name = "doc2vec_dbow"
    is_sparse = False

    def __init__(self, seed=cfg.RANDOM_STATE, vector_size=DENSE_DIM,
                 epochs=D2V_EPOCHS, infer_epochs=None):
        self.seed = seed
        self.vector_size = vector_size
        self.epochs = epochs
        self.infer_epochs = infer_epochs or epochs
        self.model = None

    @staticmethod
    def tokenize(text):
        from gensim.utils import simple_preprocess
        return simple_preprocess(str(text), deacc=True)

    @staticmethod
    @contextmanager
    def _deterministic_gensim_init():
        """
        Forces gensim's vector initialiser to use a stable hash for the duration of a
        fit or an inference pass.

        This is not defensive programming — it fixes a measured defect. gensim 4.4's
        `Doc2Vec.infer_vector` initialises the new document vector with

            pseudorandom_weak_vector(size, seed_string=' '.join(doc_words))

        and `pseudorandom_weak_vector`'s signature is `(size, seed_string=None,
        hashfxn=hash)`. It takes the *module default* `hash`, NOT the `hashfxn` passed
        to the model — so the constructor argument that makes training reproducible
        has no effect on inference. Python salts `hash(str)` per process unless
        PYTHONHASHSEED is set, so inferred vectors are stable within one process and
        differ between processes.

        This was caught by running the same fit in two separate interpreters: the
        trained word vectors were bit-identical (md5 1c0bd6c2...), the inferred gold
        document vectors were not, and gold PR-AUC moved 0.6825 / 0.6833 / 0.6836
        across three processes. That is a 0.0011 wobble — immaterial next to the
        0.0165 seed spread and the 0.093 bootstrap CI, so no reported conclusion
        depended on it — but a determinism claim that is only true within a process is
        a false claim, and the alternative (telling the reader to export
        PYTHONHASHSEED before running) puts the burden in the wrong place.

        Patched as a scoped context manager rather than at import, so importing this
        module never mutates gensim's behaviour for anything else in the project.
        """
        import gensim.models.doc2vec as d2v_mod
        original = d2v_mod.pseudorandom_weak_vector

        def stable(size, seed_string=None, hashfxn=_stable_hash):
            return original(size, seed_string=seed_string, hashfxn=hashfxn)

        d2v_mod.pseudorandom_weak_vector = stable
        try:
            yield
        finally:
            d2v_mod.pseudorandom_weak_vector = original

    def fit(self, texts):
        from gensim.models.doc2vec import Doc2Vec, TaggedDocument

        tagged = [TaggedDocument(self.tokenize(t), [i]) for i, t in enumerate(texts)]
        with self._deterministic_gensim_init():
            self.model = Doc2Vec(
                documents=tagged,
                dm=0,                       # PV-DBOW
                dbow_words=0,               # pure DBOW: do not co-train word vectors
                vector_size=self.vector_size,
                window=D2V_WINDOW,
                min_count=D2V_MIN_COUNT,
                negative=D2V_NEGATIVE,
                sample=1e-3,
                epochs=self.epochs,
                seed=self.seed,
                workers=1,                  # >1 is non-deterministic; see module docstring
                hashfxn=_stable_hash,       # `hash` is per-process salted; see above
            )
        return self

    def _infer(self, text):
        # Reset the RNG per document so the vector is a pure function of the text,
        # not of the position of the document in the batch. This governs the negative
        # sampling draws; the *initial* vector is governed by the patched
        # pseudorandom_weak_vector in transform(). Both are needed.
        self.model.random = np.random.RandomState(self.seed)
        return self.model.infer_vector(self.tokenize(text), epochs=self.infer_epochs)

    def transform(self, texts):
        with self._deterministic_gensim_init():
            return np.vstack([self._infer(t) for t in texts])

    def describe(self):
        return (f"PV-DBOW {self.vector_size}d, {self.epochs} epochs, vocab "
                f"{len(self.model.wv)}, {len(self.model.dv)} trained doc vectors, seed {self.seed}")


class SbertRepresentation:
    """
    Frozen pretrained sentence embeddings. `fit` is a no-op by design: nothing here
    is learned from our data, so the only parameters fitted on the training split are
    the 384 logistic-regression coefficients. That is what makes this the tightest
    transfer-learning comparison against DeBERTa available without a GPU.

    Note the max sequence length is 256 word-pieces, the same window
    `src/sota_model.py` tokenizes DeBERTa to — so neither model sees more of a long
    review than the other, and the truncation measurement already in
    results/gold_fn_handread.md (median hazard cue at token 39) applies to both.
    """

    name = "minilm"
    is_sparse = False

    def __init__(self, seed=cfg.RANDOM_STATE, model_name=SBERT_MODEL_NAME):
        self.seed = seed
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer
            torch.manual_seed(self.seed)
            # Pinned to CPU explicitly: this box has neither CUDA nor MPS, and an
            # implicit device pick would make the numbers device-dependent.
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def fit(self, texts):
        _ = self.model      # force the download/load now so failures surface early
        return self

    def transform(self, texts):
        # eval mode + no grad; encode() already handles both, and the model is frozen.
        return np.asarray(self.model.encode(
            [str(t) for t in texts],
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ))

    def describe(self):
        # get_sentence_embedding_dimension() is deprecated in sentence-transformers 5.x
        # in favour of get_embedding_dimension(); requirements.txt floors the package at
        # 2.7, where only the old name exists. Prefer the new name, fall back to the old.
        dim_fn = (getattr(self.model, "get_embedding_dimension", None)
                  or self.model.get_sentence_embedding_dimension)
        return (f"{self.model_name} frozen, {dim_fn()}d, "
                f"max_seq_length={self.model.max_seq_length}, device=cpu")


_REPRESENTATIONS = {
    "tfidf_lr": TfidfRepresentation,
    "tfidf_lsa": LsaRepresentation,
    "doc2vec_dbow": Doc2VecRepresentation,
    "minilm": SbertRepresentation,
}


def build_representation(variant, seed=cfg.RANDOM_STATE):
    if variant not in _REPRESENTATIONS:
        raise ValueError(f"unknown variant {variant!r}; expected one of {sorted(_REPRESENTATIONS)}")
    return _REPRESENTATIONS[variant](seed=seed)


# -----------------------------------------------------------------------------
# Classifier head
# -----------------------------------------------------------------------------
def build_head(is_sparse, seed=cfg.RANDOM_STATE, C=HEAD_C, balance_classes=True):
    """
    The one classifier every variant shares.

    `class_weight="balanced"` reproduces the baseline's class balancing: sklearn sets
    w_c = n / (k * n_c), so w_pos / w_neg = n_neg / n_pos, which is exactly the
    `scale_pos_weight` src/baseline_model.py computes. Without this the comparison
    against the weighted transformer would be unfair at any shared threshold.

    Scaling: `with_mean` is off for sparse input (centring would densify a 2500-column
    matrix); dense embeddings are centred and scaled, which L2-regularised logistic
    regression needs because LSA and Doc2Vec dimensions have very different variances.
    """
    return Pipeline([
        ("scale", StandardScaler(with_mean=not is_sparse)),
        ("clf", LogisticRegression(
            C=C,
            max_iter=HEAD_MAX_ITER,
            class_weight="balanced" if balance_classes else None,
            solver="lbfgs",
            random_state=seed,
        )),
    ])


def train_variant(variant, train_df, seed=cfg.RANDOM_STATE, verbose=True):
    """Fits one representation + head on the train split. Returns (rep, head)."""
    np.random.seed(seed)
    rep = build_representation(variant, seed=seed)
    texts = train_df[cfg.TEXT_COLUMN].astype(str).tolist()
    y = train_df[cfg.TARGET_COLUMN].astype(int).values

    if verbose:
        print(f"\n--- {variant}: fitting representation on {len(texts)} train reviews ---")
    rep.fit(texts)
    X = rep.transform(texts)
    if verbose:
        print(f"    {rep.describe()}")
        print(f"    train matrix {X.shape}")

    head = build_head(rep.is_sparse, seed=seed)
    head.fit(X, y)
    return rep, head


def variant_probabilities(rep, head, df):
    """P(hazard) for one dataframe. Text only — no tabular columns are read."""
    X = rep.transform(df[cfg.TEXT_COLUMN].astype(str).tolist())
    return head.predict_proba(X)[:, 1]


# -----------------------------------------------------------------------------
# Reference numbers from the committed artifacts (read-only; never re-run)
# -----------------------------------------------------------------------------
def load_committed_reference(slug, results_dir=None):
    """
    Reads the already-committed baseline / DeBERTa rows for `slug` so the embedding
    table can be reported next to them without retraining either model. Re-running
    analysis.py would overwrite artifacts the report already cites, so it is not an
    option; reading its output is.
    """
    path = os.path.join(results_dir or cfg.RESULTS_DIR, f"performance_{slug}.csv")
    if not os.path.exists(path):
        print(f"  [warn] no committed reference table at {path}; "
              f"the embedding table will stand alone")
        return None
    return pd.read_csv(path, index_col=0)


# -----------------------------------------------------------------------------
# Figures — same cost model and the same visual grammar as the committed figures
# -----------------------------------------------------------------------------
def _plot_embedding_curves(y_true, prob_map, name, reference, optimal_th, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    y_true = np.asarray(y_true).astype(int)
    slug = _slug(name)
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2", "#937860"]

    # --- PR curves -----------------------------------------------------------
    plt.figure(figsize=(8.5, 5.5))
    for (variant, probs), color in zip(prob_map.items(), colors):
        p, r, _ = precision_recall_curve(y_true, probs)
        plt.plot(r, p, lw=2, color=color,
                 label=f"{DISPLAY.get(variant, variant)} (AP = {average_precision_score(y_true, probs):.3f})")
    plt.axhline(y_true.mean(), color="grey", linestyle="--", lw=1,
                label=f"Random baseline ({y_true.mean():.3f})")

    if reference is not None:
        ref_lines = "\n".join(
            f"{idx}: AP {row['PR-AUC']:.3f}" for idx, row in reference.iterrows()
            if "th=0.20" not in idx)     # PR-AUC is threshold-free; one row per model
        plt.gca().text(0.02, 0.03, "Committed reference (results/):\n" + ref_lines,
                       transform=plt.gca().transAxes, fontsize=8, va="bottom",
                       bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=.85))

    plt.xlabel("Recall (Safety Coverage)")
    plt.ylabel("Precision (Alert Validity)")
    plt.title(f"Document embeddings — Precision–Recall — {name}")
    plt.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    pr_path = os.path.join(output_dir, f"pr_curve_embedding_{slug}.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # --- Cost curve, identical cost model to analysis/evaluation_pipeline.py --
    th_grid = np.linspace(0.01, 0.99, 100)

    def cost_at(probs, t):
        return (((y_true == 1) & (probs < t)).sum() * COST_FALSE_NEGATIVE
                + ((y_true == 0) & (probs >= t)).sum() * COST_FALSE_POSITIVE)

    plt.figure(figsize=(9, 5.5))
    best = {}
    for (variant, probs), color in zip(prob_map.items(), colors):
        costs = [cost_at(probs, t) for t in th_grid]
        plt.plot(th_grid, costs, lw=2, color=color, label=DISPLAY.get(variant, variant))
        best[variant] = float(th_grid[int(np.argmin(costs))])
    plt.axvline(optimal_th, color="crimson", linestyle=":",
                label=f"Deployed threshold ({optimal_th:.2f})")
    plt.xlabel("Classification Probability Threshold")
    plt.ylabel("Total Operational & Liability Cost ($)")
    plt.title(f"Business Liability Cost vs Threshold — embeddings — {name}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    cost_path = os.path.join(output_dir, f"cost_curve_embedding_{slug}.png")
    plt.savefig(cost_path, dpi=300)
    plt.close()

    for variant, th in best.items():
        print(f"  Cost-minimising threshold ({variant}) on this set: {th:.2f} "
              f"(deployed: {optimal_th:.2f})")
    return pr_path, cost_path


def _slug(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def evaluate_variants_on_set(name, df, y_true, prob_map, optimal_th=None,
                             output_dir=None, run_error_analysis=False):
    """
    Scores every variant on one dataset against one ground truth, using
    `score_variant` from analysis/evaluation_pipeline.py — the same function, the
    same cost model and the same thresholds the baseline and DeBERTa are reported
    under. Both th=0.50 and th=0.20 are emitted per variant so the rows line up with
    both the baseline row (0.50) and the deployed DeBERTa row (0.20).
    """
    optimal_th = cfg.DECISION_THRESHOLD if optimal_th is None else optimal_th
    output_dir = output_dir or cfg.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    y_true = np.asarray(y_true).astype(int)
    slug = _slug(name)

    print("\n" + "=" * 74)
    print(f"  EMBEDDING EVALUATION: {name}")
    print(f"  {len(y_true)} reviews | hazard rate {y_true.mean():.1%}")
    print("=" * 74)

    columns = {}
    for variant, probs in prob_map.items():
        label = DISPLAY.get(variant, variant)
        columns[f"{label} (th=0.50)"] = score_variant(y_true, probs, 0.50)
        columns[f"{label} (th={optimal_th:.2f})"] = score_variant(y_true, probs, optimal_th)

    summary = pd.DataFrame(columns).T
    print(_format_table(summary).to_string())

    out_csv = os.path.join(output_dir, f"performance_embedding_{slug}.csv")
    summary.to_csv(out_csv)
    print(f"\n  -> {out_csv}")

    if run_error_analysis:
        for variant, probs in prob_map.items():
            preds = (probs >= optimal_th).astype(int)
            analyze_errors(df, y_true=y_true, y_pred=preds, probs=probs,
                           label=f"embedding_{variant}_{slug}", output_dir=output_dir)

    return summary


def _fp_mode_comparison(variants, slug, output_dir):
    """
    Puts each variant's gold false-positive mode distribution beside DeBERTa's, which
    is the comparison CLAUDE.md's error-analysis section is built on: 65% of DeBERTa's
    gold false positives sit in the labelling rule's own top two failure modes
    (`illness_mentioned_not_caused_here` + `neutral_allergen_mention`). If the frozen
    embeddings inherit the same two modes at the same rate, the pathology is the
    label's, not the architecture's.
    """
    rows = []
    sources = [("deberta", f"error_analysis_deberta_{slug}_summary.csv")]
    sources += [(v, f"error_analysis_embedding_{v}_{slug}_summary.csv") for v in variants]

    for model, fname in sources:
        path = os.path.join(output_dir, fname)
        if not os.path.exists(path):
            print(f"  [warn] {fname} not found; {model} omitted from the FP-mode comparison")
            continue
        s = pd.read_csv(path)
        fp = s[s.error_type == "FP"]
        if fp.empty:
            continue
        total = int(fp["count"].sum())
        modes = dict(zip(fp.primary_mode, fp["count"]))
        top2 = (modes.get("illness_mentioned_not_caused_here", 0)
                + modes.get("neutral_allergen_mention", 0))
        rows.append({
            "model": model,
            "n_false_positives": total,
            "illness_mentioned_not_caused_here": int(modes.get("illness_mentioned_not_caused_here", 0)),
            "neutral_allergen_mention": int(modes.get("neutral_allergen_mention", 0)),
            "share_in_rule_top2_modes": round(top2 / total, 4) if total else float("nan"),
            "negated_hazard": int(modes.get("negated_hazard", 0)),
            "generic_complaint_no_hazard": int(modes.get("generic_complaint_no_hazard", 0)),
        })

    if not rows:
        return None
    comp = pd.DataFrame(rows)
    path = os.path.join(output_dir, "embedding_vs_deberta_fp_modes.csv")
    comp.to_csv(path, index=False)
    print("\n" + "=" * 74)
    print("  GOLD FALSE-POSITIVE MODES: embeddings vs DeBERTa")
    print("=" * 74)
    print(comp.to_string(index=False))
    print("\n  `share_in_rule_top2_modes` is the statistic CLAUDE.md reports as 65% for\n"
          "  DeBERTa. A similar share means the failure is inherited from the label.")
    return comp


def bootstrap_gold_pr_auc(y_true, prob_map, n_boot=1000, seed=cfg.RANDOM_STATE,
                          output_dir=None, reference=None):
    """
    Stratified bootstrap CI on gold PR-AUC, per variant.

    This exists because the project's stated noise floor (0.054 gold PR-AUC) measures
    the wrong thing for this module. That figure is *training* non-determinism on MPS
    for the transformer. Every variant here except Doc2Vec is exactly reproducible, so
    training noise is zero — but the holdout is only 772 rows, and *sampling* noise is
    not zero. Without this number a reader cannot tell whether "LSA 0.757 beats the
    committed baseline 0.728" is a result or a coin flip.

    Resampling is stratified within class so each replicate keeps the 46.0% funnel
    hazard rate; an unstratified bootstrap would inflate the interval with base-rate
    wobble that the real holdout does not have.

    LIMITATION, stated because it bounds what the CI can be used for: this quantifies
    uncertainty from the finite holdout only. It cannot be compared against the
    committed baseline or DeBERTa numbers as a difference test, because their
    probability arrays were never persisted to results/ — only their summary metrics
    were — so the paired resample that a difference test needs is not available
    without re-running analysis.py, which would overwrite committed artifacts.
    """
    output_dir = output_dir or cfg.RESULTS_DIR
    y_true = np.asarray(y_true).astype(int)
    pos, neg = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    rng = np.random.default_rng(seed)

    idx = np.stack([np.concatenate([rng.choice(pos, len(pos), replace=True),
                                    rng.choice(neg, len(neg), replace=True)])
                    for _ in range(n_boot)])

    rows = []
    for variant, probs in prob_map.items():
        probs = np.asarray(probs)
        draws = np.array([average_precision_score(y_true[i], probs[i]) for i in idx])
        rows.append({
            "variant": variant,
            "display": DISPLAY.get(variant, variant),
            "gold_pr_auc": average_precision_score(y_true, probs),
            "boot_mean": draws.mean(),
            "boot_sd": draws.std(ddof=1),
            "ci95_low": np.percentile(draws, 2.5),
            "ci95_high": np.percentile(draws, 97.5),
            "ci95_width": np.percentile(draws, 97.5) - np.percentile(draws, 2.5),
            "n_boot": n_boot,
        })

    out = pd.DataFrame(rows).sort_values("gold_pr_auc", ascending=False)
    path = os.path.join(output_dir, "embedding_gold_pr_auc_bootstrap.csv")
    out.to_csv(path, index=False)

    print("\n" + "=" * 74)
    print(f"  GOLD PR-AUC — stratified bootstrap CI ({n_boot} resamples, n={len(y_true)})")
    print("=" * 74)
    print(out[["variant", "gold_pr_auc", "boot_sd", "ci95_low", "ci95_high",
               "ci95_width"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    widest = out["ci95_width"].max()
    print(f"\n  Widest 95% CI is {widest:.3f} wide. Any gap between two models smaller")
    print(f"  than roughly half that is not resolved by a 772-row holdout, whatever the")
    print(f"  point estimates say. Read the committed reference numbers against this.")
    if reference is not None:
        for idx_name, row in reference.iterrows():
            print(f"    committed reference: {idx_name} = {row['PR-AUC']:.4f}")
    print(f"  -> {path}")
    return out


def doc2vec_seed_spread(train_df, test_df, gold_df, seeds, output_dir=None,
                        variant="doc2vec_dbow"):
    """
    Re-fits a seed-sensitive variant once per seed and reports the observed spread.

    This is the honesty requirement, and it is not optional: Doc2Vec is reproducible
    at a fixed seed here (workers=1, deterministic hashfxn, per-document RNG reset)
    but there is no reason a single seed's gold PR-AUC is the population value. The
    project already quotes a 0.054 gold PR-AUC noise floor for the transformer at a
    *fixed* seed; this measures the analogous quantity across seeds, which is the
    number a reader should compare embedding-vs-transformer gaps against.
    """
    output_dir = output_dir or cfg.RESULTS_DIR
    rows = []
    for seed in seeds:
        rep, head = train_variant(variant, train_df, seed=seed, verbose=False)
        row = {"variant": variant, "seed": seed}
        for set_name, df, col in (("heuristic_test", test_df, cfg.TARGET_COLUMN),
                                  ("gold_holdout", gold_df, cfg.TARGET_COLUMN)):
            if df is None:
                continue
            probs = variant_probabilities(rep, head, df)
            y = df[col].astype(int).values
            m = score_variant(y, probs, cfg.DECISION_THRESHOLD)
            row[f"{set_name}_pr_auc"] = m["PR-AUC"]
            row[f"{set_name}_f2_at_{cfg.DECISION_THRESHOLD:.2f}"] = m["F2 (recall-weighted)"]
            row[f"{set_name}_flag_rate"] = m["Flag Rate"]
        rows.append(row)
        print(f"    seed {seed}: "
              + "  ".join(f"{k}={v:.4f}" for k, v in row.items()
                          if isinstance(v, float)))

    spread = pd.DataFrame(rows)
    for col in [c for c in spread.columns if c.endswith("pr_auc")]:
        rng = spread[col].max() - spread[col].min()
        print(f"  {col}: mean {spread[col].mean():.4f}, "
              f"sd {spread[col].std(ddof=1):.4f}, range {rng:.4f} over {len(seeds)} seeds")
    path = os.path.join(output_dir, f"embedding_{variant}_seed_spread.csv")
    spread.to_csv(path, index=False)
    print(f"  -> {path}")
    return spread


def run_embedding_evaluation(variants=VARIANTS, seeds=(cfg.RANDOM_STATE,),
                             output_dir=None, run_error_analysis=True,
                             strict_leakage=True):
    """End to end: fit, leakage-check, score on both ground truths, persist."""
    output_dir = output_dir or cfg.RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    seeds = list(seeds)
    primary_seed = seeds[0]

    print("=" * 74)
    print("  DOCUMENT-EMBEDDING CLASSIFIERS — course technique #3")
    print(f"  variants: {', '.join(variants)}   primary seed: {primary_seed}")
    print("=" * 74)

    train_df, val_df, test_df = load_and_split_data(with_validation=True)
    gold_df = load_gold_holdout(require=False)

    # --- Leakage: the check must run before anything is reported -------------
    print("\n--- Holdout integrity (mirrors verify_setup.py section 4) ---")
    eval_sets = {"test-split": test_df[cfg.TEXT_COLUMN].tolist(),
                 "validation": val_df[cfg.TEXT_COLUMN].tolist()}
    if gold_df is not None:
        eval_sets["gold-holdout"] = gold_df[cfg.TEXT_COLUMN].tolist()
    overlap = verify_no_text_overlap(train_df[cfg.TEXT_COLUMN].tolist(), eval_sets,
                                     strict=strict_leakage)

    # --- Fit every variant once at the primary seed --------------------------
    fitted, diagnostics = {}, []
    for variant in variants:
        try:
            rep, head = train_variant(variant, train_df, seed=primary_seed)
        except Exception as e:
            print(f"  [FAIL] {variant} could not be built: {type(e).__name__}: {e}")
            print(f"         skipping it. For {variant!r} == 'minilm' the documented "
                  f"fallback is 'tfidf_lsa' (LSA), which is already in this run.")
            continue
        fitted[variant] = (rep, head)

        val_probs = variant_probabilities(rep, head, val_df)
        val_m = score_variant(val_df[cfg.TARGET_COLUMN].astype(int).values,
                              val_probs, cfg.DECISION_THRESHOLD)
        diagnostics.append({
            "variant": variant,
            "display": DISPLAY.get(variant, variant),
            "representation": rep.describe(),
            "dimensions": (rep.transform(train_df[cfg.TEXT_COLUMN].head(1).tolist()).shape[1]),
            "head": f"LogisticRegression(C={HEAD_C}, class_weight=balanced)",
            "seed": primary_seed,
            "val_pr_auc": val_m["PR-AUC"],
            f"val_f2_at_{cfg.DECISION_THRESHOLD:.2f}": val_m["F2 (recall-weighted)"],
            "val_flag_rate": val_m["Flag Rate"],
        })

    if not fitted:
        raise RuntimeError("no embedding variant could be built; nothing to report")

    diag = pd.DataFrame(diagnostics)
    diag_path = os.path.join(output_dir, "embedding_variant_diagnostics.csv")
    diag.to_csv(diag_path, index=False)
    print("\n--- Variant diagnostics (validation split; selection-free, "
          "C is NOT tuned on it) ---")
    print(diag.drop(columns=["representation"]).to_string(index=False))
    print(f"  -> {diag_path}")

    # --- Ground truth 1: heuristic label on the test split -------------------
    test_probs = {v: variant_probabilities(*fitted[v], test_df) for v in fitted}
    heuristic_summary = evaluate_variants_on_set(
        "Heuristic label (test split)", test_df,
        test_df[cfg.TARGET_COLUMN].values, test_probs,
        output_dir=output_dir, run_error_analysis=False)
    _plot_embedding_curves(
        test_df[cfg.TARGET_COLUMN].values, test_probs, "Heuristic label (test split)",
        load_committed_reference("heuristic_label_test_split", output_dir),
        cfg.DECISION_THRESHOLD, output_dir)

    gold_summary = comparison = fp_modes = spread = bootstrap = None
    if gold_df is None:
        print("\n[warning] No gold holdout set found. The numbers above measure agreement "
              "with a labelling rule that is 73% precise against expert judgement — do "
              "not report them as detection performance.")
    else:
        # --- Ground truth 2: the 772-row zero-overlap LLM-judged holdout -----
        gold_probs = {v: variant_probabilities(*fitted[v], gold_df) for v in fitted}
        gold_summary = evaluate_variants_on_set(
            "Gold LLM label (fresh holdout)", gold_df,
            gold_df[cfg.TARGET_COLUMN].values, gold_probs,
            output_dir=output_dir, run_error_analysis=run_error_analysis)
        _plot_embedding_curves(
            gold_df[cfg.TARGET_COLUMN].values, gold_probs, "Gold LLM label (fresh holdout)",
            load_committed_reference("gold_llm_label_fresh_holdout", output_dir),
            cfg.DECISION_THRESHOLD, output_dir)

        comparison = _compare_ground_truths(heuristic_summary, gold_summary, output_dir)
        bootstrap = bootstrap_gold_pr_auc(
            gold_df[cfg.TARGET_COLUMN].values, gold_probs, seed=primary_seed,
            output_dir=output_dir,
            reference=load_committed_reference("gold_llm_label_fresh_holdout", output_dir))
        if run_error_analysis:
            fp_modes = _fp_mode_comparison(list(fitted), "gold_llm_label_fresh_holdout",
                                           output_dir)

        # --- Seed spread for the seed-sensitive variant ----------------------
        seed_variants = [v for v in fitted if v in SEED_SENSITIVE_VARIANTS]
        if len(seeds) > 1 and seed_variants:
            print("\n" + "=" * 74)
            print(f"  SEED SPREAD — {', '.join(seed_variants)} refitted at seeds "
                  f"{seeds}")
            print("=" * 74)
            for v in seed_variants:
                spread = doc2vec_seed_spread(train_df, test_df, gold_df, seeds,
                                             output_dir=output_dir, variant=v)
        elif seed_variants:
            print(f"\n[note] Only one seed requested, so no spread is reported for "
                  f"{seed_variants}. Doc2Vec is reproducible at a fixed seed in this "
                  f"module but not across seeds — pass --seeds 42 43 44 before quoting "
                  f"a Doc2Vec gap as real.")

    _write_headline_table(heuristic_summary, gold_summary, output_dir)
    print(f"\nAll embedding artifacts written to: {output_dir}")
    return {
        "heuristic": heuristic_summary,
        "gold": gold_summary,
        "ground_truth_comparison": comparison,
        "fp_modes": fp_modes,
        "gold_pr_auc_bootstrap": bootstrap,
        "seed_spread": spread,
        "diagnostics": diag,
        "overlap": overlap,
    }


def _compare_ground_truths(heuristic_summary, gold_summary, output_dir):
    """Same shape as analysis/evaluation_pipeline.compare_ground_truths, for embeddings."""
    rows = []
    for model in heuristic_summary.index:
        if model not in gold_summary.index:
            continue
        for metric in ("PR-AUC", "Recall (Safety Coverage)", "Precision (Alert Validity)",
                       "F2 (recall-weighted)"):
            h, g = heuristic_summary.loc[model, metric], gold_summary.loc[model, metric]
            rows.append({"model": model, "metric": metric,
                         "vs_heuristic_label": h, "vs_gold_llm_label": g, "delta": g - h})
    comparison = pd.DataFrame(rows)
    path = os.path.join(output_dir, "embedding_ground_truth_comparison.csv")
    comparison.to_csv(path, index=False)
    print("\n" + "=" * 74)
    print("  HEURISTIC LABEL vs INDEPENDENT GOLD LABEL — embeddings")
    print("=" * 74)
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n  A negative delta is the portion of the score that came from reproducing"
          "\n  the labelling rule rather than detecting hazards. Compare each variant's"
          "\n  delta against the committed -0.251 (TF-IDF+XGBoost) and -0.183 (DeBERTa).")
    print(f"  -> {path}")
    return comparison


def _write_headline_table(heuristic_summary, gold_summary, output_dir):
    """
    One file a reader can open to see all three techniques on both ground truths.
    The baseline and DeBERTa rows are copied verbatim from the committed CSVs — they
    are not recomputed, because re-running analysis.py would overwrite artifacts the
    report cites.
    """
    frames = []
    for slug, gt in (("heuristic_label_test_split", "heuristic label (test split)"),
                     ("gold_llm_label_fresh_holdout", "gold LLM label (fresh holdout)")):
        ref = load_committed_reference(slug, output_dir)
        emb = heuristic_summary if slug.startswith("heuristic") else gold_summary
        for src, provenance in ((ref, "committed (results/performance_*.csv)"),
                                (emb, "this run (src/embedding_model.py)")):
            if src is None:
                continue
            block = src.copy()
            block.insert(0, "ground_truth", gt)
            block.insert(1, "provenance", provenance)
            frames.append(block)

    if not frames:
        return None
    table = pd.concat(frames)
    table.index.name = "model"
    path = os.path.join(output_dir, "embedding_technique_comparison.csv")
    table.to_csv(path)
    print("\n" + "=" * 74)
    print("  ALL TECHNIQUES, BOTH GROUND TRUTHS (PR-AUC)")
    print("=" * 74)
    print(table[["ground_truth", "PR-AUC", "Recall (Safety Coverage)",
                 "Precision (Alert Validity)", "F2 (recall-weighted)",
                 "Total Risk Cost"]].to_string())
    print(f"  -> {path}")
    return table


def _parse_args():
    p = argparse.ArgumentParser(
        description="Document-embedding classifiers, scored on both ground truths.")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                   help="which representations to run (default: all four)")
    p.add_argument("--seeds", nargs="+", type=int, default=[cfg.RANDOM_STATE],
                   help="seeds; the first is the reported one. Passing more than one "
                        "triggers the Doc2Vec seed-spread study.")
    p.add_argument("--output-dir", default=None, help="default: results/")
    p.add_argument("--no-error-analysis", action="store_true",
                   help="skip the failure-mode bucketing on the gold holdout")
    p.add_argument("--allow-leakage", action="store_true",
                   help="warn instead of aborting when a holdout row appears in the "
                        "fitted corpus. Diagnostic only — never report numbers from it.")
    return p.parse_args()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Windows defaults stdout to cp1252 when it is redirected to a file, and a single
    # U+2212 MINUS SIGN in a print string was enough to kill this script *after* it
    # had written most of its artifacts — the failure landed between two writes, which
    # is the worst place for it. Force UTF-8 so `> results/embedding_model_run.log`
    # behaves the same on both of this project's two machines.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass

    args = _parse_args()
    run_embedding_evaluation(
        variants=tuple(args.variants),
        seeds=tuple(args.seeds),
        output_dir=args.output_dir,
        run_error_analysis=not args.no_error_analysis,
        strict_leakage=not args.allow_leakage,
    )
