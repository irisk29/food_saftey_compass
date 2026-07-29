"""
Tests for the document-embedding classifiers (src/embedding_model.py).

Run: python -m pytest tests/ -v      (or: python tests/test_embedding_model.py)

Scope is deliberately narrow. These pin the four properties that would silently
invalidate the reported numbers if they broke, and nothing else:

  1. **Leakage.** The guard must actually fire, and it must be keyed on the same text
     normalisation `verify_setup.py` uses — otherwise "zero overlap" means nothing.
  2. **Class balancing parity with the baseline.** The comparison against a
     class-weighted transformer is only fair if the embedding head is weighted the
     same way `src/baseline_model.py` weights XGBoost. This is an arithmetic identity
     between two different libraries' conventions, which is exactly the kind of thing
     that is asserted in prose and never checked.
  3. **Determinism.** Doc2Vec has two non-obvious non-determinism sources (worker
     count and gensim's per-process-salted default `hashfxn`); LSA has a randomised
     SVD solver. All are pinned, so all must reproduce bit-for-bit.
  4. **Text-only.** The embedding model must not read a tabular column, because the
     claim that it is directly comparable to text-only DeBERTa rests on that.

No test here trains on the real 4,800-row corpus or asserts a performance number.
Performance numbers live in results/ and move with the seed; asserting one in a test
would turn a measurement into a regression trap.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config.settings as cfg
from src.embedding_model import (
    Doc2VecRepresentation,
    LsaRepresentation,
    _norm,
    _stable_hash,
    build_head,
    build_representation,
    train_variant,
    variant_probabilities,
    verify_no_text_overlap,
)

HAZARD = [
    "I got food poisoning from the chicken and spent the night in the bathroom",
    "Severe allergic reaction to the peanut sauce, ended up in the emergency room",
    "There was a cockroach in my salad, absolutely disgusting and unsafe",
    "The pork was raw in the middle and I threw up two hours later",
    "My daughter had hives after eating the cake, they swore it was nut free",
    "Cross contamination gave me a celiac reaction, I was ill for days",
]
BENIGN = [
    "Great tacos and the salsa bar is fantastic, will come back every week",
    "Service was slow but the pizza was worth the wait, lovely patio",
    "They have a gluten free menu which my wife appreciated, very accommodating",
    "Coffee was cold and the barista seemed tired, otherwise a fine place",
    "Best pad thai in town, generous portions and a friendly staff",
    "Parking is a nightmare but the burgers are solid and cheap",
]


def _toy_frame(n_repeat=6):
    """A small labelled frame with the project's column names."""
    texts = (HAZARD * n_repeat) + (BENIGN * n_repeat)
    labels = ([1] * len(HAZARD) * n_repeat) + ([0] * len(BENIGN) * n_repeat)
    return pd.DataFrame({
        cfg.TEXT_COLUMN: texts,
        cfg.TARGET_COLUMN: labels,
        # Present but must never be read. See test_representation_is_text_only.
        "stars": [1] * len(HAZARD) * n_repeat + [5] * len(BENIGN) * n_repeat,
        "vader_neg_intensity": [0.9] * len(HAZARD) * n_repeat + [0.0] * len(BENIGN) * n_repeat,
        "useful": 0, "funny": 0, "cool": 0,
        "word_count": [len(t.split()) for t in texts],
        "char_count": [len(t) for t in texts],
        "exclamation_count": 0,
    })


# -----------------------------------------------------------------------------
# 1. Leakage guard
# -----------------------------------------------------------------------------
def test_leakage_guard_raises_on_a_contaminated_strict_set():
    """One shared row in a strict set must abort, not warn."""
    fit = ["a hazard review about food poisoning", "a benign review about tacos"]
    try:
        verify_no_text_overlap(fit, {"gold-holdout": ["a benign review about tacos"]})
    except RuntimeError as e:
        assert "gold-holdout" in str(e)
        return
    raise AssertionError("expected RuntimeError on a contaminated gold holdout")


def test_leakage_guard_passes_on_disjoint_text():
    report = verify_no_text_overlap(
        ["completely different text one", "completely different text two"],
        {"gold-holdout": ["nothing in common here at all"]})
    assert report == {"gold-holdout": 0}, report


def test_leakage_guard_is_whitespace_insensitive():
    """
    The point of normalising is that a re-wrapped duplicate is still a duplicate.
    If this test fails the guard reports zero overlap on text that IS shared.
    """
    assert _norm("a  b\tc\nd ") == "a b c d"
    try:
        verify_no_text_overlap(["the same review text"],
                               {"gold-holdout": ["the   same\nreview\ttext  "]})
    except RuntimeError:
        return
    raise AssertionError("whitespace-differing duplicates must still be caught")


def test_non_strict_sets_are_reported_but_do_not_abort():
    """
    The enriched CSV has 3 duplicate texts, one straddling the train/test boundary,
    so the in-corpus splits are reported and not fatal. That policy is load-bearing
    for the module running at all — pin it so nobody "fixes" it into an abort.
    """
    report = verify_no_text_overlap(
        ["shared row"],
        {"test-split": ["shared row"], "gold-holdout": ["something else entirely"]})
    assert report["test-split"] == 1
    assert report["gold-holdout"] == 0


# -----------------------------------------------------------------------------
# 2. Class-balance parity with src/baseline_model.py
# -----------------------------------------------------------------------------
def test_head_class_weighting_matches_baseline_scale_pos_weight():
    """
    src/baseline_model.py uses XGBoost's scale_pos_weight = n_neg / n_pos.
    sklearn's class_weight="balanced" uses w_c = n / (k * n_c), so the *ratio*
    w_pos / w_neg must equal n_neg / n_pos for the comparison to be fair. Verified
    against the fitted estimator, not against the docstring.
    """
    from sklearn.utils.class_weight import compute_class_weight

    y = np.array([1] * 30 + [0] * 120)          # deliberately imbalanced, 4:1
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    baseline_spw = n_neg / n_pos

    w = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    sklearn_ratio = w[1] / w[0]
    assert abs(sklearn_ratio - baseline_spw) < 1e-9, (sklearn_ratio, baseline_spw)

    head = build_head(is_sparse=False)
    assert head.named_steps["clf"].class_weight == "balanced"


def test_head_balancing_can_be_switched_off():
    """The unweighted control must exist, otherwise 'balanced' is untestable."""
    assert build_head(is_sparse=False, balance_classes=False).named_steps["clf"].class_weight is None


def test_sparse_input_is_not_centred():
    """
    Centring a sparse matrix densifies it. StandardScaler must therefore run with
    with_mean=False for the TF-IDF control and with_mean=True for dense embeddings.
    """
    assert build_head(is_sparse=True).named_steps["scale"].with_mean is False
    assert build_head(is_sparse=False).named_steps["scale"].with_mean is True


# -----------------------------------------------------------------------------
# 3. Determinism
# -----------------------------------------------------------------------------
def test_stable_hash_is_process_independent():
    """
    gensim's default hashfxn is Python's `hash`, which is salted per process unless
    PYTHONHASHSEED is set — the exact defect this replacement exists to remove. The
    expected values below were computed once and are hard-coded on purpose: if this
    function ever changes, every committed Doc2Vec number becomes unreproducible and
    the test should fail loudly rather than agree with itself.
    """
    assert _stable_hash("hazard") == 0x08ae027b
    assert _stable_hash("hazard") == _stable_hash("hazard")
    assert _stable_hash("hazard") != _stable_hash("benign")


def test_doc2vec_is_bit_identical_at_a_fixed_seed():
    """
    Two independently constructed models at the same seed must produce identical
    vectors. This covers workers=1, the deterministic hashfxn and the per-document
    RNG reset in _infer simultaneously — remove any one of the three and this fails.
    """
    texts = (HAZARD + BENIGN) * 4
    a = Doc2VecRepresentation(seed=cfg.RANDOM_STATE, vector_size=32, epochs=5).fit(texts)
    b = Doc2VecRepresentation(seed=cfg.RANDOM_STATE, vector_size=32, epochs=5).fit(texts)
    held_out = ["a brand new review describing raw chicken and a bad stomach"]
    assert np.array_equal(a.transform(held_out), b.transform(held_out))
    assert np.array_equal(a.transform(texts[:5]), b.transform(texts[:5]))


def test_doc2vec_inference_is_order_independent():
    """
    infer_vector draws a random start from model.random, so without the per-document
    reset a document's vector depends on its position in the batch. A held-out row
    must embed identically whether it is transformed alone or inside a batch.
    """
    texts = (HAZARD + BENIGN) * 4
    rep = Doc2VecRepresentation(seed=cfg.RANDOM_STATE, vector_size=32, epochs=5).fit(texts)
    target = "a brand new review describing raw chicken and a bad stomach"
    alone = rep.transform([target])[0]
    in_batch = rep.transform(["unrelated filler about parking", target, "more filler"])[1]
    assert np.array_equal(alone, in_batch)


_CROSS_PROCESS_SNIPPET = """
import hashlib, os, sys
import numpy as np
sys.path.insert(0, {root!r})
from src.embedding_model import Doc2VecRepresentation
texts = {texts!r}
rep = Doc2VecRepresentation(seed=42, vector_size=32, epochs=5).fit(texts)
v = rep.transform(["a brand new review describing raw chicken and a bad stomach"])
print(hashlib.md5(np.ascontiguousarray(v, dtype=np.float32).tobytes()).hexdigest())
"""


def test_doc2vec_is_identical_across_PROCESSES():
    """
    The test that actually matters, and the one whose absence hid a real defect.

    Within a single process Doc2Vec was reproducible even *before* the
    `pseudorandom_weak_vector` patch, because Python's hash salt is fixed for the life
    of an interpreter. The defect only showed up across interpreters — which is the
    case that counts, since every reported number comes from a separate `python -m
    src.embedding_model` invocation. Two subprocesses must agree.

    If this fails and the in-process test passes, the cause is almost certainly that
    `Doc2VecRepresentation._deterministic_gensim_init` stopped being applied to either
    fit() or transform(). Both need it.
    """
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    code = _CROSS_PROCESS_SNIPPET.format(root=root, texts=(HAZARD + BENIGN) * 4)
    env = dict(os.environ)
    # Explicitly do NOT set PYTHONHASHSEED: the point is that the module must be
    # reproducible without it. Force the two children to disagree on the salt if the
    # patch is absent, by letting each randomise independently.
    env.pop("PYTHONHASHSEED", None)

    digests = []
    for _ in range(2):
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=root)
        assert r.returncode == 0, r.stderr[-2000:]
        digests.append(r.stdout.strip().splitlines()[-1])

    assert digests[0] == digests[1], (
        f"inferred Doc2Vec vectors differ across processes ({digests}) — the "
        f"pseudorandom_weak_vector patch is not being applied")


def test_doc2vec_differs_across_seeds():
    """
    The flip side, and the reason the module ships a seed-spread study: a fixed seed
    is reproducible but not authoritative. If this ever passes trivially (identical
    vectors across seeds) the seed is not being threaded through at all.
    """
    texts = (HAZARD + BENIGN) * 4
    a = Doc2VecRepresentation(seed=42, vector_size=32, epochs=5).fit(texts)
    b = Doc2VecRepresentation(seed=1234, vector_size=32, epochs=5).fit(texts)
    assert not np.array_equal(a.transform(texts[:5]), b.transform(texts[:5]))


def test_lsa_is_deterministic():
    """TruncatedSVD's randomised solver is seeded; two fits must agree exactly."""
    texts = (HAZARD + BENIGN) * 4
    a = LsaRepresentation(seed=cfg.RANDOM_STATE, n_components=8).fit(texts)
    b = LsaRepresentation(seed=cfg.RANDOM_STATE, n_components=8).fit(texts)
    assert np.allclose(a.transform(texts), b.transform(texts), atol=0)


def _fit_toy_dense(df, n_components=8, seed=cfg.RANDOM_STATE):
    """
    Dense variant at a width the toy corpus can support.

    The module default is 300 components, which exceeds the toy vocabulary — so these
    tests build the representation directly rather than through `train_variant`, which
    is correct: `train_variant`'s job is to apply the *reported* configuration, and a
    test must not silently shrink it.
    """
    texts = df[cfg.TEXT_COLUMN].astype(str).tolist()
    rep = LsaRepresentation(seed=seed, n_components=n_components).fit(texts)
    head = build_head(rep.is_sparse, seed=seed)
    head.fit(rep.transform(texts), df[cfg.TARGET_COLUMN].astype(int).values)
    return rep, head


def test_end_to_end_variant_is_deterministic():
    """Whole pipeline, twice, same seed — identical probabilities. Sparse and dense."""
    df = _toy_frame()

    p1 = variant_probabilities(*train_variant("tfidf_lr", df, verbose=False), df)
    p2 = variant_probabilities(*train_variant("tfidf_lr", df, verbose=False), df)
    assert np.array_equal(p1, p2), "sparse control is not deterministic"

    d1 = variant_probabilities(*_fit_toy_dense(df), df)
    d2 = variant_probabilities(*_fit_toy_dense(df), df)
    assert np.array_equal(d1, d2), "dense LSA variant is not deterministic"


# -----------------------------------------------------------------------------
# 4. Shape, text-only, and interface contracts
# -----------------------------------------------------------------------------
def test_representation_is_text_only():
    """
    The comparability claim against text-only DeBERTa depends on this. Corrupting
    every tabular column — including the two the baseline actually uses — must not
    change a single predicted probability.
    """
    df = _toy_frame()
    rep, head = _fit_toy_dense(df)
    clean = variant_probabilities(rep, head, df)

    poisoned = df.copy()
    for col in cfg.TABULAR_FEATURES + ["stars"]:
        if col in poisoned:
            poisoned[col] = -999.0
    assert np.array_equal(clean, variant_probabilities(rep, head, poisoned))


def test_probabilities_are_calibrated_range_and_right_length():
    df = _toy_frame()
    rep, head = _fit_toy_dense(df)
    probs = variant_probabilities(rep, head, df)
    assert probs.shape == (len(df),)
    assert probs.min() >= 0.0 and probs.max() <= 1.0
    # Sanity, not performance: the two classes are lexically disjoint here, so a
    # representation that cannot separate them is broken rather than merely weak.
    y = df[cfg.TARGET_COLUMN].values
    assert probs[y == 1].mean() > probs[y == 0].mean()


def test_dense_representations_have_the_requested_width():
    """A dense document embedding must be dense and of the declared dimension."""
    texts = (HAZARD + BENIGN) * 4
    lsa = LsaRepresentation(seed=cfg.RANDOM_STATE, n_components=8).fit(texts)
    out = lsa.transform(texts)
    assert out.shape == (len(texts), 8)
    assert not hasattr(out, "toarray"), "LSA output must be dense"

    d2v = Doc2VecRepresentation(seed=cfg.RANDOM_STATE, vector_size=16, epochs=3).fit(texts)
    assert d2v.transform(texts[:3]).shape == (3, 16)


def test_transform_does_not_refit_on_evaluation_text():
    """
    The subtle leak: if transform() re-fitted anything, evaluation text would enter
    the representation. Fit on hazard text only, then transform benign text, and the
    fitted vocabulary must be unchanged.
    """
    rep = LsaRepresentation(seed=cfg.RANDOM_STATE, n_components=8)
    rep.fit(HAZARD * 3)
    vocab_before = dict(rep.vec.vocabulary_)
    components_before = rep.svd.components_.copy()
    rep.transform(BENIGN * 3)
    assert rep.vec.vocabulary_ == vocab_before
    assert np.array_equal(rep.svd.components_, components_before)


def test_unknown_variant_is_rejected():
    try:
        build_representation("word2vec_by_hand")
    except ValueError:
        return
    raise AssertionError("expected ValueError for an unknown variant")


def test_declared_variants_all_build():
    """
    Every name in VARIANTS must be constructible, except that `minilm` is allowed to
    fail on a box with no cached model and no network — that is the documented
    fallback path, and it must degrade rather than crash the module import.
    """
    from src.embedding_model import VARIANTS

    for v in VARIANTS:
        rep = build_representation(v)
        assert rep.name == v
        assert hasattr(rep, "fit") and hasattr(rep, "transform") and hasattr(rep, "describe")


def test_metrics_come_from_the_shared_evaluation_module():
    """
    The whole comparison is void if this module re-implements the metrics. Assert the
    identity of the function object, not the value of a formula.
    """
    import analysis.evaluation_pipeline as ep
    import src.embedding_model as em

    assert em.score_variant is ep.score_variant
    assert em.COST_FALSE_NEGATIVE == ep.COST_FALSE_NEGATIVE == 5000.0
    assert em.COST_FALSE_POSITIVE == ep.COST_FALSE_POSITIVE == 50.0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
