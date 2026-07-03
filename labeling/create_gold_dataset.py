"""
Build a hand-checked (LLM-as-judge) gold dataset for `is_hazard`.

Why this exists (see CLAUDE.md "Known risks / gaps"): the current `is_hazard` label
is a keyword+stars heuristic (see preprocessing/final_project_preprocessing.ipynb),
which is noisy and risks label leakage since the model is also fed lexicon features
built from the same keyword list. This script samples ~750 reviews from the already
cleaned/enriched dataset and re-labels them with an LLM-as-judge prompt that asks
"is there an actually described health/food-safety hazard here?" — producing a much
higher-quality ground truth to evaluate the heuristic against and/or to fine-tune on.

LLM provider: free, no-cost APIs only (course project, no budget). Two supported:
  - Groq   (default) — https://console.groq.com/keys — generous free tier, fast,
    OpenAI-compatible REST API, runs a large open-weights model (Llama 3.3 70B).
  - Gemini            — https://aistudio.google.com/apikey — Google's free tier
    (gemini-2.0-flash / gemini-1.5-flash).
Both are genuinely free (not a trial credit) and backed by well-known model
providers, which is why they were picked over an anonymous/unvetted free API.

Usage:
    pip install requests python-dotenv   # both already in requirements.txt
    # put GROQ_API_KEY=... (or GEMINI_API_KEY=...) in a .env file at the repo root
    python labeling/create_gold_dataset.py --provider groq --n 750

Resumable: progress is written incrementally to --out, and re-running the script
skips reviews already present in that file (matched by row index), so a rate-limit
error or Ctrl-C does not lose work.
"""

import argparse
import json
import os
import re
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config.settings as cfg

load_dotenv()

LABELING_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(LABELING_DIR, "gold_dataset.csv")

# ---------------------------------------------------------------------------
# LLM-as-judge prompt
# ---------------------------------------------------------------------------
# Deliberately strict: a low star rating or a generic complaint ("bad food",
# "rude staff", "overpriced") is NOT a hazard. Only an *actually described*
# health/food-safety incident counts. This is the distinction the regex+stars
# heuristic cannot make (it fires on keyword presence + low rating alone).
SYSTEM_PROMPT = """You are a food-safety domain expert acting as a strict annotator \
for a research dataset. You will be shown one Yelp restaurant review. Your job is \
to judge whether the review describes a REAL, CONCRETE health or food-safety hazard \
that happened to the reviewer or someone they saw — not a complaint about taste, \
service, price, ambiance, or a low star rating by itself.

Label as a hazard (is_hazard=1) ONLY if the review describes at least one of:
- The reviewer or a companion got sick, had food poisoning, or an allergic/anaphylactic \
reaction after eating there.
- Physical contamination: a foreign object (hair, glass, plastic, insect, etc.) found in food.
- Unsafe food handling that is directly described (e.g., visibly spoiled/rotten food served, \
raw/undercooked meat served, food left out and clearly unsafe, cross-contamination with an \
allergen that caused a reaction).
- The reviewer had to seek medical attention, went to a hospital, or used an EpiPen because \
of the food.

Do NOT label as a hazard (is_hazard=0):
- Generic negative reviews (bad taste, cold food, slow service, rude staff, overpriced) \
with no described illness or contamination.
- Low star ratings alone.
- Hypothetical or general safety commentary not tied to a specific incident in this review \
("you should always check kitchens are clean").
- Mentions of allergies/gluten-free/etc. as a *neutral factual note* (e.g., "they have a \
gluten-free menu", "I'm allergic to shellfish so I got the chicken") with no adverse event.

Respond with ONLY a compact JSON object, no markdown, no explanation outside the JSON:
{"is_hazard": 0 or 1, "hazard_type": "allergic_reaction" | "food_poisoning" | "contamination" | "unsafe_handling" | "none", "confidence": "high" | "medium" | "low", "rationale": "one short sentence"}"""

USER_TEMPLATE = "Yelp review (star rating given by reviewer: {stars}):\n\"\"\"\n{text}\n\"\"\""


# ---------------------------------------------------------------------------
# Provider calls
# ---------------------------------------------------------------------------
class RateLimitError(Exception):
    """Raised on HTTP 429 so callers can honor the server's own Retry-After."""

    def __init__(self, retry_after, detail=""):
        self.retry_after = retry_after
        self.detail = detail  # raw response body — carries the *actual* quota metric hit
        super().__init__(f"rate limited by provider, retry after {retry_after}s")


class QuotaUnavailableError(Exception):
    """
    Raised when the provider reports a hard `limit: 0` quota — not a temporary
    throttle that resets, but zero quota granted at all (billing linked to the
    project, or the account's region isn't in the free-tier eligible list).
    No amount of retrying fixes this, so it is NOT retried like RateLimitError.
    """


# Google's free-tier "no quota granted" response includes this exact phrase
# with the numeral 0 for the limit — distinguishing it from a real, resettable
# rate limit (which reports a nonzero limit and a shrinking remaining count).
_ZERO_QUOTA_PATTERN = re.compile(r"limit:\s*0\b")


def _raise_for_rate_limit(resp):
    if resp.status_code == 429:
        if _ZERO_QUOTA_PATTERN.search(resp.text):
            raise QuotaUnavailableError(
                "Provider reports zero free-tier quota (limit: 0) — this is not a "
                "temporary rate limit and will not resolve by waiting. This usually "
                "means either (a) the API key's project has a billing account linked, "
                "which makes it ineligible for the free tier, or (b) your account's "
                "region isn't in the free-tier eligible list. Check "
                "https://aistudio.google.com/apikey (should say \"Free\", not \"Paid\") "
                "and whether the underlying Google Cloud project has billing enabled.\n"
                f"Full response: {resp.text[:500]}"
            )
        # Respect the server's own back-off instruction instead of guessing — but
        # ALSO keep the raw body: the Retry-After header alone can't distinguish
        # "back off 30s, per-minute limit" from "your daily quota is exhausted,
        # 30s is just a generic hint" — the body's error message/quota metric name
        # tells you which one it actually is.
        raise RateLimitError(float(resp.headers.get("retry-after", 30)), detail=resp.text[:500])
    resp.raise_for_status()


def call_groq(text, stars, model, api_key, timeout=30):
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(stars=stars, text=text[:4000])},
            ],
        },
        timeout=timeout,
    )
    _raise_for_rate_limit(resp)
    content = resp.json()["choices"][0]["message"]["content"]
    # Groq echoes back live quota state on every response — surface it so the
    # user can see the free-tier budget draining in real time, not just guess.
    quota = {
        "remaining_requests": resp.headers.get("x-ratelimit-remaining-requests"),
        "remaining_tokens": resp.headers.get("x-ratelimit-remaining-tokens"),
    }
    return content, quota


def call_gemini(text, stars, model, api_key, timeout=30):
    resp = requests.post(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        json={
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {"role": "user", "parts": [{"text": USER_TEMPLATE.format(stars=stars, text=text[:4000])}]}
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
            },
        },
        timeout=timeout,
    )
    _raise_for_rate_limit(resp)
    content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    # Gemini's generateContent response doesn't expose remaining-quota headers
    # (quota is tracked project-side in AI Studio / Cloud console, not per-call).
    return content, {}


# Free-tier ceilings as documented at time of writing (both providers can and
# do change these without notice — check console.groq.com / aistudio.google.com
# if you hit unexpected 429s). `max_requests` is set comfortably BELOW the
# documented daily cap, not at it, so a single run never risks tipping the
# account into a temporary block; the script is resumable, so hitting the cap
# just means "finish the rest tomorrow."
PROVIDERS = {
    "groq": {
        "call": call_groq,
        # llama-3.3-70b-versatile's free-tier TPD (tokens/day) is only 100,000 —
        # at ~550 tokens/call (long system prompt + review text) that's only
        # ~180 calls/day, nowhere near enough for a 750-row run. 8B models get a
        # much larger daily token budget on Groq's free tier since they're
        # cheaper to serve, and are plenty capable for this yes/no classification
        # + short JSON output task — no need for the 70B model here.
        "default_model": "llama-3.1-8b-instant",
        "env_var": "GROQ_API_KEY",
        # Live-checked: this model's TPM (tokens/minute) budget is 6,000, refilled
        # continuously. At ~550 tokens/call, sustained pacing needs >=5.5s between
        # calls to stay under it — 6s leaves a safety margin.
        "sleep_s": 6.0,
        "max_requests": 900,
    },
    "gemini": {
        "call": call_gemini,
        "default_model": "gemini-2.0-flash",
        "env_var": "GEMINI_API_KEY",
        "sleep_s": 4.5,        # ~13 req/min, under the 15 req/min free-tier RPM limit
        "max_requests": 1400,  # documented free-tier RPD is ~1,500 for this model
    },
}


def judge_one(text, stars, provider_cfg, model, api_key, max_error_retries=4, max_rate_limit_waits=20):
    """
    Call the LLM judge. Returns (result_dict, quota_dict).

    Rate-limit waits and genuine errors are budgeted separately and handled
    differently: a 429 isn't a failure, it's the provider telling you exactly
    how long to wait — so it's always honored in full (sanity-capped at 10 min
    against a malformed header) with a generous retry budget, since waiting
    costs nothing on a free tier. A capped/short-circuited wait here previously
    caused premature retries that kept re-triggering 429 until retries ran out
    and the whole run crashed even though the quota would have recovered fine.
    Genuine errors (malformed JSON, network blips, 5xx) still bail out quickly
    so a persistently broken request doesn't loop forever.
    """
    error_attempts = 0
    rate_limit_waits = 0
    while True:
        try:
            raw, quota = provider_cfg["call"](text, stars, model, api_key)
            parsed = json.loads(raw)
            return {
                "llm_is_hazard": int(parsed["is_hazard"]),
                "llm_hazard_type": parsed.get("hazard_type", "none"),
                "llm_confidence": parsed.get("confidence", "low"),
                "llm_rationale": parsed.get("rationale", ""),
            }, quota
        except QuotaUnavailableError:
            # Zero granted quota doesn't reset with time — retrying wastes minutes
            # for nothing. Fail immediately with the actionable message attached.
            raise
        except RateLimitError as e:
            rate_limit_waits += 1
            if rate_limit_waits == 1 and e.detail:
                # Print the raw error body once — this is where the *actual* quota
                # metric lives (e.g. "requests per day" vs "requests per minute").
                # The Retry-After header alone can't tell those apart.
                print(f"    rate limit response body: {e.detail}")
            if rate_limit_waits > max_rate_limit_waits:
                raise RuntimeError(
                    f"Still rate limited after {max_rate_limit_waits} waits "
                    f"(last wait requested: {e.retry_after:.0f}s). The free-tier quota "
                    f"may be exhausted for longer than usual — stop here and resume "
                    f"this exact command later; already-labeled rows are preserved. "
                    f"Last response body: {e.detail}"
                ) from e
            wait = min(e.retry_after, 600)  # honor the real wait; sanity cap only against bad headers
            print(f"    rate limited — sleeping {wait:.0f}s (provider said {e.retry_after:.0f}s) "
                  f"[wait {rate_limit_waits}/{max_rate_limit_waits}]")
            time.sleep(wait)
        except Exception as e:  # transient 5xx, malformed JSON, network blip, etc.
            error_attempts += 1
            if error_attempts > max_error_retries:
                raise RuntimeError(f"LLM judge failed after {max_error_retries} retries: {e}") from e
            wait = (2 ** error_attempts) * 2
            print(f"    retry {error_attempts}/{max_error_retries} after error: {e} (sleeping {wait}s)")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def build_sample(df, n, seed):
    """
    Stratified 50/50 sample on the existing weak `is_hazard` label.

    Rationale: the raw class balance is ~20% positive / 80% negative. A gold set
    drawn in that same proportion would only contain ~150 positives out of 750,
    too few to reliably estimate the heuristic's precision on the hazard class.
    Oversampling positives to 50/50 gives enough examples of both the heuristic's
    false positives (label=1, LLM says 0) and false negatives (label=0, LLM says 1)
    to actually characterize where the keyword+stars rule breaks down.
    """
    half = n // 2
    pos = df[df[cfg.TARGET_COLUMN] == 1]
    neg = df[df[cfg.TARGET_COLUMN] == 0]

    n_pos = min(half, len(pos))
    n_neg = min(n - n_pos, len(neg))

    sample = pd.concat([
        pos.sample(n=n_pos, random_state=seed),
        neg.sample(n=n_neg, random_state=seed),
    ])
    return sample.sample(frac=1, random_state=seed)  # shuffle


def build_holdout_sample(df, n, seed):
    """
    Plain random sample from the unseen holdout pool (labeling/build_holdout_pool.py).

    No stratification here: these rows were never scored by the keyword+stars
    heuristic, so there is no existing label to balance against. They were
    already pre-filtered to be allergy/hazard-adjacent when the pool was built,
    which is what gives the sample its "chance of hazard" instead of the ~2-5%
    base rate you'd get sampling raw Yelp reviews directly.
    """
    n = min(n, len(df))
    return df.sample(n=n, random_state=seed)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", choices=PROVIDERS.keys(), default="groq")
    parser.add_argument("--model", default=None, help="Override the provider's default model")
    parser.add_argument("--n", type=int, default=750, help="Gold set size (default 750)")
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_STATE)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--max-requests", type=int, default=None,
        help="Hard cap on API calls this run, to stay under the provider's free-tier "
             "daily limit (defaults to a conservative per-provider value). The run stops "
             "cleanly and can be resumed later — it does not lose progress.",
    )
    parser.add_argument(
        "--source", default="enriched",
        help="'enriched' (default) samples 50/50 from the existing labeled dataset. "
             "Pass a path to a holdout pool CSV (e.g. labeling/holdout_candidate_pool.csv, "
             "built by build_holdout_pool.py) to sample from never-before-seen reviews instead.",
    )
    args = parser.parse_args()

    provider_cfg = PROVIDERS[args.provider]
    model = args.model or provider_cfg["default_model"]
    api_key = os.getenv(provider_cfg["env_var"])
    if not api_key:
        sys.exit(
            f"Missing {provider_cfg['env_var']} in environment/.env. "
            f"Get a free key: "
            f"{'https://console.groq.com/keys' if args.provider == 'groq' else 'https://aistudio.google.com/apikey'}"
        )

    is_holdout = args.source != "enriched"
    if is_holdout:
        if not os.path.exists(args.source):
            sys.exit(f"Holdout pool not found: {args.source}. Run build_holdout_pool.py first.")
        df = pd.read_csv(args.source)
        df[cfg.TEXT_COLUMN] = df[cfg.TEXT_COLUMN].fillna("")
        sample = build_holdout_sample(df, args.n, args.seed)
        has_heuristic_label = False
    else:
        df = pd.read_csv(cfg.INPUT_DATA_PATH)
        df[cfg.TEXT_COLUMN] = df[cfg.TEXT_COLUMN].fillna("")
        sample = build_sample(df, args.n, args.seed)
        has_heuristic_label = True

    out_path = args.out or (
        os.path.join(LABELING_DIR, "gold_dataset_holdout.csv") if is_holdout else DEFAULT_OUT
    )
    args.out = out_path

    # Resume support: keep already-labeled rows (matched by original df index).
    done = pd.DataFrame()
    if os.path.exists(args.out):
        done = pd.read_csv(args.out, index_col="source_index")
        print(f"Resuming: {len(done)} rows already labeled in {args.out}")

    max_requests = args.max_requests or provider_cfg["max_requests"]

    results = [done] if not done.empty else []
    remaining = sample[~sample.index.isin(done.index)]
    print(f"Labeling {len(remaining)}/{len(sample)} remaining reviews with "
          f"{args.provider}:{model} (budget: {max_requests} calls this run) ...")

    n_calls = 0
    hit_budget = False
    for i, (idx, row) in enumerate(remaining.iterrows(), 1):
        if n_calls >= max_requests:
            hit_budget = True
            print(f"\nHit --max-requests={max_requests} for this run — stopping to stay "
                  f"under the free-tier daily limit. {len(remaining) - i + 1} rows left; "
                  f"re-run this exact command (tomorrow, or after the quota resets) to "
                  f"continue — already-labeled rows are skipped automatically.")
            break

        judged, quota = judge_one(row[cfg.TEXT_COLUMN], row["stars"], provider_cfg, model, api_key)
        n_calls += 1
        record = row.to_dict()
        record["source_index"] = idx
        record.update(judged)
        results.append(pd.DataFrame([record]).set_index("source_index"))

        # Write after every row so a crash/rate-limit only costs the in-flight call.
        pd.concat(results).to_csv(args.out, index_label="source_index")

        quota_str = ""
        if quota.get("remaining_requests") is not None:
            quota_str = f" | quota left: {quota['remaining_requests']} reqs, {quota.get('remaining_tokens', '?')} tokens"
        if has_heuristic_label:
            agree = "match" if judged["llm_is_hazard"] == row[cfg.TARGET_COLUMN] else "DISAGREE"
            print(f"  [{i}/{len(remaining)}] heuristic={row[cfg.TARGET_COLUMN]} "
                  f"llm={judged['llm_is_hazard']} ({agree}) conf={judged['llm_confidence']}{quota_str}")
        else:
            print(f"  [{i}/{len(remaining)}] llm={judged['llm_is_hazard']} "
                  f"conf={judged['llm_confidence']}{quota_str}")

        time.sleep(provider_cfg["sleep_s"]+1)

    final = pd.concat(results)
    print("\n" + "=" * 60)
    print(f"Gold dataset written to: {args.out}")
    print(f"Rows: {len(final)}" + (" (INCOMPLETE — budget cap reached, resume later)" if hit_budget else ""))
    if has_heuristic_label:
        n_agree = (final[cfg.TARGET_COLUMN] == final["llm_is_hazard"]).sum()
        print(f"Heuristic label vs. LLM judge agreement: {n_agree}/{len(final)} "
              f"({100 * n_agree / len(final):.1f}%)")
        print(f"LLM-judged hazard rate: {final['llm_is_hazard'].mean():.1%} "
              f"(heuristic label hazard rate in this sample: {final[cfg.TARGET_COLUMN].mean():.1%})")
    else:
        print(f"LLM-judged hazard rate on this never-before-seen sample: "
              f"{final['llm_is_hazard'].mean():.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
