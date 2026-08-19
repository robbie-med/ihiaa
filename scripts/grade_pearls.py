"""Score every pearl against a binary rubric, and measure the rubric's reliability.

FIRST ATTEMPT FAILED, AND THE FAILURE IS WHY THIS FILE LOOKS LIKE IT DOES.

v1 asked for four 0-3 graded scales (actionability, specificity, generalizability,
consequence). Measured test-retest on 60 pearls:

    composite exact agreement   17%
    same letter grade           53%
    mean abs difference       1.67 / 12

53% letter agreement is a coin flip. A graded scale asks the model to place fuzzy
boundaries ("is this a 2 or a 3?") that it will not place the same way twice, so the
number looked objective while behaving randomly. Two contributing bugs: temperature
was 0.2 rather than 0, and oversized batches truncated the JSON and silently lost
scores.

v2 replaces every scale with a BINARY question. "Does it name a drug, dose, threshold
or maneuver?" has a defensible yes/no answer; "rate specificity 0-3" does not. Binary
decisions are far more reproducible, and the composite is then computed in Python, so
the model never picks the grade.

    Q1 specific     names a drug, dose, numeric threshold, test, or maneuver
    Q2 actionable   tells a clinician what to DO, not merely what is true
    Q3 local        site-specific logistics (storage, paging, scheduling, room names)
    Q4 harm         not knowing it risks patient harm or a missed diagnosis
    Q5 nonobvious   a competent intern would not already know it

    score = Q1 + Q2 + Q4 + Q5          (0-4)
    Q3 is a VETO -- any yes forces D, whatever else the pearl scores.

    A = 4    B = 3    C = 2    D <= 1 or vetoed

D is excluded from the site; C is retained but demoted below A/B.

Reliability is MEASURED, never asserted: --reliability re-scores a fixed random sample
and reports per-question agreement plus letter-grade stability. If that number is poor,
the rubric is subjective no matter how principled it reads, and it should not be used.
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import ppq  # noqa: E402

EPISODES = BASE / "data" / "episodes"
RUBRIC_VERSION = "rubric-v2-binary"
QUESTIONS = ("specific", "actionable", "local", "harm", "nonobvious")

SYS = """You screen candidate "clinical pearls" pulled from residency lectures.

Answer FIVE yes/no questions per pearl. Answer only from the text shown. Do not infer
facts that are not there, and do not reward how well it is written.

"specific"   : true if it names a drug, a dose, a numeric threshold, a named test,
               a scoring tool, or a concrete physical maneuver. Otherwise false.
"actionable" : true if it tells a clinician what to DO or DECIDE. False if it only
               states what is true (background, epidemiology, pathophysiology).
"local"      : true if it is operational trivia specific to one institution -- where
               supplies are stored, who to page, clinic scheduling, room or unit
               names, this program's internal workflow. Otherwise false.
"harm"       : true if a clinician not knowing this could plausibly lead to patient
               harm, a missed diagnosis, or a dangerous drug interaction.
"nonobvious" : true if a competent intern would NOT already reliably know it.
               False for common knowledge ("diabetes causes neuropathy").

Return JSON only, same order as input:
{"scores":[{"id":"...","specific":true,"actionable":true,"local":false,
            "harm":false,"nonobvious":true}]}"""


def load_all() -> list:
    out = []
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        for p in d.get("pearls", []):
            out.append((f, p))
    return out


def grade_batch(batch: list, model: str = None) -> dict:
    payload = {"pearls": [{"id": p["pearl_id"], "text": p["text"]} for p in batch]}
    r = ppq.chat_json(SYS, json.dumps(payload, ensure_ascii=False),
                      model=model, max_tokens=3000, temperature=0.0)
    out = {}
    for s in r.get("scores", []):
        if s.get("id"):
            out[s["id"]] = {q: bool(s.get(q, False)) for q in QUESTIONS}
    return out


def score_pearls(pearls: list, size: int = 8, model: str = None) -> dict:
    """Small batches: v1 lost a whole batch to truncated JSON at size 15."""
    got = {}
    for i in range(0, len(pearls), size):
        try:
            got.update(grade_batch(pearls[i:i + size], model=model))
        except Exception as e:                        # noqa: BLE001
            print(f"  ! {type(e).__name__}: {str(e)[:70]}", flush=True)
        if (i // size) % 15 == 0:
            print(f"  scored {min(i+size, len(pearls))}/{len(pearls)}", flush=True)
    return got


def grade_of(s: dict) -> tuple:
    if s.get("local"):                       # veto
        return 0, "D"
    n = sum(1 for q in ("specific", "actionable", "harm", "nonobvious") if s.get(q))
    return n, ("A" if n == 4 else "B" if n == 3 else "C" if n == 2 else "D")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reliability", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=None, help="override grading model")
    ap.add_argument("--apply", action="store_true",
                    help="write scores into data/episodes/*.json")
    a = ap.parse_args()

    items = load_all()
    if a.limit:
        items = items[:a.limit]
    pearls = [p for _, p in items]
    print(f"{len(pearls)} pearls | model={a.model or ppq.CHAT_MODEL} | {RUBRIC_VERSION}",
          flush=True)

    scores = score_pearls(pearls, model=a.model)
    print(f"got {len(scores)} scores", flush=True)

    if a.reliability:
        rng = random.Random(12345)
        pool = [p for p in pearls if p["pearl_id"] in scores]
        sample = rng.sample(pool, min(a.reliability, len(pool)))
        print(f"\nre-scoring {len(sample)} pearls ...", flush=True)
        second = score_pearls(sample, model=a.model)

        per_q = Counter()
        both = same_grade = 0
        for p in sample:
            s1, s2 = scores.get(p["pearl_id"]), second.get(p["pearl_id"])
            if not (s1 and s2):
                continue
            both += 1
            for q in QUESTIONS:
                per_q[q] += (s1[q] == s2[q])
            same_grade += grade_of(s1)[1] == grade_of(s2)[1]
        n = both or 1
        print(f"\n--- RELIABILITY (n={both}) ---")
        for q in QUESTIONS:
            print(f"  {q:12s} agreement : {100*per_q[q]/n:5.0f}%")
        print(f"  {'LETTER GRADE':12s} stability : {100*same_grade/n:5.0f}%")
        if 100 * same_grade / n < 85:
            print("\n  ** below 85% -- rubric is NOT reliable enough to apply **")
        else:
            print("\n  reliable enough to apply")

    if a.apply:
        dist = Counter()
        for f in {f for f, _ in items}:
            d = json.loads(f.read_text())
            changed = False
            for p in d.get("pearls", []):
                s = scores.get(p["pearl_id"])
                if not s:
                    continue
                n, g = grade_of(s)
                p["score"] = {**s, "points": n, "grade": g, "rubric": RUBRIC_VERSION}
                dist[g] += 1
                changed = True
            if changed:
                f.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        print("\n--- GRADE DISTRIBUTION ---")
        tot = max(sum(dist.values()), 1)
        for g in "ABCD":
            print(f"  {g}: {dist[g]:5d}  ({100*dist[g]/tot:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
