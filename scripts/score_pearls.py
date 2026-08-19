"""Deterministic, rule-based grading of pearls. No model in the loop.

An earlier version asked an LLM to grade. Measured test-retest was 53% letter-grade
agreement -- a coin flip -- so the score looked objective while behaving randomly.
Grading is classification over surface features, and those features are detectable
with rules, so this scorer is:

  * reproducible BY CONSTRUCTION -- same input always yields the identical output,
    which is a property of the code rather than something to measure and hope for
  * auditable -- every pearl records exactly which rules fired and on which token,
    so any grade can be explained and any bad rule can be found and fixed
  * free and instant over the whole corpus, so it can be re-run on every build

Five signals, four scored plus one veto:

    specific      numbers with units, thresholds, doses, frequencies, named tests,
                  drug-name morphology, or a known clinical concept from the corpus
    actionable    a directive verb -- start / stop / avoid / check / refer / taper
    local  (VETO) institution-specific logistics: storage, paging, scheduling,
                  rooms, "our clinic". Any hit forces grade D outright.
    harm          risk vocabulary -- contraindicated, fatal, missed diagnosis,
                  precipitates withdrawal, black box
    nonobvious    rarity: uses terms that are uncommon ACROSS this corpus, measured
                  by document frequency. Boilerplate scores low automatically.

    points = specific + actionable + harm + nonobvious      (0-4)
    A = 4    B = 3    C = 2    D <= 1, or any local hit

D is excluded from the site; C is retained but demoted below A/B.

Run `--explain` to see fired rules, or `--selftest` to verify determinism and check
the known-bad case ("where the medications are stored") lands in D.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EPISODES = BASE / "data" / "episodes"
RUBRIC_VERSION = "rules-v1"

# --------------------------------------------------------------------------
# Lexicons. Deliberately explicit and editable -- these ARE the rubric, and a
# reader should be able to disagree with a grade by pointing at a line here.
# --------------------------------------------------------------------------
UNITS = (r"mg|mcg|µg|ug|g|kg|mL|ml|L|units?|IU|mEq|mmol|mmHg|%|cm|mm|"
         r"hours?|hrs?|days?|weeks?|months?|years?|minutes?|mins?")

RE_DOSE = re.compile(rf"\b\d+(?:\.\d+)?\s*(?:{UNITS})\b", re.I)
RE_THRESHOLD = re.compile(
    r"(?:less than|greater than|more than|below|above|at least|no more than|"
    r"under|over|goal of|target of|cutoff|threshold)\s*\d"
    r"|[<>]\s*\d|\d+\s*/\s*\d+\b", re.I)
RE_FREQ = re.compile(r"\b(?:BID|TID|QID|QHS|PRN|daily|twice a day|once a day|"
                     r"every \d+|per day|per week|weekly|monthly|q\d+h)\b", re.I)
RE_DRUGMORPH = re.compile(
    r"\b\w{3,}(?:cillin|mycin|micin|statin|pril|sartan|olol|azole|prazole|"
    r"triptan|gabalin|floxacin|cycline|parin|azepam|azolam|codone|morphone|"
    r"fentanil|fentanyl|glutide|gliflozin|gliptin|tidine|setron|caine|"
    r"vastatin|dipine|osin|terol|sone|solone|nib|mab)\b", re.I)
RE_TEST = re.compile(
    r"\b(?:A1C|HbA1c|TSH|CBC|BMP|CMP|LFTs?|INR|PT|PTT|BNP|CRP|ESR|GFR|eGFR|"
    r"LDL|HDL|EKG|ECG|CT|MRI|ultrasound|x-?ray|biopsy|culture|PHQ-?9|GAD-?7|"
    r"AUDIT|CAGE|MMSE|MoCA|ABI|monofilament|spirometry|colonoscopy|"
    r"mammogram|pap smear|Wells|CHA2DS2|FRAX|ASCVD)\b", re.I)

RE_ACTION = re.compile(
    r"\b(?:start|starting|initiate|give|administer|prescribe|order|check|obtain|"
    r"screen|test for|avoid|don'?t|do not|never|always|stop|discontinue|hold|"
    r"switch|change to|add|titrate|taper|increase|decrease|reduce|refer|admit|"
    r"consult|treat|repeat|confirm|rule out|monitor|follow up|recheck|use|"
    r"consider|ensure|make sure|document|counsel|educate|examine|palpate|"
    r"inspect|measure|calculate|ask about|evaluate|reassess|wait)\b", re.I)

# The veto. Institution-specific operational trivia.
RE_LOCAL = re.compile(
    r"\b(?:stored|storage|store the|storing|kept in|we keep|they keep|"
    r"our clinic|our office|our program|our residency|our hospital|our system|"
    r"our EMR|this clinic|this office|the day cent(?:er|re)|day cent(?:er|re)|"
    r"front desk|the pharmacy here|pager|page the|paging|"
    r"supply (?:closet|room)|the cabinet|the fridge|refrigerator in|"
    r"room \d+|clinic schedule|scheduling|schedules? (?:are|is)|"
    r"downstairs|upstairs|down the hall|our nurses|the nurses here|"
    r"who to call|call the front|sign(?:-| )?out sheet|the binder)\b", re.I)

RE_HARM = re.compile(
    r"\b(?:fatal|lethal|death|die|dying|kill|life-?threatening|"
    r"contraindicat\w+|black box|boxed warning|"
    r"missed?|miss the|delay(?:ed)? diagnosis|"
    r"precipitat\w+|withdrawal|overdose|toxicity|toxic|"
    r"h(?:a)?emorrhag\w+|bleed\w*|sepsis|septic|shock|arrest|"
    r"stroke|infarct\w*|perforat\w+|necrosis|necrotic|gangrene|"
    r"amputat\w+|blind\w*|seizure|anaphyla\w+|"
    r"emergency|emergent|urgent|immediately|dangerous|serious harm|"
    r"never give|do not give|interaction)\b", re.I)

STOP = set("""a an the and or but if then than that this these those is are was were be been
being of to in on for with without at by from as it its it's you your we our they their
he she his her them us i me my not no do does did doing done can could should would may
might will shall have has had about into over under more most less least very just also
so such other another each any all some when while because there here what which who whom
whose how why patient patients get got getting make makes made take takes taken taking
use used using go goes going come comes see sees seen look looks like really lot lots
thing things stuff kind sort way ways time times good bad big small new old""".split())

TOKEN = re.compile(r"[a-z][a-z0-9-]{2,}")


def tokens(text: str) -> set:
    return {t for t in TOKEN.findall(text.lower()) if t not in STOP}


def load_all() -> list:
    out = []
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        for p in d.get("pearls", []):
            out.append((f, p))
    return out


def build_idf(pearls: list) -> dict:
    """Document frequency across pearls. Rare vocabulary -> likely non-obvious."""
    df = Counter()
    for p in pearls:
        df.update(tokens(p["text"]))
    n = max(len(pearls), 1)
    return {t: math.log(n / c) for t, c in df.items()}


def score_one(p: dict, idf: dict, concepts: set, rare_cut: float) -> dict:
    text = p.get("text", "")
    fired = {}

    hits = []
    for name, rx in (("dose", RE_DOSE), ("threshold", RE_THRESHOLD),
                     ("frequency", RE_FREQ), ("drug", RE_DRUGMORPH),
                     ("test", RE_TEST)):
        m = rx.search(text)
        if m:
            hits.append(f"{name}:{m.group(0)[:22]}")
    concept_hit = next((c for c in concepts if c and c in text.lower()), None)
    if concept_hit and not hits:
        hits.append(f"concept:{concept_hit[:22]}")
    specific = bool(hits)
    if hits:
        fired["specific"] = hits[:3]

    m = RE_ACTION.search(text)
    actionable = bool(m)
    if m:
        fired["actionable"] = [m.group(0)]

    m = RE_LOCAL.search(text)
    local = bool(m)
    if m:
        fired["local"] = [m.group(0)]

    m = RE_HARM.search(text)
    harm = bool(m)
    if m:
        fired["harm"] = [m.group(0)]

    # Rarity: mean IDF of the pearl's three rarest content words.
    ts = tokens(text)
    top = sorted((idf.get(t, 0.0) for t in ts), reverse=True)[:3]
    rarity = sum(top) / len(top) if top else 0.0
    nonobvious = rarity >= rare_cut
    fired["rarity"] = round(rarity, 2)

    points = sum((specific, actionable, harm, nonobvious))
    grade = "D" if local else ("A" if points == 4 else "B" if points == 3
                               else "C" if points == 2 else "D")
    return {"specific": specific, "actionable": actionable, "local": local,
            "harm": harm, "nonobvious": nonobvious, "points": points,
            "grade": grade, "fired": fired, "rubric": RUBRIC_VERSION}


def concept_set() -> set:
    """Clinical concepts the corpus itself already named (topic labels)."""
    site = BASE / "site" / "data.json"
    if not site.exists():
        return set()
    d = json.loads(site.read_text())
    return {t["label"].lower() for t in d.get("topics", [])
            if len(t.get("label", "")) > 4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--explain", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rare-cut", type=float, default=4.0)
    a = ap.parse_args()

    items = load_all()
    pearls = [p for _, p in items]
    if not pearls:
        print("no pearls found")
        return 1
    idf = build_idf(pearls)
    concepts = concept_set()
    print(f"{len(pearls)} pearls | {len(concepts)} known concepts | {RUBRIC_VERSION}")

    scored = [(f, p, score_one(p, idf, concepts, a.rare_cut)) for f, p in items]

    if a.selftest:
        # 1. determinism: scoring twice must be byte-identical
        again = [score_one(p, idf, concepts, a.rare_cut) for _, p in items]
        same = all(json.dumps(x[2], sort_keys=True) == json.dumps(y, sort_keys=True)
                   for x, y in zip(scored, again))
        print(f"\n  determinism (2 runs identical) : {'PASS' if same else 'FAIL'}")

        # 2. the known-bad case must be vetoed
        cases = [
            ("The medications are stored in the cabinet at the day center.", "D"),
            ("Start metformin 500 mg with dinner to reduce GI upset.", None),
            ("Wait 14 days from last opioid use before starting naltrexone or you "
             "will precipitate withdrawal.", None),
        ]
        for txt, want in cases:
            s = score_one({"text": txt}, idf, concepts, a.rare_cut)
            ok = "" if want is None else ("PASS" if s["grade"] == want else "FAIL")
            print(f"  {s['grade']} pts={s['points']} {ok:4s} | {txt[:58]}")
            print(f"        fired: {s['fired']}")

    dist = Counter(s["grade"] for _, _, s in scored)
    print("\n--- GRADE DISTRIBUTION ---")
    for g in "ABCD":
        print(f"  {g}: {dist[g]:5d}  ({100*dist[g]/len(scored):.1f}%)")
    vetoed = sum(1 for _, _, s in scored if s["local"])
    print(f"  ({vetoed} vetoed as institution-local)")

    if a.explain:
        print(f"\n--- SAMPLE (first {a.explain} of each grade) ---")
        for g in "ABCD":
            print(f"\n  == {g} ==")
            for _, p, s in [x for x in scored if x[2]["grade"] == g][:a.explain]:
                print(f"   {p['text'][:88]}")
                print(f"     {s['fired']}")

    if a.apply:
        for f in {f for f, _, _ in scored}:
            d = json.loads(f.read_text())
            byid = {p["pearl_id"]: s for _, p, s in scored if _ == f}
            for p in d.get("pearls", []):
                if p["pearl_id"] in byid:
                    p["score"] = byid[p["pearl_id"]]
            f.write_text(json.dumps(d, indent=2, ensure_ascii=False))
        print(f"\napplied to {len({f for f,_,_ in scored})} episode files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
