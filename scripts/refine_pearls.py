"""Turn the model's raw pearl candidates into checked, attributable pearls. Free.

Extraction (pipeline.stage_enrich) is the only paid step and stores exactly what
the model returned in `pearls_raw`. Everything here is deterministic and re-runs
over the whole corpus in seconds, so it can be improved without paying again:

  anchor     find the quoted span in the word-timed transcript. A quote the model
             stitched together with "..." is anchored on its first real fragment.
             A pearl whose quote cannot be found is dropped, never guessed.
  numbers    every number in the pearl text must actually be spoken near that
             moment (digits, or words like "ninety", "two point five"). The model
             fills in doses and thresholds from its own knowledge; in a clinical
             reference a number the lecturer never said is the worst possible
             error. Unheard numbers are recorded in `numbers_unheard`, and the
             grader will not rank such a pearl above C.
  speaker    diarization says who was talking at the anchor. In a lecture where
             one presenter dominates, a pearl spoken by someone else (a resident's
             answer at morning report, a question from the floor) gets
             `by_presenter: false` instead of being credited to the attending.
  type       the model invented ~30 categories; they fold into a fixed set.
  dedupe     near-identical pearls within one lecture keep the first.
  ids        `<slug>-<second>`, suffixed on collision, stable across runs.

Results are cached per episode on a hash of the inputs, so a rebuild only
re-refines lectures whose raw pearls or rules changed.

    python3 scripts/refine_pearls.py [--force]
"""
import argparse
import bisect
import difflib
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EPISODES = BASE / "data" / "episodes"
REFINE_VERSION = "r3"
MIN_RATIO = 0.68
WINDOW = (-30, 180)            # seconds around the anchor searched for numbers

TYPES = {
    "dosing": "dosing", "drug": "dosing", "pharmacology": "dosing", "drug_interaction": "pitfall",
    "pitfall": "pitfall", "red_flag": "red_flag",
    "exam_technique": "exam_technique", "procedure": "procedure",
    "dx_criteria": "dx_criteria", "dx_criteri": "dx_criteria", "diagnostic_criteria": "dx_criteria",
    "diagnosis": "dx_criteria", "referral_criteria": "practice", "referral_guideline": "practice",
    "practice": "practice", "treatment": "practice", "lifestyle": "practice",
    "patient_education": "practice", "behavioral_intervention": "practice",
    "communication": "practice", "documentation": "practice", "billing": "practice",
    "history": "practice", "judgment": "judgment", "clinical_reasoning": "judgment",
    "concept": "concept", "pathophysiology": "concept", "risk_factor": "concept",
    "condition": "concept", "outcome": "concept", "evidence": "concept", "resource": "concept",
}

# ------------------------------------------------------------ number words
ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * (i + 2) for i, w in enumerate(
    "twenty thirty forty fifty sixty seventy eighty ninety".split())}
SCALES = {"hundred": 100, "thousand": 1000, "million": 1_000_000}
UNIT_WORD = (r"(?:milli|micro|centi|kilo)?(?:gram|liter|litre|meter|metre)s?$|"
             r"(?:mg|mcg|ml|cm|mm|kg|inch|inches|foot|feet|dose|doses|tablet|pill|unit|"
             r"day|days|week|weeks|month|months|year|years|hour|hours|minute|minutes|time|times)$")


def fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else f"{x:g}"


def spoken_numbers(tokens: list) -> set:
    """Numbers in a stretch of speech, from digits or words.

    'ninety percent' -> 90, 'two hundred' -> 200, 'two point five' -> 2.5,
    'five to seven' -> 5 and 7, 'forty five fifty five' -> 45 and 55,
    'one and a half' -> 1.5, '1,000' -> 1000.
    """
    out, cur, total, have, i = set(), 0, 0, False, 0
    toks = [t for tok in tokens for t in re.split(r"[-–]", tok.lower()) if t]

    seq = []                                   # (value, first token, last token)
    start = 0

    def flush():
        nonlocal cur, total, have
        if have:
            out.add(fmt(total + cur))
            seq.append((total + cur, start, i - 1))
        cur, total, have = 0, 0, False

    while i < len(toks):
        t = re.sub(r"[^a-z0-9.,/]", "", toks[i]).strip(".,")
        for frag in re.findall(r"\d[\d,]*(?:\.\d+)?", t):
            flush()
            out.add(fmt(float(frag.replace(",", ""))))
        if t in ONES:
            if have and cur % 10 and cur % 100 >= 10 or (have and cur % 100 and cur % 100 < 10):
                flush()
            if not have:
                start = i
            cur += ONES[t]
            have = True
        elif t in TENS:
            if have and cur % 100:
                flush()
            if not have:
                start = i
            cur += TENS[t]
            have = True
        elif t in SCALES and have:
            cur = max(cur, 1) * SCALES[t]
            if SCALES[t] > 100:
                total, cur = total + cur, 0
        elif t == "a" and i + 1 < len(toks) and toks[i + 1] in SCALES:
            cur, have, start = 1, True, i
        elif t == "and" and have and cur >= 100 and cur % 100 == 0 and i + 1 < len(toks) \
                and (toks[i + 1] in ONES or toks[i + 1] in TENS):
            pass                                # "a hundred and twenty"
        elif t == "point" and have and i + 1 < len(toks) and toks[i + 1] in ONES:
            frac = ""
            while i + 1 < len(toks) and toks[i + 1] in ONES and ONES[toks[i + 1]] < 10:
                i += 1
                frac += str(ONES[toks[i]])
            out.add(fmt(total + cur + float("0." + frac)))
            cur, total, have = 0, 0, False
        elif t == "point" and not have and i + 1 < len(toks) and toks[i + 1] in ONES:
            frac = ""
            while i + 1 < len(toks) and toks[i + 1] in ONES and ONES[toks[i + 1]] < 10:
                i += 1
                frac += str(ONES[toks[i]])
            out.add(fmt(float("0." + frac)))
        elif t == "and" and have and toks[i + 1:i + 3] == ["a", "half"]:
            out.add(fmt(total + cur + 0.5))
            i += 2
            flush()
        elif t == "half" and not have:
            out.add("0.5")
        elif t in ("a", "an", "per") and i + 1 < len(toks) and re.match(UNIT_WORD, toks[i + 1]):
            out.add("1")                        # "a centimeter", "an hour", "per day"
        else:
            flush()
        i += 1
    flush()
    # Clinical shorthand: "one twenty over eighty" is 120/80, "one forty five" 145.
    for (a, _, ea), (b, sb, _) in zip(seq, seq[1:]):
        if sb == ea + 1 and 1 <= a <= 9 and float(a).is_integer() and 10 <= b <= 99:
            out.add(fmt(a * 100 + b))
    return out


NUM_IN_TEXT = re.compile(r"(?<![A-Za-z0-9.])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)(?![A-Za-z0-9])")


def written_numbers(text: str) -> list:
    """Standalone numbers a pearl asserts. Skips names like COVID-19, B12, A1c, T2DM."""
    out = []
    for m in NUM_IN_TEXT.finditer(text or ""):
        a = m.start()
        if a >= 2 and text[a - 1] == "-" and text[a - 2].isalpha():
            continue
        out.append(fmt(float(m.group(1).replace(",", ""))))
    return list(dict.fromkeys(out))


# --------------------------------------------------------------- anchoring
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())


def anchor(verbatim: str, toks: list, times: list, flat: str, pos: list):
    """-> (time, how) or (None, None)."""
    frags = [re.sub(r"\s+", " ", _norm(f)).strip() for f in re.split(r"\.\.\.|…", verbatim or "")]
    frags = [f for f in frags if len(f.split()) >= 4] or \
        [re.sub(r"\s+", " ", _norm(verbatim)).strip()]
    v = frags[0]
    if len(v.split()) < 4:
        return None, None
    at = flat.find(v[:120])
    if at >= 0:
        i = bisect.bisect_right(pos, at) - 1
        return times[max(0, i)], "exact"
    probe, best, t = v.split()[:14], 0.0, None
    step = max(1, len(probe) // 3)
    sm = difflib.SequenceMatcher(None, probe, [], autojunk=False)
    for i in range(0, max(1, len(toks) - len(probe)), step):
        sm.set_seq2(toks[i:i + len(probe)])
        if sm.real_quick_ratio() < MIN_RATIO or sm.quick_ratio() < MIN_RATIO:
            continue
        r = sm.ratio()
        if r > best:
            best, t = r, times[i]
    return (t, "fuzzy") if best >= MIN_RATIO else (None, None)


def anchor_by_segment(raw: dict, segments: list):
    """Prompt v2 cites a segment number. Trust it only if the quote really is
    there: the cited segment and the next three must contain most of the quote."""
    seg = raw.get("seg")
    if not isinstance(seg, int) or not 0 <= seg < len(segments):
        return None, None
    q = _norm(raw.get("verbatim", "")).split()
    if len(q) < 4:
        return None, None
    near = _norm(" ".join(s["text"] for s in segments[seg:seg + 4])).split()
    hit = sum(1 for w in set(q) if w in set(near)) / len(set(q))
    return (segments[seg]["t"], "segment") if hit >= 0.8 else (None, None)


def _dedash(s: str) -> str:
    s = re.sub(r"\s*[—–]\s*", ", ", s or "")
    s = re.sub(r",\s*,", ",", s)
    return re.sub(r",\s*\.", ".", s)


def content(s: str) -> set:
    return set(re.findall(r"[a-z]{4,}", s.lower()))


# ------------------------------------------------------------------ refine
def refine(ep: dict) -> list:
    words = ep.get("words") or []
    toks = [_norm(w.get("w")).strip() for w in words]
    times = [w["s"] for w in words]
    flat, pos, o = [], [], 0
    for t in toks:
        pos.append(o)
        flat.append(t)
        o += len(t) + 1
    flat = " ".join(flat)

    spk = [w.get("spk") for w in words]
    c = Counter(spk)
    main, n_main = c.most_common(1)[0] if c else (None, 0)
    # Attribution only means something when one voice clearly carries the talk
    # and the lecture has a single named presenter.
    attributable = bool(words) and n_main / len(words) >= 0.6 and len(ep.get("speakers") or []) <= 1

    out, seen_text = [], []
    for raw in ep.get("pearls_raw", []):
        text = _dedash((raw.get("text") or "").strip())
        if len(text.split()) < 4:
            continue
        t, how = anchor(raw.get("verbatim", ""), toks, times, flat, pos)
        if t is None:
            t, how = anchor_by_segment(raw, ep.get("segments") or [])
        if t is None:
            continue
        ct = content(text)
        if any(ct and len(ct & s) / len(ct | s) > 0.6 for s in seen_text):
            continue
        seen_text.append(ct)

        lo = bisect.bisect_left(times, t + WINDOW[0])
        hi = bisect.bisect_right(times, t + WINDOW[1])
        heard = spoken_numbers([w.get("w", "") for w in words[lo:hi]])
        unheard = [x for x in written_numbers(text) if x not in heard]

        p = {"text": text, "verbatim": (raw.get("verbatim") or "").strip(),
             "type": TYPES.get((raw.get("type") or "").strip().lower(), "practice"),
             "t": round(t, 1), "match": how, "episode": ep["slug"],
             "speaker": ep.get("speaker", ""),
             "model": ep.get("enrich_model", ""), "prompt_version": ep.get("prompt_version", "")}
        if raw.get("applies_to"):
            p["applies_to"] = raw["applies_to"].strip()
        if unheard:
            p["numbers_unheard"] = unheard
        if attributable:
            i = bisect.bisect_left(times, t)
            win = Counter(spk[i:i + 40])
            if win and win.most_common(1)[0][0] != main:
                p["by_presenter"] = False
        out.append(p)

    out.sort(key=lambda p: p["t"])
    used = Counter()
    for p in out:
        base = f"{ep['slug']}-{int(p['t'])}"
        used[base] += 1
        p["pearl_id"] = base if used[base] == 1 else f"{base}-{used[base]}"
    return out


def fingerprint(ep: dict) -> str:
    h = hashlib.sha1(REFINE_VERSION.encode())
    h.update(json.dumps(ep.get("pearls_raw", []), sort_keys=True).encode())
    h.update(json.dumps(len(ep.get("words") or [])).encode())
    h.update(json.dumps([s["id"] for s in ep.get("speakers") or []]).encode())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("slugs", nargs="*")
    a = ap.parse_args()

    n_ep = n_raw = n_kept = n_unheard = n_other = 0
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        if a.slugs and d["slug"] not in a.slugs:
            continue
        if not d.get("pearls_raw"):
            continue
        fp = fingerprint(d)
        if d.get("pearls_fingerprint") == fp and not a.force:
            ps = d.get("pearls", [])
        else:
            old = {p["pearl_id"]: p.get("score") for p in d.get("pearls", [])}
            ps = refine(d)
            for p in ps:                      # keep grades until score_pearls reruns
                if old.get(p["pearl_id"]):
                    p["score"] = old[p["pearl_id"]]
            d["pearls"], d["pearls_fingerprint"] = ps, fp
            f.write_text(json.dumps(d, indent=2, ensure_ascii=False))
            n_ep += 1
        n_raw += len(d["pearls_raw"])
        n_kept += len(ps)
        n_unheard += sum(1 for p in ps if p.get("numbers_unheard"))
        n_other += sum(1 for p in ps if p.get("by_presenter") is False)
    print(f"  refined {n_ep} lecture(s); {n_kept} of {n_raw} raw pearls kept; "
          f"{n_unheard} with a number not heard near the anchor; "
          f"{n_other} said by someone other than the presenter")
    return 0


def selftest() -> int:
    cases = [("ninety percent of adults", {"90"}), ("two hundred parts per million", {"200"}),
             ("two point five milligrams", {"2.5"}), ("five to seven days", {"5", "7"}),
             ("forty five fifty five dollars", {"45", "55"}), ("one and a half hours", {"1.5"}),
             ("1,000 units", {"1000"}), ("a hundred and twenty", {"120"}),
             ("twenty one days", {"21"}), ("one twenty over eighty", {"120", "80"}),
             ("one forty five", {"145"}),
             ("within a centimeter of the septum", {"1"}),
             ("less than 6 mg/dL", {"6"})]
    bad = 0
    for s, want in cases:
        got = spoken_numbers(s.split())
        ok = want <= got
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {s!r:36} -> {sorted(got)}")
    for s, want in [("COVID-19 and B12 with A1c below 8", ["8"]), ("5-7 days, 90%", ["5", "7", "90"]),
                    ("bolus 500-1,000 mL", ["500", "1000"])]:
        got = written_numbers(s)
        bad += got != want
        print(f"  {'PASS' if got == want else 'FAIL'}  {s!r:36} -> {got}")
    print("selftest:", "FAILED" if bad else "ok")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    raise SystemExit(main())
