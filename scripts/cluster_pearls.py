"""Find where lectures agree and where attendings genuinely contradict each other.

The first version embedded every pearl, chained everything above a similarity
threshold into connected components, then judged the 400 most similar pairs.
Measured on the corpus, that went wrong in three ways:

  * a third of the judging budget went on pairs from the SAME lecture, and over
    half on the same speaker -- a talk agreeing with itself
  * chaining glued unrelated claims into blobs of up to 14 pearls, and one
    conflicting pair then labelled the whole blob a "conflict"
  * the most similar pairs are near-duplicates, while real contradictions sit at
    moderate similarity ("always X" vs "avoid X in Y"), so they were rarely judged

This version:

  1. CANDIDATES  each pearl's nearest neighbours from OTHER lectures (duplicate
                 uploads excluded). Ranked by similarity, nudged up when the pair
                 has opposite directives ("avoid" vs "give") or different numbers
                 attached to the same unit -- the surface signs of disagreement.
  2. JUDGE       a cheap model labels each pair in batches: same | agree |
                 complement | contradict | unrelated, plus a short topic.
  3. VERIFY      every "contradict" goes to a stronger model, one pair at a time,
                 with both verbatim quotes, who said it, when, and who it applies
                 to, and is asked to find a reading under which BOTH are true
                 (different population, setting, severity, newer evidence). Only
                 pairs that survive are "conflict"; the rest become "depends",
                 shown with the condition that reconciles them.
  4. GROUP       agreement groups are built ONLY from pairs judged same/agree, so
                 nothing is grouped on similarity alone. A group taught by two or
                 more different speakers is consensus; one speaker repeating
                 themselves across years is "repeated".

Every verdict is cached by the text of both pearls, so a nightly run only pays
for pairs it has never seen. A first full run costs a few cents.

    python3 scripts/cluster_pearls.py [--max-new 1500] [--k 8] [--min-sim 0.62]
"""
import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import ppq  # noqa: E402

EPISODES = BASE / "data" / "episodes"
OUT = BASE / "data" / "clusters.json"
CACHE = BASE / "data" / "cache"          # embeddings: large, cheap to redo, not committed
# Judge and verify verdicts are the paid part. They live in git, so a run that is
# cut off still keeps everything it judged, and every verdict can be audited.
VERDICTS = BASE / "data" / "relations"
DEADLINE = [float("inf")]
JUDGE_VERSION, VERIFY_VERSION = "j2", "v2"

JUDGE_SYS = """You compare pairs of clinical teaching points from different residency lectures.

For each pair return one relation:
- "same"       : the same teaching point; nothing added
- "agree"      : compatible and mutually reinforcing, not identical
- "complement" : same topic, different aspect; neither bears on the other
- "contradict" : a clinician could not follow both in the same patient
- "unrelated"  : different topics after all

Judge only what the text claims. Different detail or emphasis is NOT a
contradiction. If "applies_to" differs (e.g. pregnancy vs not), guidance that
differs between those groups is "complement", not "contradict".

Input: {"pairs":[{"id":"...","a":{...},"b":{...}}]}
Return: {"verdicts":[{"id":"...","relation":"...","topic":"<=6 words, the shared clinical question","why":"<=15 words"}]}"""

VERIFY_SYS = """Two attendings taught the following in residency lectures. A first pass
flagged them as contradicting each other. Your job is to try hard to REFUTE that.

Look for any reading under which a clinician could follow both:
- different patient population, age, pregnancy, comorbidity or severity
- different setting (ED vs clinic vs ICU), timing, or stage of illness
- different drug, formulation, route or dose range
- one is a general rule and the other a stated exception
- they are about different questions that only sound alike

Answer:
- "compatible": no real disagreement once read carefully
- "depends":    both can be right; which applies depends on a condition (name it)
- "conflict":   they give opposing guidance for the same patient in the same
                situation, and a resident would have to choose one

Also say, in one short sentence, what the actual point of disagreement is or what
condition decides it. Do not decide who is right.

Return JSON: {"verdict":"compatible|depends|conflict","topic":"<=6 words","why":"<=25 words","condition":"<=12 words, only for depends"}"""

NEG = re.compile(r"\b(?:never|avoid|don'?t|do not|does not|not recommended|contraindicated|"
                 r"no need|unnecessary|shouldn'?t|should not|stop|discontinue|no longer|"
                 r"not indicated|ineffective|harmful|instead of)\b", re.I)
NUM_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|kg|ml|units?|%|mmhg|hours?|days?|weeks?|"
                      r"months?|years?|minutes?)\b", re.I)


def h(text: str) -> str:
    return hashlib.sha1(text.strip().encode()).hexdigest()[:16]


def load_pearls() -> list:
    """Pearls worth relating: graded A-C, from lectures that are not re-uploads.

    Pearls with empty text are skipped: the embeddings endpoint rejects a whole
    batch if any element is empty.
    """
    out = []
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        if d.get("duplicate_of"):
            continue
        for p in d.get("pearls", []):
            if not (p.get("text") or "").strip():
                continue
            if (p.get("score") or {}).get("grade") == "D":
                continue
            out.append({**p, "episode_title": d.get("title", ""), "pubDate": d.get("pubDate", ""),
                        "speaker_ids": [s["id"] for s in d.get("speakers") or []],
                        "h": h(p["text"])})
    return out


def embed_pearls(pearls: list) -> np.ndarray:
    """Embed pearl text, cached by text hash so re-ids and re-runs are free."""
    path = CACHE / "pearl_vecs_by_text.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = list({p["h"]: p for p in pearls if p["h"] not in cache}.values())
    for i in range(0, len(todo), 64):
        batch = todo[i:i + 64]
        for p, v in zip(batch, ppq.embed([p["text"] for p in batch])):
            cache[p["h"]] = v
        print(f"  embedded {min(i + 64, len(todo))}/{len(todo)}", flush=True)
    if todo:
        path.write_text(json.dumps(cache))
    M = np.array([cache[p["h"]] for p in pearls], dtype=np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M


def disagreement_cues(a: str, b: str) -> float:
    s = 0.0
    if bool(NEG.search(a)) != bool(NEG.search(b)):
        s += 0.12
    na = {(u.lower().rstrip("s"), v) for v, u in NUM_UNIT.findall(a)}
    nb = {(u.lower().rstrip("s"), v) for v, u in NUM_UNIT.findall(b)}
    ua, ub = {u for u, _ in na}, {u for u, _ in nb}
    if ua & ub and not (na & nb):
        s += 0.12
    return s


def candidates(pearls: list, M: np.ndarray, k: int, min_sim: float) -> list:
    """(priority, sim, i, j) for cross-lecture neighbour pairs, best first."""
    sim = M @ M.T
    ep = np.array([p["episode"] for p in pearls])
    seen, out = set(), []
    for i in range(len(pearls)):
        row = sim[i].copy()
        row[ep == ep[i]] = -1                    # never pair a lecture with itself
        for j in np.argsort(-row)[:k]:
            j = int(j)
            s = float(row[j])
            if s < min_sim:
                break
            key = (min(i, j), max(i, j))
            if key in seen or pearls[i]["h"] == pearls[j]["h"]:
                continue
            seen.add(key)
            a, b = pearls[key[0]], pearls[key[1]]
            pri = s + disagreement_cues(a["text"], b["text"])
            if set(a["speaker_ids"]) & set(b["speaker_ids"]):
                pri -= 0.05                      # same person: less informative
            out.append((round(pri, 6), round(s, 4), key[0], key[1]))
    out.sort(key=lambda x: (-x[0], x[2], x[3]))
    return out


def pkey(a: dict, b: dict) -> str:
    return "|".join(sorted((a["h"], b["h"])))


def side(p: dict) -> dict:
    d = {"text": p["text"]}
    if p.get("applies_to"):
        d["applies_to"] = p["applies_to"]
    return d


def load_cache(name: str) -> dict:
    path = VERDICTS / name
    return json.loads(path.read_text()) if path.exists() else {}


def save_cache(name: str, data: dict) -> None:
    VERDICTS.mkdir(parents=True, exist_ok=True)
    (VERDICTS / name).write_text(json.dumps(data, sort_keys=True, indent=0))


def judge(pearls: list, cands: list, max_new: int) -> dict:
    cache = load_cache(f"judge_{JUDGE_VERSION}.json")
    todo = [c for c in cands if pkey(pearls[c[2]], pearls[c[3]]) not in cache][:max_new]
    print(f"  judge: {len(cands) - len(todo)} pairs cached or deferred, {len(todo)} to judge", flush=True)
    for n in range(0, len(todo), 10):
        if time.time() > DEADLINE[0]:
            print(f"  judge: time budget reached after {n} pairs; the rest wait for the next run",
                  flush=True)
            save_cache(f"judge_{JUDGE_VERSION}.json", cache)
            break
        batch = todo[n:n + 10]
        payload = {"pairs": [{"id": str(x), "a": side(pearls[i]), "b": side(pearls[j])}
                             for x, (_, _, i, j) in enumerate(batch)]}
        try:
            r = ppq.chat_json(JUDGE_SYS, json.dumps(payload, ensure_ascii=False), max_tokens=2500)
        except Exception as e:                        # noqa: BLE001
            print(f"  ! judge: {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        got = {str(v.get("id")): v for v in r.get("verdicts", [])}
        for x, (_, _, i, j) in enumerate(batch):
            v = got.get(str(x))
            rel = (v or {}).get("relation", "")
            if rel in ("same", "agree", "complement", "contradict", "unrelated"):
                cache[pkey(pearls[i], pearls[j])] = {"relation": rel,
                                                     "topic": (v.get("topic") or "")[:60],
                                                     "why": (v.get("why") or "")[:160]}
        if (n // 10) % 10 == 9 or n + 10 >= len(todo):
            save_cache(f"judge_{JUDGE_VERSION}.json", cache)
            print(f"  judged {min(n + 10, len(todo))}/{len(todo)}", flush=True)
    return cache


def verify(pearls: list, pairs: list) -> dict:
    cache = load_cache(f"verify_{VERIFY_VERSION}.json")
    todo = [(i, j) for i, j in pairs if pkey(pearls[i], pearls[j]) not in cache]
    print(f"  verify: {len(pairs) - len(todo)} cached, {len(todo)} to check with {ppq.VERIFY_MODEL}",
          flush=True)
    for n, (i, j) in enumerate(todo):
        if time.time() > DEADLINE[0] + 600:    # verification gets 10 extra minutes
            print(f"  verify: time budget reached after {n}; the rest wait for the next run", flush=True)
            break
        a, b = pearls[i], pearls[j]
        user = json.dumps({x: {"teaching": p["text"], "quote": p.get("verbatim", ""),
                               "applies_to": p.get("applies_to", ""),
                               "lecture": p["episode_title"], "speaker": p.get("speaker", ""),
                               "date": p.get("pubDate", "")[:16]}
                           for x, p in (("A", a), ("B", b))}, ensure_ascii=False)
        r, used = None, None
        # If the stronger model is unavailable, fall back rather than silently
        # verifying nothing (which would show zero conflicts as if checked).
        for model in dict.fromkeys((ppq.VERIFY_MODEL, ppq.CHAT_MODEL)):
            try:
                r = ppq.chat_json(VERIFY_SYS, user, model=model, max_tokens=600, temperature=0.0)
                used = model
                break
            except Exception as e:                    # noqa: BLE001
                print(f"  ! verify ({model}): {type(e).__name__}: {str(e)[:80]}", flush=True)
        if r is None:
            continue
        v = r.get("verdict", "")
        if v in ("compatible", "depends", "conflict"):
            cache[pkey(a, b)] = {"verdict": v, "topic": (r.get("topic") or "")[:60],
                                 "why": (r.get("why") or "")[:200],
                                 "condition": (r.get("condition") or "")[:100], "model": used}
        if n % 10 == 9:
            save_cache(f"verify_{VERIFY_VERSION}.json", cache)
    if todo:
        save_cache(f"verify_{VERIFY_VERSION}.json", cache)
    return cache


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="neighbours per pearl")
    ap.add_argument("--min-sim", type=float, default=0.62)
    ap.add_argument("--max-new", type=int, default=1500, help="new pairs to judge this run")
    ap.add_argument("--group-sim", type=float, default=0.72,
                    help="every pair inside an agreement group must be at least this similar")
    ap.add_argument("--max-group", type=int, default=12)
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop judging after this long; output is written either way")
    a = ap.parse_args()
    if a.max_minutes:
        DEADLINE[0] = time.time() + a.max_minutes * 60

    pearls = load_pearls()
    if len(pearls) < 2:
        print("not enough pearls yet")
        return 0
    print(f"{len(pearls)} pearls from {len({p['episode'] for p in pearls})} lectures", flush=True)
    M = embed_pearls(pearls)
    cands = candidates(pearls, M, a.k, a.min_sim)
    print(f"  {len(cands)} cross-lecture candidate pairs (k={a.k}, sim>={a.min_sim})", flush=True)

    jc = judge(pearls, cands, a.max_new)
    rel = {}
    for pri, s, i, j in cands:
        v = jc.get(pkey(pearls[i], pearls[j]))
        if v:
            rel[(i, j)] = (s, v)
    contra = [(i, j) for (i, j), (_, v) in rel.items() if v["relation"] == "contradict"]
    vc = verify(pearls, contra)

    def same_speaker(i, j):
        return bool(set(pearls[i]["speaker_ids"]) & set(pearls[j]["speaker_ids"]))

    def pair(i, j, s, extra):
        return {"a": pearls[i]["pearl_id"], "b": pearls[j]["pearl_id"], "sim": s,
                "same_speaker": same_speaker(i, j), **extra}

    conflicts, depends = [], []
    for i, j in contra:
        v = vc.get(pkey(pearls[i], pearls[j]))
        if not v:
            continue
        s = rel[(i, j)][0]
        if v["verdict"] == "conflict":
            conflicts.append(pair(i, j, s, {"topic": v["topic"], "why": v["why"]}))
        elif v["verdict"] == "depends":
            depends.append(pair(i, j, s, {"topic": v["topic"], "why": v["why"],
                                          "condition": v["condition"]}))

    # Agreement groups. Only pairs JUDGED same/agree can join, and two groups merge
    # only if every pearl in one is close to every pearl in the other (complete
    # linkage). Plain union-find chains A~B~C~... into one blob of loosely related
    # claims, which is exactly what made the old "clusters" unreadable.
    sim = M @ M.T
    comp = {}                                    # pearl index -> frozenset of members

    def group_of(x):
        return comp.get(x, frozenset([x]))

    edges = sorted(((s, i, j) for (i, j), (s, v) in rel.items()
                    if v["relation"] in ("same", "agree")), key=lambda e: (-e[0], e[1], e[2]))
    for s, i, j in edges:
        A, B = group_of(i), group_of(j)
        if A == B or len(A) + len(B) > a.max_group:
            continue
        if min(float(sim[x, y]) for x in A for y in B) < a.group_sim:
            continue
        U = A | B
        for x in U:
            comp[x] = U
    members = {g: sorted(g) for g in set(comp.values())}
    topics = defaultdict(Counter)
    for (i, j), (s, v) in rel.items():
        if v["relation"] in ("same", "agree") and v["topic"] and i in comp and comp[i] == comp.get(j):
            topics[comp[i]][v["topic"].strip().lower()] += 1
    groups = []
    for root, idx in members.items():
        idx.sort(key=lambda x: (pearls[x]["episode"], pearls[x]["t"]))
        spk = {sid for x in idx for sid in pearls[x]["speaker_ids"]}
        names = sorted({pearls[x].get("speaker", "") for x in idx})
        t = sorted(topics[root].items(), key=lambda kv: (-kv[1], kv[0]))
        groups.append({"pearl_ids": [pearls[x]["pearl_id"] for x in idx],
                       "topic": t[0][0] if t else pearls[idx[0]]["text"][:50],
                       "speakers": names, "n_speakers": len(spk),
                       "n_lectures": len({pearls[x]["episode"] for x in idx}),
                       "kind": "consensus" if len(spk) >= 2 else "repeated"})
    groups.sort(key=lambda g: (-g["n_speakers"], -len(g["pearl_ids"]), g["topic"], g["pearl_ids"][0]))
    conflicts.sort(key=lambda c: (c["same_speaker"], -c["sim"], c["a"]))
    depends.sort(key=lambda c: (-c["sim"], c["a"]))

    counts = Counter(v["relation"] for _, v in rel.values())
    refuted = sum(1 for i, j in contra
                  if (vc.get(pkey(pearls[i], pearls[j])) or {}).get("verdict") == "compatible")
    OUT.write_text(json.dumps({
        "version": 2, "judge": JUDGE_VERSION, "verify": VERIFY_VERSION,
        "stats": {"pearls": len(pearls), "candidates": len(cands), "judged": len(rel),
                  "relations": dict(sorted(counts.items())),
                  "verified_conflicts": len(conflicts), "depends": len(depends),
                  "refuted": refuted},
        "conflicts": conflicts, "depends": depends, "groups": groups,
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {len(conflicts)} verified conflicts, {len(depends)} context-dependent, "
          f"{refuted} flagged then refuted; "
          f"{len(groups)} agreement groups "
          f"({sum(1 for g in groups if g['kind'] == 'consensus')} across speakers)")
    print(f"   judged {len(rel)} of {len(cands)} candidate pairs; relations {dict(counts)}")
    print(f"   spend ${ppq.total_spend():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
