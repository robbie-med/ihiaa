"""Cluster pearls so that near-duplicates merge and contradictions surface.

Across 255 lectures and five years, several attendings teach the same topic.
That produces three things worth separating, and cosine similarity alone can
only find the first:

  * DUPLICATE  -- the same teaching point stated twice. Collapse it, but keep
                  every source, because N attendings independently saying the
                  same thing is a stronger signal than one saying it once.
  * CONSENSUS  -- different speakers, compatible claims. Reinforcing.
  * CONFLICT   -- claims that cannot both be acted on. "Always anticoagulate"
                  and "never anticoagulate" sit almost on top of each other in
                  embedding space, so similarity CANNOT distinguish these from
                  agreement. Only a semantic judgement can.

So: embed -> cluster by similarity -> ask a cheap model to judge only the pairs
that are already close. Judging every pair would be O(n^2) LLM calls; judging
within clusters keeps it linear-ish and affordable.

Conflicts are never auto-resolved. Both sides are kept, attributed and dated,
and shown together -- a resident seeing two attendings disagree is being given
real information, not a bug.

    python3 scripts/cluster_pearls.py [--threshold 0.82] [--max-pairs 400]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import ppq  # noqa: E402

EPISODES = BASE / "data" / "episodes"
OUT = BASE / "data" / "clusters.json"

JUDGE_SYS = """You compare pairs of clinical teaching points taken from residency lectures.

For each pair return one verdict:
- "duplicate"  : same teaching point, no added information
- "consensus"  : compatible and mutually reinforcing, but not identical
- "complement" : same topic, different aspect; both worth keeping separately
- "conflict"   : a clinician could not follow both; they give opposing guidance

Judge only what the text actually claims. Differing detail or emphasis is NOT a
conflict. Reserve "conflict" for genuinely opposing guidance, and say why.

Input: {"pairs":[{"id":"...","a":"...","b":"..."}]}
Return: {"verdicts":[{"id":"...","verdict":"...","why":"<12 words>"}]}"""


def load_pearls() -> list:
    """Only pearls with usable text.

    The embeddings endpoint rejects the ENTIRE batch if any element is an empty
    string, so one malformed pearl out of five thousand took down the whole
    clustering step with an opaque 400. Filtering here keeps a single bad record
    from being fatal.
    """
    out = []
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        for p in d.get("pearls", []):
            if not (p.get("text") or "").strip():
                continue
            out.append({**p, "episode_title": d.get("title", ""),
                        "pubDate": d.get("pubDate", "")})
    return out


def embed_pearls(pearls: list) -> np.ndarray:
    """Embed pearl text, cached on disk so re-clustering is free."""
    cache_path = BASE / "data" / "cache" / "pearl_vecs.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    todo = [p for p in pearls if p["pearl_id"] not in cache]
    for i in range(0, len(todo), 64):
        batch = todo[i:i + 64]
        vecs = ppq.embed([p["text"] for p in batch])
        for p, v in zip(batch, vecs):
            cache[p["pearl_id"]] = v
        print(f"  embedded {min(i+64, len(todo))}/{len(todo)}", flush=True)
    if todo:
        cache_path.write_text(json.dumps(cache))

    M = np.array([cache[p["pearl_id"]] for p in pearls], dtype=np.float32)
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M


def cluster(sim: np.ndarray, threshold: float) -> list:
    """Greedy connected-component clustering over the similarity graph."""
    n = sim.shape[0]
    seen, groups = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j)
            comp.append(j)
            for k in np.where(sim[j] >= threshold)[0]:
                if k not in seen:
                    stack.append(int(k))
        groups.append(sorted(comp))
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.74)
    ap.add_argument("--max-pairs", type=int, default=400)
    a = ap.parse_args()

    pearls = load_pearls()
    if len(pearls) < 2:
        print("not enough pearls yet")
        return 0
    print(f"{len(pearls)} pearls", flush=True)

    M = embed_pearls(pearls)
    sim = M @ M.T
    np.fill_diagonal(sim, 0.0)

    groups = cluster(sim, a.threshold)
    multi = [g for g in groups if len(g) > 1]
    print(f"{len(groups)} clusters, {len(multi)} with >1 pearl", flush=True)

    # Judge the closest pairs inside each multi-pearl cluster, strongest first,
    # bounded so cost stays predictable as the corpus grows.
    pairs = []
    for g in multi:
        for i, j in itertools.combinations(g, 2):
            pairs.append((float(sim[i, j]), i, j))
    pairs.sort(reverse=True)
    pairs = pairs[:a.max_pairs]
    print(f"judging {len(pairs)} closest pairs", flush=True)

    verdicts = {}
    for i in range(0, len(pairs), 12):
        batch = pairs[i:i + 12]
        payload = {"pairs": [{"id": f"{x}-{y}", "a": pearls[x]["text"],
                              "b": pearls[y]["text"]} for _, x, y in batch]}
        try:
            r = ppq.chat_json(JUDGE_SYS, json.dumps(payload, ensure_ascii=False),
                              max_tokens=3000)
        except Exception as e:                        # noqa: BLE001
            print(f"  ! {type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        for v in r.get("verdicts", []):
            verdicts[v.get("id", "")] = v
        print(f"  judged {min(i+12, len(pairs))}/{len(pairs)}", flush=True)

    out = []
    for g in multi:
        rel = []
        for s, i, j in [(s, i, j) for s, i, j in pairs if i in g and j in g]:
            v = verdicts.get(f"{i}-{j}")
            if not v:
                continue
            rel.append({"a": pearls[i]["pearl_id"], "b": pearls[j]["pearl_id"],
                        "sim": round(s, 3), "verdict": v.get("verdict", ""),
                        "why": v.get("why", "")})
        kinds = {r["verdict"] for r in rel}
        # Label the cluster from the vocabulary its pearls SHARE, rather than
        # showing the longest member verbatim. A shared-term label is short,
        # says what the cluster is about, and is computed the same way every run.
        import re as _re
        _stop = set("""the a an and or of to in for with without on at by from is are was
        were be been being this that these those it its as if then than when while
        because there here what which who how why not no do does did can could should
        would may might will have has had about into over under more most less least
        very just also so such other another each any all some patient patients use
        used using consider start check give avoid treat""".split())
        cnt = Counter()
        for i in g:
            for w in set(_re.findall(r"[a-z][a-z-]{3,}", pearls[i]["text"].lower())):
                if w not in _stop:
                    cnt[w] += 1
        shared = [w for w, c in cnt.most_common() if c >= max(2, len(g) // 2)][:3]
        label = " · ".join(shared) if shared else pearls[g[0]]["text"][:60]

        out.append({
            "pearl_ids": [pearls[i]["pearl_id"] for i in g],
            "speakers": sorted({pearls[i]["speaker"] for i in g}),
            "size": len(g),
            "label": label,
            "verdict": ("conflict" if "conflict" in kinds else
                        "duplicate" if "duplicate" in kinds else
                        "consensus" if "consensus" in kinds else "related"),
            "n_conflict": sum(1 for r in rel if r["verdict"] == "conflict"),
            "has_conflict": "conflict" in kinds,
            "has_duplicate": "duplicate" in kinds,
            "relations": rel,
        })
    out.sort(key=lambda c: (not c["has_conflict"], -c["size"]))

    OUT.write_text(json.dumps({"clusters": out, "threshold": a.threshold},
                              indent=2, ensure_ascii=False))
    nc = sum(1 for c in out if c["has_conflict"])
    print(f"\n-> {len(out)} clusters written, {nc} containing a conflict")
    print(f"   spend ${ppq.total_spend():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
