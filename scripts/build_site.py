"""Generate the static site from data/episodes/*.json.

Output is a self-contained folder that works on GitHub Pages, Cloudflare Pages,
or a local `python3 -m http.server` with no build step and no backend:

    site/index.html   app shell (all CSS/JS inline)
    site/data.json    episodes, pearls, topics, chunk vectors (int8)

Search is hybrid and runs entirely in the browser:
  * keyword   -- substring/token match over transcripts and pearls, instant
  * semantic  -- cosine over int8-quantised chunk vectors

Semantic search needs the QUERY embedded, which is the one thing a static page
cannot do. Set a worker URL in the UI (or leave blank) -- keyword search works
unconditionally, so the page is never dead without it.
"""
import json
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
EPISODES = BASE / "data" / "episodes"
SITE = BASE / "site"

LANGS = {"en": "English", "ko": "한국어", "fr": "Français", "de": "Deutsch"}


def norm_topic(label: str) -> str:
    """Collapse surface variants so the topic index does not fragment.

    This is the poor-man's stand-in for MeSH binding (see PLAN §1). It handles
    case/plural/punctuation drift only. Real MeSH descriptor IDs replace this
    the moment the ontology is wired in -- the topic *slug* is deliberately the
    only thing the rest of the site keys on, so that swap is local.
    """
    s = re.sub(r"[^a-z0-9 ]", " ", (label or "").lower())
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\b(disease|disorders?|syndromes?)\b", "", s).strip() or s
    if s.endswith("ies"):
        s = s[:-3] + "y"
    elif s.endswith("es") and not s.endswith("ses"):
        s = s[:-2]
    elif s.endswith("s") and not s.endswith("ss"):
        s = s[:-1]
    return s


def quantize(vec: list) -> list:
    """int8 quantisation. 1536d float32 -> 512d int8 is the difference between
    a 17MB browser payload and a 5.6MB one at 500 episodes (PLAN §5.7)."""
    m = max((abs(x) for x in vec), default=1.0) or 1.0
    return [max(-127, min(127, int(round(x / m * 127)))) for x in vec]


def build() -> dict:
    eps, pearls, topics = [], [], defaultdict(list)
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        if not d.get("transcript"):
            print(f"  skip {f.name}: not transcribed yet")
            continue

        chunks = [{"t": c["t"], "text": c["text"], "v": quantize(c["vec"])}
                  for c in d.get("chunks", []) if c.get("vec")]

        # Grades come from scripts/score_pearls.py (deterministic rules, no model).
        # D is excluded from the site entirely -- that band is institution-local
        # logistics and non-clinical filler. C is kept but sorts below A/B.
        for p in d.get("pearls", []):
            g = (p.get("score") or {}).get("grade", "C")
            if g == "D":
                continue
            p = dict(p)
            p["grade"] = g
            p["points"] = (p.get("score") or {}).get("points", 0)
            p.pop("score", None)
            p["episode_title"] = d["title"]
            p["link"] = d["link"]
            pearls.append(p)

        seen = set()
        for t in d.get("topics_raw", []):
            label = (t.get("label") or "").strip()
            if not label:
                continue
            slug = norm_topic(label)
            if not slug or (slug, d["slug"]) in seen:
                continue
            seen.add((slug, d["slug"]))
            topics[slug].append({"label": label, "kind": t.get("kind", "concept"),
                                 "episode": d["slug"]})

        # Transcript + segments + vectors are the bulk of the payload. They go
        # into per-episode files fetched on demand; a single combined data.json
        # extrapolated to ~82MB at 257 episodes, which no one should download
        # to read one pearl.
        (SITE / "ep").mkdir(parents=True, exist_ok=True)
        (SITE / "ep" / f"{d['slug']}.json").write_text(json.dumps({
            "slug": d["slug"], "transcript": d["transcript"],
            "segments": d.get("segments", []), "chunks": chunks,
            "fixes": d.get("fixes", []),
            "fixes_rejected": d.get("fixes_rejected", []),
        }, ensure_ascii=False, separators=(",", ":")))

        eps.append({
            "slug": d["slug"], "title": d["title"], "speaker": d["speaker"],
            "link": d["link"], "pubDate": d.get("pubDate", ""),
            "duration_sec": d.get("duration_sec", 0),
            "abstract": d.get("abstract", ""), "key_points": d.get("key_points", []),
            "n_fixes": len(d.get("fixes", [])),
            "n_speakers": d.get("n_speakers", 0),
            "n_chunks": len(chunks), "n_words": len(d["transcript"].split()),
            "stt_model": d.get("stt_model", ""), "enrich_model": d.get("enrich_model", ""),
            "embed_model": d.get("embed_model", ""), "embed_dims": d.get("embed_dims", 0),
            "i18n": d.get("i18n", {}),
        })

    topic_list = []
    for slug, hits in sorted(topics.items(), key=lambda kv: -len(kv[1])):
        label = max((h["label"] for h in hits), key=len)
        topic_list.append({"slug": slug, "label": label,
                           "kind": hits[0]["kind"],
                           "episodes": sorted({h["episode"] for h in hits})})

    speakers = defaultdict(list)
    for e in eps:
        speakers[e["speaker"]].append(e["slug"])

    # Pearl clusters (near-duplicates, consensus, contradictions) are produced
    # separately by cluster_pearls.py; absent on a fresh corpus.
    cl = BASE / "data" / "clusters.json"
    clusters = json.loads(cl.read_text())["clusters"] if cl.exists() else []

    # Best pearls first, everywhere they are listed.
    pearls.sort(key=lambda p: (-p.get("points", 0), p["episode"], p["t"]))

    return {"episodes": eps, "pearls": pearls, "topics": topic_list,
            "clusters": clusters,
            "speakers": [{"name": k, "episodes": v} for k, v in sorted(speakers.items())],
            "langs": LANGS,
            "stats": {"n_episodes": len(eps), "n_pearls": len(pearls),
                      "n_topics": len(topic_list),
                      "hours": round(sum(e["duration_sec"] for e in eps) / 3600, 1),
                      "words": sum(e["n_words"] for e in eps)}}


if __name__ == "__main__":
    SITE.mkdir(exist_ok=True)
    data = build()
    (SITE / "data.json").write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    idx = (SITE / "data.json").stat().st_size
    ep_total = sum(f.stat().st_size for f in (SITE / "ep").glob("*.json"))
    s = data["stats"]
    n = max(s["n_episodes"], 1)
    print(f"  {s['n_episodes']} episodes | {s['n_pearls']} pearls | "
          f"{s['n_topics']} topics | {s['hours']}h | {s['words']:,} words")
    print(f"  data.json (index, always loaded) = {idx/1e6:.2f} MB"
          f"   -> {idx/1e6*257/n:.0f} MB at 257 episodes")
    print(f"  ep/*.json (lazy, per episode)    = {ep_total/1e6:.2f} MB"
          f"   -> {ep_total/1e6/n:.2f} MB each, fetched on demand")
