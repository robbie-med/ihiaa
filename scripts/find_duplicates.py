"""Find lectures that were uploaded to the feed more than once. Rules only.

Some recordings appear twice under different titles ("Dr. Okeefe on Coma..." and
"Dr. Yasmin Keefe on Coma..."; "Dr. Gerald C Miller on Syphilis..." with and
without the period). Each copy was transcribed and mined separately, so its
pearls showed up twice and were then "discovered" to be duplicates of each other,
filling the cluster view with a lecture agreeing with itself.

Two uploads are the same recording when their lengths match (within a minute or
3%) AND their transcripts share a meaningful fraction of 8-word phrases. ASR is
not perfectly repeatable, so identical audio gives ~20-80% overlap, while two
different lectures on the same topic share almost none.

The earliest-published copy is kept; the others get `duplicate_of` and are left
out of the site, the pearl index and clustering. Nothing is deleted.

    python3 scripts/find_duplicates.py
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from email.utils import parsedate_to_datetime

BASE = Path(__file__).resolve().parent.parent
EPISODES = BASE / "data" / "episodes"
MIN_OVERLAP = 0.15


def shingles(text: str) -> set:
    w = re.findall(r"[a-z]+", text.lower())
    return {" ".join(w[i:i + 8]) for i in range(0, max(0, len(w) - 8), 3)}


def when(ep: dict) -> float:
    try:
        return parsedate_to_datetime(ep.get("pubDate", "")).timestamp()
    except (TypeError, ValueError):
        return float("inf")


def main() -> int:
    eps, sh = {}, {}
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        eps[d["slug"]] = (f, d)
        s = shingles(d.get("transcript_raw") or "")
        if len(s) > 150:
            sh[d["slug"]] = s

    inv = defaultdict(set)
    for slug, s in sh.items():
        for x in s:
            inv[x].add(slug)
    shared = Counter()
    for x, slugs in inv.items():
        if 1 < len(slugs) < 6:
            slugs = sorted(slugs)
            for i in range(len(slugs)):
                for j in range(i + 1, len(slugs)):
                    shared[(slugs[i], slugs[j])] += 1

    dup_of = {}
    for (a, b), n in sorted(shared.items()):
        da, db = eps[a][1], eps[b][1]
        ov = n / min(len(sh[a]), len(sh[b]))
        la, lb = da.get("duration_sec", 0), db.get("duration_sec", 0)
        if ov < MIN_OVERLAP or abs(la - lb) > max(60, 0.03 * max(la, lb)):
            continue
        keep, drop = sorted((a, b), key=lambda s: (when(eps[s][1]), s))
        dup_of[drop] = keep
        print(f"  {ov:.2f}  {eps[drop][1]['title'][:44]:44s} -> {eps[keep][1]['title'][:44]}")
    # chains (c dup of b dup of a) collapse to the root
    for k in list(dup_of):
        while dup_of[k] in dup_of:
            dup_of[k] = dup_of[dup_of[k]]

    changed = 0
    for slug, (f, d) in eps.items():
        want = dup_of.get(slug)
        if d.get("duplicate_of") != want:
            if want:
                d["duplicate_of"] = want
            else:
                d.pop("duplicate_of", None)
            f.write_text(json.dumps(d, indent=2, ensure_ascii=False))
            changed += 1
    print(f"  {len(dup_of)} duplicate upload(s); {changed} file(s) updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
