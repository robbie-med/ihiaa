"""Poll the Podbean feed and materialise metadata for any new episodes.

Keyed on <guid>, which is stable across re-uploads -- unlike the enclosure URL,
which changes, and unlike pubDate, which is unreliable (the archive was bulk
re-uploaded, so publication order is not teaching order).

Per the project decision, `pubDate` and the episode `<link>` are the canonical
date and hyperlink. No recording-date inference is attempted.

Prints the slugs of episodes that still need processing, one per line, so CI can
decide whether there is any work to do.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
FEED = "https://feed.podbean.com/hopepartners/feed.xml"
META = BASE / "data" / "episodes_meta.json"
EPISODES = BASE / "data" / "episodes"


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return re.sub(r"-\d{1,2}-\d{1,2}-\d{2,4}$", "", s)[:60] or "episode"


def parse(xml: str) -> dict:
    out = {}
    for it in re.findall(r"<item>.*?</item>", xml, re.S):
        g = re.search(r"<guid[^>]*>(.*?)</guid>", it, re.S)
        t = re.search(r"<title>(.*?)</title>", it, re.S)
        e = re.search(r'<enclosure url="([^"]+)"[^>]*length="(\d+)"', it)
        if not (g and t and e):
            continue
        title = t.group(1).strip().replace("&amp;", "&")
        spk = re.match(r"^\s*(?:Dr\.?\s*|Drs\.?\s*)?(.+?)\s+on\s+", title)
        d = re.search(r"<itunes:duration>(\d+)</itunes:duration>", it)
        lk = re.search(r"<link>(.*?)</link>", it, re.S)
        pd = re.search(r"<pubDate>(.*?)</pubDate>", it)
        slug = slugify(title)
        base, n = slug, 2
        while slug in out:                       # keep slugs unique
            slug, n = f"{base}-{n}", n + 1
        out[slug] = {
            "slug": slug, "guid": g.group(1).strip(), "title": title,
            "speaker": re.sub(r",?\s*(MD|DO|PhD|PharmD|RN|NP|PA)\.?$", "",
                              spk.group(1)).strip() if spk else "Unknown",
            "link": lk.group(1).strip() if lk else "",
            "pubDate": pd.group(1).strip() if pd else "",
            "duration_sec": int(d.group(1)) if d else 0,
            "enclosure": e.group(1), "enclosure_bytes": int(e.group(2)),
        }
    return out


def main() -> int:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    with urllib.request.urlopen(FEED, timeout=120) as r:
        feed = parse(r.read().decode("utf-8", "replace"))

    old = json.loads(META.read_text()) if META.exists() else {}
    by_guid = {v["guid"] for v in old.values()}
    merged = dict(old)
    new = []
    for slug, v in feed.items():
        if v["guid"] in by_guid:
            continue
        merged[slug] = v
        new.append(slug)

    META.parent.mkdir(parents=True, exist_ok=True)
    META.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    done = {p.stem for p in EPISODES.glob("*.json")
            if json.loads(p.read_text()).get("chunks")}
    todo = [s for s in merged if s not in done]
    if limit:
        todo = todo[:limit]

    print(f"feed={len(feed)} known={len(old)} new={len(new)} todo={len(todo)}",
          file=sys.stderr)
    for s in todo:
        print(s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
