"""One-shot ingest for a single episode: download -> audio -> pipeline.

Built for a CI runner with a small disk. The source mp4s are 350MB-1.2GB each,
so each episode is downloaded, stripped to a ~24kbps mono audio track, and the
video deleted before the next one starts. Peak disk stays around one video.

Audio is kept (a few MB per lecture): re-transcribing later against a better
model should never require re-downloading from Podbean, whose CDN links rot.
"""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import pipeline  # noqa: E402

VIDEO = BASE / "data" / "video"
AUDIO = BASE / "data" / "audio"


def have_audio(slug: str) -> bool:
    p = AUDIO / f"{slug}.m4a"
    return p.exists() and p.stat().st_size > 10_000


def make_audio(slug: str, url: str) -> None:
    """Download the enclosure, extract mono 24kbps audio, drop the video."""
    VIDEO.mkdir(parents=True, exist_ok=True)
    AUDIO.mkdir(parents=True, exist_ok=True)
    src = VIDEO / f"{slug}.src"
    print(f"  downloading {url[:78]} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ihi-aa/1.0"})
    with urllib.request.urlopen(req, timeout=1800) as r, src.open("wb") as fh:
        while chunk := r.read(1 << 20):
            fh.write(chunk)
    print(f"    {src.stat().st_size/1e6:.0f} MB -> extracting audio", flush=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn",
         "-ac", "1", "-ar", "16000", "-c:a", "aac", "-b:a", "24k",
         str(AUDIO / f"{slug}.m4a")], check=True)
    src.unlink(missing_ok=True)
    print(f"    audio {(AUDIO / f'{slug}.m4a').stat().st_size/1e6:.1f} MB", flush=True)


def main() -> int:
    slugs = sys.argv[1:]
    if not slugs:
        print("usage: ingest.py <slug> [slug ...]", file=sys.stderr)
        return 2
    meta = json.loads((BASE / "data" / "episodes_meta.json").read_text())
    for slug in slugs:
        m = meta.get(slug)
        if not m:
            print(f"!! unknown slug {slug}", file=sys.stderr)
            continue
        print(f"\n### {slug}", flush=True)
        try:
            if not have_audio(slug):
                make_audio(slug, m["enclosure"])
            pipeline.run(slug, m)
        except Exception as e:                    # noqa: BLE001
            # One bad episode must not abort a batch of fifty.
            print(f"!! {slug} failed: {type(e).__name__}: {str(e)[:200]}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
