"""Parallel backfill of the whole archive.

Serially the pipeline runs ~5 minutes an episode, almost all of it waiting on
HTTP, so 255 episodes would take about a day. The work is I/O bound, so a small
thread pool collapses that to a few hours. Concurrency is kept modest to stay
well clear of provider rate limits.

Safe to interrupt and re-run: every stage is cached per episode, so a restart
resumes rather than re-paying. Video is deleted as soon as audio is extracted,
so peak disk stays at roughly one source file per worker.

    python3 scripts/backfill.py [--workers N] [--limit N] [slug ...]
    python3 scripts/backfill.py --reenrich all|<slug ...>

--reenrich re-runs only pearl/topic extraction with the current prompt, on
lectures that are already transcribed (no audio, no transcription cost).

With slugs, only those episodes are considered. Exits non-zero when any episode
fails, so CI shows the failure instead of quietly carrying on.
"""
import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import ingest    # noqa: E402
import pipeline  # noqa: E402

PRINT_LOCK = threading.Lock()
STATE = {"done": 0, "failed": 0, "total": 0}
FAILED = []


def log(msg: str) -> None:
    with PRINT_LOCK:
        n, t = STATE["done"] + STATE["failed"], STATE["total"]
        print(f"[{n:3d}/{t}] {msg}", flush=True)


DEADLINE = [float("inf")]


def one(slug: str, meta: dict, reenrich: bool = False) -> bool:
    if time.time() > DEADLINE[0]:
        STATE["skipped"] = STATE.get("skipped", 0) + 1
        return False
    try:
        if reenrich:
            ep = pipeline.load(slug)
            ep["reenrich"] = True
            ep = pipeline.stage_enrich(slug, ep)
            pipeline.save(slug, ep)
            STATE["done"] += 1
            log(f"OK   {slug[:52]:52s} {len(ep.get('pearls',[])):3d} pearls (re-extracted)")
            return True
        if not ingest.have_audio(slug):
            ingest.make_audio(slug, meta["enclosure"])
        ep = pipeline.load(slug) or dict(meta)
        ep.update({k: v for k, v in meta.items() if k not in ep or not ep[k]})
        for stage in (pipeline.stage_transcribe, pipeline.stage_correct,
                      pipeline.stage_enrich, pipeline.stage_embed):
            ep = stage(slug, ep)
            pipeline.save(slug, ep)
        STATE["done"] += 1
        log(f"OK   {slug[:52]:52s} {len(ep.get('pearls',[])):3d} pearls")
        return True
    except Exception as e:                            # noqa: BLE001
        STATE["failed"] += 1
        FAILED.append(slug)
        log(f"FAIL {slug[:52]:52s} {type(e).__name__}: {str(e)[:80]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reenrich", action="store_true",
                    help="re-extract pearls on transcribed lectures (slugs, or 'all')")
    ap.add_argument("--force", action="store_true",
                    help="with --reenrich: redo lectures already on the current prompt")
    ap.add_argument("--summary", default="", help="write a JSON run summary here (for CI)")
    ap.add_argument("--exclude", default="", help="file of slugs to leave out (one per line)")
    ap.add_argument("--max-minutes", type=float, default=0,
                    help="stop starting new lectures after this long, so CI reaches its commit")
    ap.add_argument("slugs", nargs="*")
    a = ap.parse_args()
    if a.max_minutes:
        DEADLINE[0] = time.time() + a.max_minutes * 60

    if a.reenrich:
        want = None if a.slugs in ([], ["all"]) else set(a.slugs)
        todo = [(p.stem, {}) for p in sorted(pipeline.EPISODES.glob("*.json"))
                if (want is None or p.stem in want)]
        todo = [(s, m) for s, m in todo
                if pipeline.load(s).get("transcript") and not pipeline.load(s).get("duplicate_of")]
        if a.exclude and Path(a.exclude).exists():
            skip = set(Path(a.exclude).read_text().split())
            todo = [(s, m) for s, m in todo if s not in skip]
        # Resumable: lectures already on the current prompt are skipped, so a
        # repeated request (or the next batch) only pays for what is left.
        if not a.force:
            todo = [(s, m) for s, m in todo
                    if pipeline.load(s).get("enrich_version") != pipeline.ENRICH_VERSION]
        if a.limit:
            todo = todo[:a.limit]
        STATE["total"] = len(todo)
        print(f"re-extract: {len(todo)} lectures, {a.workers} workers", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for _ in as_completed([pool.submit(one, s, m, True) for s, m in todo]):
                pass
        left = STATE.get("skipped", 0)
        if a.summary:
            Path(a.summary).write_text(json.dumps(
                {"attempted": len(todo), "done": STATE["done"], "failed": STATE["failed"],
                 "not_started": left, "failed_slugs": sorted(FAILED)}))
        print(f"\ndone={STATE['done']} failed={STATE['failed']} not started (time budget)={left}",
              flush=True)
        if left:
            print("  re-run with the same request to continue where this stopped", flush=True)
        return 1 if STATE["failed"] or left else 0

    meta = json.loads((BASE / "data" / "episodes_meta.json").read_text())
    for s in a.slugs:
        if s not in meta:
            print(f"!! unknown slug {s}", flush=True)
    todo = []
    for slug, m in meta.items():
        if a.slugs and slug not in a.slugs:
            continue
        if pipeline.is_settled(pipeline.load(slug)):
            continue
        todo.append((slug, m))
    if a.limit:
        todo = todo[:a.limit]

    STATE["total"] = len(todo)
    mins = sum(m["duration_sec"] for _, m in todo) / 60
    print(f"backfill: {len(todo)} episodes, {mins/60:.1f} h audio, "
          f"{a.workers} workers", flush=True)

    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        futs = {pool.submit(one, s, m): s for s, m in todo}
        for _ in as_completed(futs):
            pass

    print(f"\ndone={STATE['done']} failed={STATE['failed']}", flush=True)
    return 1 if STATE["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
