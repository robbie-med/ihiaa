"""Parallel backfill of the whole archive.

Serially the pipeline runs ~5 minutes an episode, almost all of it waiting on
HTTP, so 255 episodes would take about a day. The work is I/O bound, so a small
thread pool collapses that to a few hours. Concurrency is kept modest to stay
well clear of provider rate limits.

Safe to interrupt and re-run: every stage is cached per episode, so a restart
resumes rather than re-paying. Video is deleted as soon as audio is extracted,
so peak disk stays at roughly one source file per worker.

    python3 scripts/backfill.py [--workers N] [--limit N]
"""
import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "scripts"))
import ingest    # noqa: E402
import pipeline  # noqa: E402
import ppq       # noqa: E402

PRINT_LOCK = threading.Lock()
STATE = {"done": 0, "failed": 0, "total": 0}


def log(msg: str) -> None:
    with PRINT_LOCK:
        n, t = STATE["done"] + STATE["failed"], STATE["total"]
        print(f"[{n:3d}/{t}] {msg}", flush=True)


def one(slug: str, meta: dict) -> bool:
    try:
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
        log(f"FAIL {slug[:52]:52s} {type(e).__name__}: {str(e)[:80]}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    meta = json.loads((BASE / "data" / "episodes_meta.json").read_text())
    todo = []
    for slug, m in meta.items():
        ep = pipeline.load(slug)
        if ep.get("chunks"):
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
