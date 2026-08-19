"""Per-episode pipeline: transcribe -> correct -> enrich -> embed.

Every stage is idempotent and resumable. Stage output is written to
data/episodes/<slug>.json and a stage is skipped if its output already exists,
because re-running a stage means paying for it again.

Design decisions that came out of measurement, not theory (see COST_REPORT.md):

  * No audio speed-up. 1.5x cut cost 33% but turned "sumatriptan and naproxen"
    into "symmetrictine and proxet". Unacceptable in a drug reference.
  * PPQ ignores the STT `prompt` parameter -- verified byte-identical output
    with and without a 30-term vocabulary. Deepgram keyterm boosting is not
    reachable through this proxy, so drug names must be repaired downstream.
  * Hence stage `correct`: a cheap LLM pass that repairs clinical terminology
    using surrounding context. The raw ASR text is ALWAYS retained alongside
    the corrected text so any correction can be audited or reverted.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppq            # noqa: E402  chat + embeddings
import stt_deepgram   # noqa: E402  speech-to-text

BASE = Path(__file__).resolve().parent.parent
EPISODES = BASE / "data" / "episodes"
PROMPT_VERSION = "v1"


def load(slug: str) -> dict:
    p = EPISODES / f"{slug}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save(slug: str, ep: dict) -> None:
    EPISODES.mkdir(parents=True, exist_ok=True)
    (EPISODES / f"{slug}.json").write_text(json.dumps(ep, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------- transcribe
def stage_transcribe(slug: str, ep: dict) -> dict:
    """Transcribe via Deepgram direct (see stt_deepgram for why, with numbers)."""
    if ep.get("transcript_raw") and ep.get("stt_model") == stt_deepgram.MODEL:
        print(f"  transcribe: cached ({len(ep['transcript_raw'].split())} words)")
        return ep
    audio = BASE / "data" / "audio" / f"{slug}.m4a"
    mins = ep["duration_sec"] / 60
    print(f"  transcribe: {mins:.1f} min, {audio.stat().st_size/1e6:.1f} MB "
          f"via {stt_deepgram.MODEL} ...")
    out = stt_deepgram.transcribe(audio, minutes=mins)
    ep["transcript_raw"] = out["text"]
    ep["words"] = out["words"]
    ep["segments"] = out["segments"]
    ep["stt_model"] = out["model"]
    ep["n_speakers"] = out.get("n_speakers", 0)
    # A re-transcribe invalidates everything derived from the old text.
    for k in ("transcript", "fixes", "abstract", "key_points",
              "topics_raw", "pearls", "chunks", "i18n"):
        ep.pop(k, None)
    print(f"    -> {len(ep['transcript_raw'].split())} words, "
          f"{len(ep['words'])} word timings, {len(ep['segments'])} segments, "
          f"{ep['n_speakers']} speakers")
    return ep


# ------------------------------------------------------------------- correct
CORRECT_SYS = """You find speech-to-text errors in medical lecture transcripts.

The ASR system mangles drug names, eponyms, anatomy, and clinical terms. Identify
those errors. Do NOT rewrite the transcript -- report ONLY a list of replacements.

Rules:
- Report only clear CLINICAL terminology errors, judged from context.
  e.g. "Sumitriptan"->"sumatriptan", "Aciduminophen"->"acetaminophen",
       "ethedrin"->"Excedrin", "woods"->"wounds"
- "from" must be copied EXACTLY as it appears in the text, including case.
- Never "fix" ordinary speech, filler, grammar, or punctuation.
- If a passage is garbled beyond recognition, skip it. Do not guess.
- Prefer precision over recall. A wrong "fix" corrupts a clinical reference.

Return JSON: {"fixes":[{"from":"<exact text>","to":"<correction>"}]}"""


def stage_correct(slug: str, ep: dict) -> dict:
    """Repair ASR terminology by collecting replacements, not rewriting text.

    Asking the model to re-emit the whole chunk cost ~1600 output tokens per
    1200 words and ran at roughly 5 minutes a chunk -- and any rewrite risks
    silent paraphrase, which is why the earlier version needed a length-drift
    guard. Collecting a fix list instead is ~50x fewer output tokens, applies
    deterministically via string replacement, and leaves an auditable record of
    exactly what changed. transcript_raw is always retained.
    """
    if ep.get("transcript"):
        print(f"  correct: cached ({len(ep.get('fixes',[]))} fixes)")
        return ep
    raw = ep["transcript_raw"]
    words, size = raw.split(), 3000
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]

    all_fixes = []
    for i, c in enumerate(chunks):
        print(f"  correct: chunk {i+1}/{len(chunks)} ...")
        try:
            r = ppq.chat_json(CORRECT_SYS, c, max_tokens=3000)
            all_fixes.extend(r.get("fixes", []))
        except Exception as e:                        # noqa: BLE001
            print(f"    ! {type(e).__name__}: {str(e)[:90]}")

    text, applied, rejected = raw, [], []

    # Applying fixes by naive substring replace is actively dangerous. Two real
    # failures caught here on the first run:
    #   'd' -> 'deep vein thrombosis'   fired 1500x, hitting the "d" in "do"
    #   'triptan' -> 'triptans'         turned existing "triptans" into "triptanss"
    # So: require a specific enough token, match only on word boundaries, and
    # refuse any single fix that wants to rewrite implausibly much of the text.
    MIN_LEN, MAX_HITS = 4, 60
    for f in all_fixes:
        frm, to = (f.get("from") or "").strip(), (f.get("to") or "").strip()
        if not frm or not to or frm == to or len(frm) > 60:
            continue
        if len(frm) < MIN_LEN:
            rejected.append({"from": frm, "to": to, "why": "too short"})
            continue
        pat = re.compile(r"(?<!\w)" + re.escape(frm) + r"(?!\w)")
        n = len(pat.findall(text))
        if not n:
            continue
        if n > MAX_HITS:
            rejected.append({"from": frm, "to": to, "why": f"{n} hits", "n": n})
            continue
        text = pat.sub(to.replace("\\", r"\\"), text)
        applied.append({"from": frm, "to": to, "n": n})

    ep["transcript"] = text
    ep["fixes"] = applied
    ep["fixes_rejected"] = rejected
    ep["correct_model"] = ppq.CHAT_MODEL
    ep["prompt_version"] = PROMPT_VERSION
    print(f"    -> {len(applied)} fixes applied "
          f"({sum(f['n'] for f in applied)} replacements), "
          f"{len(rejected)} rejected as unsafe")
    return ep


# -------------------------------------------------------------------- enrich
ENRICH_SYS = """You are indexing a family-medicine residency lecture for a searchable
clinical knowledge base used by residents.

Return JSON with exactly these keys:
{
 "abstract": "3-4 sentence summary",
 "key_points": ["6-10 substantive teaching points"],
 "topics": [{"label":"Title Case clinical concept","kind":"condition|drug|procedure|concept"}],
 "pearls": [
   {"text":"one specific, actionable teaching point in 1-2 sentences",
    "verbatim":"the EXACT contiguous span from the transcript this came from",
    "type":"dosing|pitfall|red_flag|exam_technique|dx_criteria|practice|judgment"}
 ]
}

Rules for pearls -- these matter more than coverage:
- A pearl must be ACTIONABLE and SPECIFIC. "Diabetes is important" is not a pearl.
  "Check monofilament sensation at 10 sites; loss of 4 predicts ulceration" is.
- "verbatim" MUST be copied character-for-character from the transcript provided.
  Never paraphrase it. It is the provenance anchor used to locate the audio.
- If you cannot find an exact supporting span, omit the pearl entirely.
- Prefer 8-15 excellent pearls over 40 mediocre ones.
- Topics should be canonical clinical concepts, not phrasings from the talk."""


def stage_enrich(slug: str, ep: dict) -> dict:
    if ep.get("pearls"):
        print(f"  enrich: cached ({len(ep['pearls'])} pearls)")
        return ep
    txt = ep["transcript"]
    words, size = txt.split(), 3500
    chunks = [" ".join(words[i:i + size]) for i in range(0, len(words), size)]
    abstract, kp, topics, pearls = "", [], [], []
    for i, c in enumerate(chunks):
        print(f"  enrich: chunk {i+1}/{len(chunks)} ...")
        user = (f"Lecture: {ep['title']}\nSpeaker: {ep['speaker']}\n"
                f"(part {i+1} of {len(chunks)})\n\nTRANSCRIPT:\n{c}")
        try:
            r = ppq.chat_json(ENRICH_SYS, user, max_tokens=8000)
        except Exception as e:                        # noqa: BLE001
            print(f"    ! {type(e).__name__}: {str(e)[:120]}")
            continue
        if i == 0:
            abstract = r.get("abstract", "")
        kp += r.get("key_points", [])
        topics += r.get("topics", [])
        pearls += r.get("pearls", [])
    ep["abstract"] = abstract
    ep["key_points"] = kp[:12]
    ep["topics_raw"] = topics
    # Keep the model's raw output so anchoring can be re-tuned and re-run for
    # free, without paying for enrichment again.
    ep["pearls_raw"] = pearls
    ep["pearls"] = anchor_pearls(pearls, ep)
    ep["enrich_model"] = ppq.CHAT_MODEL
    print(f"    -> {len(ep['pearls'])} pearls anchored, {len(topics)} topic mentions")
    return ep


def anchor_pearls(pearls: list, ep: dict) -> list:
    """Attach a timestamp to each pearl by locating its verbatim span.

    Exact substring matching dropped 14 of 24 pearls, because the model lightly
    normalises the span it quotes. So: try exact first, then slide a window over
    the word stream and keep the best fuzzy match. A pearl still needs a real
    location above MIN_RATIO -- one that cannot be found is DROPPED, never kept
    with a guessed timestamp. An unanchored clinical claim is precisely the
    failure mode this design exists to prevent.

    `match` records how the anchor was found so the UI can be honest about it.
    """
    import difflib

    MIN_RATIO = 0.68
    words = ep.get("words") or []
    norm = lambda s: re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    toks = [norm(w["w"] or "").strip() for w in words]
    times = [w["s"] for w in words]
    flat = " ".join(toks)

    # char offset -> timestamp, for the exact-match path
    pos, idx = [], 0
    for t, ts in zip(toks, times):
        pos.append((idx, ts))
        idx += len(t) + 1

    out, dropped = [], 0
    for p in pearls:
        v = re.sub(r"\s+", " ", norm(p.get("verbatim", ""))).strip()
        vt = v.split()
        if len(vt) < 4:
            dropped += 1
            continue

        t, how = None, None
        at = flat.find(v[:120])
        if at >= 0:
            t, how = next((s for (o, s) in reversed(pos) if o <= at), 0), "exact"
        else:
            probe, best = vt[:14], 0.0
            step = max(1, len(probe) // 3)
            for i in range(0, max(1, len(toks) - len(probe)), step):
                r = difflib.SequenceMatcher(None, probe, toks[i:i + len(probe)]).ratio()
                if r > best:
                    best, t = r, times[i]
            how = "fuzzy"
            if best < MIN_RATIO:
                t = None

        if t is None:
            dropped += 1
            continue
        out.append({"pearl_id": f"{ep['slug']}-{int(t)}",
                    "text": p.get("text", "").strip(),
                    "verbatim": p.get("verbatim", "").strip(),
                    "type": p.get("type", "judgment"), "t": round(t, 1),
                    "match": how, "episode": ep["slug"], "speaker": ep["speaker"],
                    "model": ppq.CHAT_MODEL, "prompt_version": PROMPT_VERSION})
    if dropped:
        print(f"    ({dropped} pearls dropped - no locatable span)")
    return out


# --------------------------------------------------------------------- embed
def stage_embed(slug: str, ep: dict) -> dict:
    if ep.get("chunks"):
        print(f"  embed: cached ({len(ep['chunks'])} chunks)")
        return ep
    segs = ep.get("segments") or []
    chunks, cur, start = [], [], 0.0
    for s in segs:
        if not cur:
            start = s["t"]
        cur.append(s["text"])
        if len(" ".join(cur).split()) >= 180:
            chunks.append({"t": start, "text": " ".join(cur)})
            cur = cur[-1:]                       # small overlap for continuity
    if cur:
        chunks.append({"t": start, "text": " ".join(cur)})

    vecs = []
    for i in range(0, len(chunks), 64):
        batch = [c["text"] for c in chunks[i:i + 64]]
        print(f"  embed: {i+len(batch)}/{len(chunks)} ...")
        vecs += ppq.embed(batch)
    for c, v in zip(chunks, vecs):
        c["vec"] = [round(x, 5) for x in v]
    ep["chunks"] = chunks
    ep["embed_model"] = ppq.EMBED_MODEL
    ep["embed_dims"] = ppq.EMBED_DIMS
    print(f"    -> {len(chunks)} chunks @ {ppq.EMBED_DIMS}d")
    return ep


def run(slug: str, meta: dict) -> dict:
    print(f"\n=== {slug} : {meta['title'][:58]} ===")
    ep = load(slug) or dict(meta)
    ep.update({k: v for k, v in meta.items() if k not in ep or not ep[k]})
    for stage in (stage_transcribe, stage_correct, stage_enrich, stage_embed):
        ep = stage(slug, ep)
        save(slug, ep)
    return ep


if __name__ == "__main__":
    meta = json.loads((BASE / "data" / "episodes_meta.json").read_text())
    only = sys.argv[1:] or list(meta)
    for slug in only:
        run(slug, meta[slug])
    print(f"\nTOTAL SPEND: ${ppq.total_spend():.4f}")
