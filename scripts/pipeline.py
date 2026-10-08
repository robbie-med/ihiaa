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


def is_settled(ep: dict) -> bool:
    """True when an episode needs no more processing.

    Either it is fully processed (has chunks), or it has been marked
    `status: "unavailable"` -- a source that can never yield a transcript, such as
    a recording with a silent audio track. Without that second case such an
    episode stays "outstanding" forever, and every scheduled run re-downloads it
    and pays to transcribe the silence again.
    """
    return bool(ep.get("chunks")) or ep.get("status") == "unavailable"


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
ENRICH_VERSION = "v2"
ENRICH_SYS = """You are indexing a family-medicine residency lecture for a searchable
clinical knowledge base used by residents.

The transcript is split into numbered segments: "[#12 S0] text". S0, S1... are
different voices; the presenter is usually the voice that talks most.

Return JSON with exactly these keys:
{
 "abstract": "3-4 sentence summary",
 "key_points": ["6-10 substantive teaching points"],
 "topics": [{"label":"Title Case clinical concept","kind":"condition|drug|procedure|concept"}],
 "pearls": [
   {"text": "one specific, actionable teaching point in 1-2 sentences",
    "seg": 12,
    "verbatim": "the EXACT words from that segment (and the next few) this came from",
    "applies_to": "who/when it applies, as stated: e.g. 'adults with CKD', 'in pregnancy', 'outpatient'; '' if general",
    "type": "dosing|pitfall|red_flag|exam_technique|dx_criteria|practice|judgment|procedure|concept"}
 ]
}

Rules for pearls -- these matter more than coverage:
- ACTIONABLE and SPECIFIC. "Diabetes is important" is not a pearl.
  "Check monofilament sensation at 10 sites; loss of 4 predicts ulceration" is.
- Only what the PRESENTER teaches. Skip audience questions and wrong answers from
  the room unless the presenter explicitly confirms them.
- Every number, dose, threshold or duration in "text" must have been SAID in the
  quoted passage. Never add numbers, doses or guideline values from your own
  knowledge, and never do arithmetic the speaker did not do. If the speaker gave
  no number, the pearl has no number.
- Keep the speaker's claim as stated, even if you think guidelines say otherwise.
  Do not correct or soften it; that is for the reader to judge.
- "verbatim" MUST be copied character-for-character from the segment cited in
  "seg". One contiguous span, no "...". It locates the audio. If you cannot quote
  a supporting span, omit the pearl.
- "applies_to" carries the population, setting or condition the speaker attached
  to the advice. It is what distinguishes "give X" in pregnancy from "avoid X" in
  heart failure, so do not drop it.
- Prefer 8-15 excellent pearls over 40 mediocre ones.
- Topics should be canonical clinical concepts, not phrasings from the talk."""


def _dedash(s: str) -> str:
    """Strip em/en dashes from model prose.

    The extraction model writes them constantly; the lecturers do not talk that
    way, and they are a visible tell that the text is machine-written. Applied
    at ingest so the habit does not come back on every new episode.
    """
    s = re.sub(r"\s*[\u2014\u2013]\s*", ", ", s or "")
    s = re.sub(r",\s*,", ",", s)
    return re.sub(r",\s*\.", ".", s)


def numbered_segments(ep: dict) -> list:
    """Segments with the terminology fixes applied, as '[#i Sk] text' lines."""
    fixes = [(re.compile(r"(?<!\w)" + re.escape(f["from"]) + r"(?!\w)"), f["to"])
             for f in ep.get("fixes", [])]
    out = []
    for i, sg in enumerate(ep.get("segments") or []):
        text = sg["text"]
        for rx, to in fixes:
            text = rx.sub(to.replace("\\", r"\\"), text)
        out.append((f"[#{i} S{sg.get('spk', 0)}] {text}", len(text.split())))
    return out


def stage_enrich(slug: str, ep: dict) -> dict:
    if ep.get("pearls_raw") and ep.get("enrich_version") == ENRICH_VERSION:
        print(f"  enrich: cached ({len(ep.get('pearls', []))} pearls)")
        return ep
    if ep.get("pearls_raw") and not ep.get("reenrich"):
        print(f"  enrich: cached, older prompt {ep.get('enrich_version', 'v1')} "
              f"({len(ep.get('pearls', []))} pearls)")
        return ep
    # Chunks of ~3000 words that overlap by ~300, so a point made across a chunk
    # boundary is seen whole at least once. refine_pearls drops the duplicates.
    segs = numbered_segments(ep)
    chunks, i = [], 0
    while i < len(segs):
        j, n = i, 0
        while j < len(segs) and n < 3000:
            n += segs[j][1]
            j += 1
        chunks.append("\n".join(s for s, _ in segs[i:j]))
        if j >= len(segs):
            break
        back, k = 0, j
        while k > i + 1 and back < 300:
            k -= 1
            back += segs[k][1]
        i = k
    abstract, kp, topics, pearls, failed = "", [], [], [], 0
    for n, c in enumerate(chunks):
        print(f"  enrich: chunk {n+1}/{len(chunks)} ...")
        user = (f"Lecture: {ep['title']}\nPresenter: {ep.get('speaker', 'Unknown')}\n"
                f"(part {n+1} of {len(chunks)})\n\nTRANSCRIPT:\n{c}")
        try:
            r = ppq.chat_json(ENRICH_SYS, user, max_tokens=8000)
        except Exception as e:                        # noqa: BLE001
            print(f"    ! {type(e).__name__}: {str(e)[:120]}")
            failed += 1
            continue
        if n == 0:
            abstract = r.get("abstract", "")
        kp += r.get("key_points", [])
        topics += r.get("topics", [])
        pearls += r.get("pearls", [])
    # Re-extracting must never trade a complete set of pearls for a partial one
    # because a request failed. Keep the old ones and report the lecture as failed.
    if failed and ep.get("pearls_raw"):
        raise RuntimeError(f"{failed}/{len(chunks)} chunks failed; previous pearls kept")
    ep["abstract"] = _dedash(abstract)
    ep["key_points"] = [_dedash(k) for k in kp[:12]]
    ep["topics_raw"] = topics
    # The model's output is kept verbatim; scripts/refine_pearls.py turns it into
    # anchored, number-checked pearls and can be re-run for free.
    ep["pearls_raw"] = pearls
    ep["enrich_model"] = ppq.CHAT_MODEL
    ep["enrich_version"] = ENRICH_VERSION
    ep.pop("reenrich", None)
    import refine_pearls
    ep["pearls"] = refine_pearls.refine(ep)
    ep["pearls_fingerprint"] = refine_pearls.fingerprint(ep)
    print(f"    -> {len(ep['pearls'])} of {len(pearls)} pearls kept, {len(topics)} topic mentions")
    return ep


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
