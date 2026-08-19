"""Speech-to-text via the Deepgram API.

Uses `nova-3-medical`, which is tuned for clinical vocabulary -- important here
because general-purpose models mangle drug names, and a mis-transcribed drug
name in a clinical reference is the failure mode worth engineering against.
Keyterm boosting further pins down the terms most often misheard.

Returns the same shape as the other providers so the pipeline does not care
which one it is talking to.
"""
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LEDGER = BASE_DIR / "data" / "cost_ledger.jsonl"

MODEL = "nova-3-medical"
PRICE_PER_MIN = 0.0043

# Boosted vocabulary. nova-3 accepts `keyterm`; these are terms the general
# model is most likely to mangle across a primary-care curriculum. Extend per
# episode via the `keyterms` argument rather than growing this list forever.
BASE_KEYTERMS = [
    "sumatriptan", "rizatriptan", "naproxen", "acetaminophen", "Excedrin",
    "topiramate", "propranolol", "amitriptyline", "erenumab", "galcanezumab",
    "ubrogepant", "rimegepant", "dihydroergotamine", "metoclopramide",
    "prochlorperazine", "ondansetron", "metformin", "empagliflozin",
    "semaglutide", "lisinopril", "amlodipine", "hydrochlorothiazide",
    "atorvastatin", "gabapentin", "onychomycosis", "osteomyelitis",
    "Charcot", "monofilament", "debridement", "cellulitis", "neuropathy",
]


def _key() -> str:
    k = os.environ.get("DEEPGRAM_API_KEY")
    if k:
        return k.strip()
    return (BASE_DIR / ".deepgram_key").read_text().strip()


def _log_cost(usd: float, detail: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a") as fh:
        fh.write(json.dumps({"kind": "stt", "model": MODEL,
                             "usd": round(usd, 6), "provider": "deepgram",
                             **detail}) + "\n")


def transcribe(audio_path: Path, minutes: float, keyterms: list = None) -> dict:
    """Transcribe a whole file. No 25MB cap here, so no chunking needed."""
    params = [("model", MODEL), ("smart_format", "true"), ("punctuate", "true"),
              ("paragraphs", "true"), ("diarize", "true"), ("utterances", "true")]
    for t in (BASE_KEYTERMS + (keyterms or [])):
        params.append(("keyterm", t))
    url = "https://api.deepgram.com/v1/listen?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(
        url, data=audio_path.read_bytes(),
        headers={"Authorization": f"Token {_key()}",
                 "Content-Type": "audio/mp4"})

    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                out = json.loads(r.read())
            break
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:300]}"
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last) from e
        except Exception as e:                        # noqa: BLE001
            last = repr(e)
        time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"deepgram failed: {last}")

    alt = out["results"]["channels"][0]["alternatives"][0]
    _log_cost(minutes * PRICE_PER_MIN,
              {"file": audio_path.name, "minutes": round(minutes, 2)})

    words = [{"w": w.get("punctuated_word") or w.get("word"),
              "s": round(w.get("start", 0), 2), "e": round(w.get("end", 0), 2),
              "spk": w.get("speaker")} for w in alt.get("words", [])]

    # Prefer utterances (diarised, sentence-ish) over raw paragraph splits --
    # they give cleaner transcript lines and carry a speaker id per line.
    segments = []
    for u in out["results"].get("utterances", []) or []:
        segments.append({"t": round(u.get("start", 0), 2),
                         "text": (u.get("transcript") or "").strip(),
                         "spk": u.get("speaker")})
    if not segments:
        for p in alt.get("paragraphs", {}).get("paragraphs", []):
            txt = " ".join(s.get("text", "") for s in p.get("sentences", []))
            segments.append({"t": round(p.get("start", 0), 2), "text": txt.strip(),
                             "spk": p.get("speaker")})

    n_spk = len({w["spk"] for w in words if w.get("spk") is not None})
    return {"text": alt.get("transcript", ""), "words": words,
            "segments": [s for s in segments if s["text"]],
            "model": MODEL, "n_speakers": n_spk}
