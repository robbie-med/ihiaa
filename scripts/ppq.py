"""Thin provider layer over an OpenAI-compatible endpoint (currently PPQ.AI).

Everything the pipeline needs from a model vendor goes through this file: STT,
embeddings, chat. That is deliberate -- PPQ is a small proxy and may not exist in
ten years. Swapping vendors should be editing BASE/model ids here, not a rewrite.

Two hard-won details baked in:

  * Base URL and model ids resolve from the environment at CALL time, never at
    import time. Resolving at import is what produces PPQ's notorious lying 401
    ("your api key is invalid") when the real fault is a wrong upstream URL.
  * Every paid call appends to a local, gitignored ledger so usage can be
    audited offline.
"""
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LEDGER = BASE_DIR / "data" / "cost_ledger.jsonl"

# Live PPQ pricing, fetched 2026-08-18. Kept here so the cost report is
# reproducible offline; refresh with scripts/refresh_pricing.py.
PRICING = {
    "stt_per_minute": 0.00633,                       # deepgram nova-3
    "openai/text-embedding-3-small": {"in": 0.022},  # USD per 1M tokens
    "deepseek/deepseek-v4-flash": {"in": 0.084189, "out": 0.168378},
    "deepseek/deepseek-v3.2": {"in": 0.283795, "out": 0.422},
}

STT_MODEL = "nova-3"
EMBED_MODEL = "openai/text-embedding-3-small"
EMBED_DIMS = 512          # keeps the in-browser index small as the corpus grows
CHAT_MODEL = "deepseek/deepseek-v4-flash"


def _base() -> str:
    """Resolved at call time, never at import time. See module docstring."""
    return os.environ.get("PPQ_BASE", "https://api.ppq.ai/v1").rstrip("/")


def _key() -> str:
    k = os.environ.get("PPQ_API_KEY")
    if k:
        return k.strip()
    return (BASE_DIR / ".ppq_key").read_text().strip()


def _log_cost(kind: str, model: str, usd: float, detail: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "model": model, "usd": round(usd, 6), **detail}
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def _post(path: str, payload: dict, timeout: int = 300) -> dict:
    req = urllib.request.Request(
        f"{_base()}{path}",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {_key()}",
                 "Content-Type": "application/json"},
    )
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()[:400]
            last = f"HTTP {e.code}: {body}"
            if e.code == 401:
                # Almost never a bad key -- check the URL actually being hit.
                raise RuntimeError(
                    f"401 from {_base()}{path}. Verify the base URL before "
                    f"assuming the key is bad. Body: {body}") from e
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last) from e
        except Exception as e:                       # noqa: BLE001
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"{path} failed after retries: {last}")


# --------------------------------------------------------------------------
# Speech to text
# --------------------------------------------------------------------------
def transcribe(audio_path: Path, minutes: float, prompt: str = "") -> dict:
    """Multipart upload to /audio/transcriptions. PPQ caps uploads at 25MB."""
    size_mb = audio_path.stat().st_size / 1e6
    if size_mb > 25:
        raise ValueError(f"{audio_path.name} is {size_mb:.1f}MB, over the 25MB cap")

    boundary = "----ihiaa" + os.urandom(8).hex()
    fields = [("model", STT_MODEL), ("response_format", "verbose_json"),
              ("timestamp_granularities[]", "word")]
    if prompt:
        fields.append(("prompt", prompt))

    parts = []
    for field, val in fields:
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                     f'name="{field}"\r\n\r\n{val}\r\n'.encode())
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; "
                 f'name="file"; filename="{audio_path.name}"\r\n'
                 "Content-Type: audio/mp4\r\n\r\n".encode())
    parts.append(audio_path.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{_base()}/audio/transcriptions", data=body,
        headers={"Authorization": f"Bearer {_key()}",
                 "Content-Type": f"multipart/form-data; boundary={boundary}"})

    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                out = json.loads(r.read())
            _log_cost("stt", STT_MODEL, minutes * PRICING["stt_per_minute"],
                      {"file": audio_path.name, "minutes": round(minutes, 2)})
            return out
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode()[:300]}"
            if e.code not in (429, 500, 502, 503, 504):
                raise RuntimeError(last) from e
        except Exception as e:                       # noqa: BLE001
            last = repr(e)
        time.sleep(2 ** attempt)
    raise RuntimeError(f"transcription failed: {last}")


# --------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------
def embed(texts: list[str]) -> list[list[float]]:
    # The API rejects the whole batch if any element is empty, so callers must
    # not contain empty strings. Fail loudly here rather than eat a vague 400.
    if any(not (t or "").strip() for t in texts):
        raise ValueError("embed(): batch contains an empty string")
    out = _post("/embeddings", {"model": EMBED_MODEL, "input": texts,
                                "dimensions": EMBED_DIMS})
    toks = out.get("usage", {}).get("prompt_tokens", sum(len(t) // 4 for t in texts))
    _log_cost("embed", EMBED_MODEL,
              toks / 1e6 * PRICING[EMBED_MODEL]["in"],
              {"n": len(texts), "tokens": toks})
    return [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]


# --------------------------------------------------------------------------
# Chat / structured extraction
# --------------------------------------------------------------------------
def chat_json(system: str, user: str, model: str = None, max_tokens: int = 8000,
              temperature: float = 0.2) -> dict:
    model = model or CHAT_MODEL
    out = _post("/chat/completions", {
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    })
    u = out.get("usage", {})
    pi, po = PRICING.get(model, {"in": 0, "out": 0}).values()
    _log_cost("chat", model,
              u.get("prompt_tokens", 0) / 1e6 * pi + u.get("completion_tokens", 0) / 1e6 * po,
              {"in": u.get("prompt_tokens"), "out": u.get("completion_tokens")})

    # content can come back null (empty generation / provider hiccup) -- treat
    # that as an empty result rather than exploding on json.loads(None).
    txt = (out.get("choices") or [{}])[0].get("message", {}).get("content")
    if not txt:
        return {}
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            return json.loads(txt[s:e + 1])
        raise


def total_spend() -> float:
    if not LEDGER.exists():
        return 0.0
    return sum(json.loads(l)["usd"] for l in LEDGER.read_text().splitlines() if l.strip())
