"""Report how much Deepgram credit is left, so transcription never fails silently.

Deepgram bills per minute of audio, and new lectures cannot be transcribed once
the balance runs out. This asks the Deepgram API for every project's balance
and prints the dollar amounts only, never the key.

    DEEPGRAM_API_KEY=... python3 scripts/check_credits.py [--warn-below 5]

Exit status 2 when the balance is below --warn-below, so CI can flag it.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.deepgram.com/v1"
BASE = Path(__file__).resolve().parent.parent
PRICE_PER_MIN = 0.0043          # nova-3 pre-recorded, as used in stt_deepgram.py


def get(path: str, key: str) -> dict:
    req = urllib.request.Request(API + path, headers={"Authorization": f"Token {key}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--warn-below", type=float, default=5.0, help="USD")
    a = ap.parse_args()

    key = (os.environ.get("DEEPGRAM_API_KEY") or "").strip()
    if not key and (BASE / ".deepgram_key").exists():
        key = (BASE / ".deepgram_key").read_text().strip()
    if not key:
        print("No Deepgram key (set DEEPGRAM_API_KEY).")
        return 1

    try:
        projects = get("/projects", key).get("projects", [])
    except urllib.error.HTTPError as e:
        print(f"Deepgram rejected the key listing projects: HTTP {e.code} {e.read()[:200]!r}")
        return 1

    total, lines = 0.0, []
    for p in projects:
        try:
            bal = get(f"/projects/{p['project_id']}/balances", key).get("balances", [])
        except urllib.error.HTTPError as e:
            lines.append(f"- {p.get('name', 'project')}: balance not readable with this key "
                         f"(HTTP {e.code}; the key may lack billing access)")
            continue
        for b in bal:
            amt = float(b.get("amount", 0))
            total += amt
            lines.append(f"- {p.get('name', 'project')}: {amt:.2f} {b.get('units', 'usd').upper()}")
        if not bal:
            lines.append(f"- {p.get('name', 'project')}: no prepaid balance on record")

    hours = total / PRICE_PER_MIN / 60 if total else 0
    summary = ["### Deepgram credit", *lines,
               "", f"**Total: ${total:.2f}**, about {hours:.0f} hours of audio "
               f"(~{hours / 1.2:.0f} lectures at ~70 min each)."]
    print("\n".join(summary))
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as fh:
            fh.write("\n".join(summary) + "\n")
    if projects and total < a.warn_below:
        print(f"::warning title=Deepgram credit low::${total:.2f} left; new lectures "
              f"will fail to transcribe once it runs out.")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
