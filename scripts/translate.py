"""Tier-1 translation: abstracts, key points, and pearls into ko / fr / de.

Deliberately NOT the full transcripts. Translating 228 hours of verbatim ASR
prose into three languages means paying for millions of words nobody reads;
abstracts + key points + pearls are a few percent of the volume and carry most
of the value. Full transcripts are Tier 2 -- translated lazily on first request
and cached (see ADDENDUM §4).

Safety rules are in the prompt and they matter more than fluency here: drug
names, doses, numbers and units are preserved verbatim, because a mistranslated
dose in a clinical reference is the failure mode worth engineering against.
Output is stored under ep["i18n"][lang]; English is never overwritten and stays
visible in the UI alongside any translation.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ppq  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
EPISODES = BASE / "data" / "episodes"
LANGS = {"ko": "Korean", "fr": "French", "de": "German"}

SYS = """You translate clinical teaching material for physicians into {lang}.

Absolute rules:
- NEVER translate or alter: drug names (keep the generic/INN spelling), doses,
  numbers, units, lab values, percentages, or scoring-scale names.
- Keep standard medical abbreviations in their internationally used form.
- Translate for a physician audience: use professional clinical register.
- Preserve meaning exactly. Never add, omit, explain, or soften content.

Input is JSON. Return JSON with the SAME keys and array lengths, values translated.
Return ONLY the JSON object."""


def translate_episode(slug: str, ep: dict) -> dict:
    i18n = ep.get("i18n", {})
    for code, name in LANGS.items():
        if i18n.get(code):
            print(f"  {code}: cached")
            continue
        pearls = {p["pearl_id"]: p["text"] for p in ep.get("pearls", [])}
        kps = ep.get("key_points", [])
        print(f"  {code}: translating {len(kps)} points, {len(pearls)} pearls ...")

        # Batched. A single call carrying 37 pearls hung for 6+ minutes and
        # never returned; smaller requests also mean one failure costs one
        # batch instead of the whole episode.
        head = _call(name, {"abstract": ep.get("abstract", ""), "key_points": kps})
        if not head:
            print("    ! header batch failed, skipping language")
            continue
        # A length mismatch means items were dropped or invented -- keep English
        # rather than ship a misaligned list.
        if len(head.get("key_points", [])) != len(kps):
            head["key_points"] = kps

        items = list(pearls.items())
        got = {}
        for i in range(0, len(items), 10):
            batch = dict(items[i:i + 10])
            r = _call(name, {"pearls": batch})
            got.update((r or {}).get("pearls", {}) or {})
            print(f"    pearls {min(i+10,len(items))}/{len(items)}")

        i18n[code] = {"abstract": head.get("abstract", ""),
                      "key_points": head.get("key_points", kps),
                      "pearls": got, "_model": ppq.CHAT_MODEL, "_mt": True}
        print(f"    -> ok ({len(got)}/{len(pearls)} pearls)")
    ep["i18n"] = i18n
    return ep


if __name__ == "__main__":
    only = sys.argv[1:]
    for f in sorted(EPISODES.glob("*.json")):
        ep = json.loads(f.read_text())
        if only and ep["slug"] not in only:
            continue
        if not ep.get("pearls"):
            continue
        print(f"\n=== {ep['slug']} ===")
        ep = translate_episode(ep["slug"], ep)
        f.write_text(json.dumps(ep, indent=2, ensure_ascii=False))
    print(f"\nTOTAL SPEND: ${ppq.total_spend():.4f}")
