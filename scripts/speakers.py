"""Resolve who gave each lecture into one stable identity per person. Rules only.

The old approach took whatever sat between "Dr." and " on " in the title. That
produced 108 "speakers" for a much smaller faculty: 49 lectures as "Unknown"
(titles like "Pressors - Engstrom" or "Rheumatology with Dr. Lucashu"), and one
person split across several names ("Calvert" / "Jon Calvert", "Dunn" / "Kelly
Dunn" / "Dunn speaks", "Hermann" / "Herrmann", "Okeefe" / "Yasmin Keefe").

Evidence, per episode:
  1. the title          every "Dr./Drs./Rev./Officer NAME", "NAME, MD", "NAME on ..."
  2. the feed description, which often names the speaker when the title does not
                        ("Moral Injury" -> "Dr. Jim Ritchie presents on moral injury")
  3. the transcript     self-introductions ("I'm Doctor. Kelly Dunn"), used only to
                        supply a first name for a surname the title already gave

Merging, in this order, every step recorded so it can be audited:
  override     data/speaker_overrides.json, for nicknames and judgement calls
  same-episode "Dr. Miller" and "Dr. Gerald Miller" in one episode are one person
  variant      same first name, surname within one edit ("Hermann"/"Herrmann"),
               middle initials ignored ("Gerald C Miller"/"Gerald Miller")
  self-intro   transcript gives the first name for a bare surname
  surname      a bare surname joins the ONE full name that has it, unless the
               surname is common (Lee, Jones, Miller...), where that is a guess

Output: each episode gets `speakers` [{id, name}] and a display `speaker`;
data/speakers.json lists every person with every alias and why it was merged.
No model, no network, same answer every run.

    python3 scripts/speakers.py            # apply
    python3 scripts/speakers.py --report   # print the registry
    python3 scripts/speakers.py --selftest
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
META = BASE / "data" / "episodes_meta.json"
EPISODES = BASE / "data" / "episodes"
OVERRIDES = BASE / "data" / "speaker_overrides.json"
REGISTRY = BASE / "data" / "speakers.json"

HONORIFIC = r"(?:Drs?|Rev|Officer|Pastor|Prof)"
CRED = r"(?:MD|DO|PhD|MS|MPH|FACP|FAAFP|RN|NP|PA-C|PA|PharmD|MSW|LCSW)"
TOKEN = r"(?:[A-Z]\.?(?=\s)|[A-Z][A-Za-z'’]*(?:-[A-Z][A-Za-z'’]+)?(?![A-Za-z]))"
# Capitalised words in Title Case titles that end a name rather than continue it.
STOP = set("""On Speaks Speak Lecture Lectures Overviews Outpatient Morning Report Reports
With Of In At For From The And Teaching Teaches Presents Shares Discusses Leads
Part Pt Q A Rounds Grand Lecturing Covers Reviews Gives Explores Dives""".split())
# Bare surnames too common to attach to a full name without corroboration.
COMMON = set("""smith johnson williams brown jones garcia miller davis rodriguez martinez
hernandez lopez gonzalez wilson anderson thomas taylor moore jackson martin lee perez
thompson white harris sanchez clark ramirez lewis robinson walker young allen king
wright scott torres nguyen hill flores green adams nelson baker hall rivera campbell
mitchell carter roberts gomez phillips evans turner diaz parker cruz edwards collins
reyes stewart morris morales murphy cook rogers gutierrez ortiz morgan cooper peterson
bailey reed kelly howard ramos kim cox ward richardson watson brooks chavez wood james
bennett gray mendoza ruiz hughes price alvarez castillo sanders patel myers long ross
foster jimenez powell jenkins perry russell sullivan bell coleman butler henderson barnes
fisher vasquez simmons graham marshall owens harrison gibson wallace""".split())


# ---------------------------------------------------------------- parsing
def clean(name: str) -> str:
    s = name.replace("’", "'").replace(".", " ")
    s = re.sub(r"'s\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" ,&")


def _names_after(text: str, start: int) -> list:
    """Read 'Jon Calvert' or 'Jones & Shannon' or 'Greuel and White' from start."""
    out, cur, i = [], [], start
    toks = re.finditer(rf"\s*({TOKEN}|,|&|\band\b)", text[start:])
    for m in toks:
        if m.start() != i - start:
            break
        tok = m.group(1)
        i = start + m.end()
        if tok in (",", "&", "and"):
            if cur:
                out.append(" ".join(cur))
                cur = []
            # ", MD" after a name is a credential, not another person
            nxt = re.match(rf"\s*{CRED}\b", text[i:])
            if tok == "," and nxt:
                i += nxt.end()
            continue
        bare = tok.rstrip(".")
        if bare in STOP or bare in ("Dr", "Drs", "Rev", "Officer") or (len(bare) > 1 and bare.isupper()):
            break
        cur.append(tok)
        if len(cur) == 4:
            break
    if cur:
        out.append(" ".join(cur))
    return [clean(n) for n in out if clean(n)]


def parse_names(text: str, leading: bool) -> list:
    """All (name, honorific) mentions in a title or description."""
    if not text:
        return []
    found = []
    for m in re.finditer(rf"\b({HONORIFIC})\.?\s+(?=[A-Z])", text):
        for n in _names_after(text, m.end()):
            found.append((n, "Dr." if m.group(1).startswith("Dr") else m.group(1) + "."))
    # "Kelly Dunn, MD on ...", "Kyle Jones, DO", "Mark Crouch MD in PNG"
    for m in re.finditer(rf"\b((?:{TOKEN}\s+){{1,3}}{TOKEN}),?\s+{CRED}\b", text):
        n = clean(m.group(1))
        if n.split()[0] not in STOP:
            found.append((n, "Dr." if re.search(r"\b(MD|DO)\b", text[m.end() - 4:m.end()]) else ""))
    # A name that opens the text with no honorific: "Dianne Hughes on WIR",
    # "Sara Gadd speaks on ...", "Sasha and Irina from Kazakhstan share ..."
    if leading and not found:
        m = re.match(rf"\s*(?:The\s+)?((?:{TOKEN}\s*(?:,|&|and)?\s*){{1,4}}?)\s*(?:,\s*{CRED}\s+)?"
                     r"(?:on|speaks?|presents?|lectures?|shares?|discuss(?:es)?|gives?|teach(?:es)?|"
                     r"leads?|and team|from)\b", text)
        if m:
            fam = text.lstrip().startswith("The ")
            for n in _names_after(text, m.start(1)):
                found.append((("The " + n) if fam else n, ""))
    seen, out = set(), []
    for n, h in found:
        if n.lower() not in seen and n not in STOP:
            seen.add(n.lower())
            out.append((n, h))
    return out


def self_intro_first_names(transcript: str, surname: str) -> list:
    """First names spoken right before a known surname: 'Doctor. Kelly Dunn'."""
    head = transcript[: max(4000, len(transcript) // 6)]
    rx = rf"\bDoctor\.?\s+([A-Z][a-z]{{2,}})\s+{re.escape(surname)}\b"
    return [m for m in re.findall(rx, head) if m not in STOP]


# ------------------------------------------------------------- resolution
def key(s: str) -> str:
    return re.sub(r"[^a-z ]", "", s.lower().replace("-", " ")).strip()


def damerau1(a: str, b: str) -> bool:
    """True if a and b differ by at most one edit or one adjacent transposition."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        return len(d) == 1 or (len(d) == 2 and d[1] == d[0] + 1
                               and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    if len(a) > len(b):
        a, b = b, a
    return any(a == b[:i] + b[i + 1:] for i in range(len(b)))


def same_surname(a: str, b: str) -> bool:
    """Exact, or one typo apart when both are long enough for that to mean
    anything. 'Herrmann'/'Hermann' yes; 'Lee'/'Leo' no."""
    a, b = key(a), key(b)
    return a == b or (min(len(a), len(b)) >= 5 and damerau1(a, b))


def split(name: str):
    """-> (first, surname) with middle initials dropped; first is '' for a bare name."""
    parts = [p for p in name.split() if not (len(p) == 1 and len(name.split()) > 2)]
    if name.startswith("The "):
        return "", name
    if len(parts) == 1:
        return "", parts[0]
    return parts[0], parts[-1]


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower().replace("'", "")).strip("-")


def resolve(episodes: dict, overrides: dict) -> tuple:
    """episodes: slug -> {title, description, transcript}. Returns (by_episode, people)."""
    alias = {key(k): v for k, v in overrides.get("aliases", {}).items()}
    forced = overrides.get("episodes", {})
    keep_apart = {key(x) for x in overrides.get("distinct", [])}

    # 1. raw mentions per episode
    mentions = {}
    for slug, e in episodes.items():
        if slug in forced:
            mentions[slug] = [(n, "Dr.", "override") for n in forced[slug]]
            continue
        got = [(n, h, "title") for n, h in parse_names(e["title"], leading=True)]
        desc = e.get("description", "")
        if desc and desc.lower() != e["title"].lower():
            got += [(n, h, "description") for n, h in parse_names(desc, leading=True)]
        mentions[slug] = got

    # 2. aliases, then fold bare surnames into a full name in the same episode
    how = defaultdict(set)                       # canonical name -> evidence
    per_ep = {}
    for slug, ms in mentions.items():
        names, hon = [], {}
        for n, h, src in ms:
            tgt = alias.get(key(n))
            if tgt:
                how[(tgt, n)].add("override")
                n = tgt
            else:
                how[(n, n)].add(src)
            if n not in names:
                names.append(n)
            if h and not hon.get(n):
                hon[n] = h
        full = [n for n in names if split(n)[0]]
        merged = []
        for n in names:
            f, s = split(n)
            if not f:
                host = next((x for x in full if same_surname(split(x)[1], s)), None)
                if host:
                    how[(host, n)].add("same-episode")
                    hon.setdefault(host, hon.get(n, ""))
                    n = host
                elif key(s) not in COMMON:
                    # Not for common surnames: in a morning report "Doctor. Young
                    # Lee" is a resident being called on, not the presenter.
                    firsts = self_intro_first_names(episodes[slug].get("transcript", ""), s)
                    if firsts:
                        fn = Counter(firsts).most_common(1)[0][0] + " " + s
                        how[(fn, n)].add("self-intro")
                        hon.setdefault(fn, hon.get(n, "") or "Dr.")
                        n = fn
            if n not in merged:
                merged.append(n)
        per_ep[slug] = (merged, hon)

    # 3. variants across episodes: same first name + surname within one edit
    forms = Counter(n for ns, _ in per_ep.values() for n in ns)
    canon = {}
    fulls = sorted([n for n in forms if split(n)[0]], key=lambda n: (-len(n.split()), -forms[n], n))
    groups = []
    for n in fulls:
        f, s = split(n)
        g = next((g for g in groups if key(f) == key(split(g[0])[0])
                  and same_surname(s, split(g[0])[1])), None)
        (g.append(n) if g else groups.append([n]))
    for g in groups:
        head = max(g, key=lambda n: (len(n.split()), forms[n], n))
        for n in g:
            canon[n] = head
            if n != head:
                how[(head, n)].add("variant")

    # 4. bare surnames: merge spelling variants with each other, then attach to
    #    the single full name carrying that surname, unless common or held apart
    bare = sorted([n for n in forms if not split(n)[0]], key=lambda n: (-forms[n], n))
    for n in bare:
        if n in canon:
            continue
        sib = [m for m in bare if m not in canon and m != n and same_surname(m, n)]
        for m in sib:
            canon[m] = n
            how[(n, m)].add("variant")
        canon[n] = n
    heads = sorted(set(canon[n] for n in fulls))
    for n in bare:
        if canon[n] != n:
            continue
        s = key(n)
        cands = [h for h in heads if same_surname(split(h)[1], n)]
        if len(cands) == 1 and s not in COMMON and s not in keep_apart and not n.startswith("The "):
            for m in [m for m in bare if canon[m] == n]:
                canon[m] = cands[0]
                how[(cands[0], m)].add("surname")

    # 5. assemble
    people = {}
    by_episode = {}
    for slug, (names, hon) in per_ep.items():
        ids = []
        for n in names:
            c = canon.get(n, n)
            pid = slugify(c)
            p = people.setdefault(pid, {"id": pid, "name": c, "honorific": "", "episodes": [],
                                        "aliases": {}})
            if hon.get(n) and not p["honorific"]:
                p["honorific"] = hon[n]
            if slug not in p["episodes"]:
                p["episodes"].append(slug)
            if pid not in ids:
                ids.append(pid)
        by_episode[slug] = ids

    for (head, form), srcs in how.items():
        pid = slugify(canon.get(head, head))
        if pid in people and form != people[pid]["name"]:
            a = people[pid]["aliases"].setdefault(form, set())
            a.update(srcs)
    for p in people.values():
        # "Joel S Leitch" -> "Joel S. Leitch" for display
        p["name"] = re.sub(r"\b([A-Z])(?= )", r"\1.", p["name"])
        p["display"] = (p["honorific"] + " " + p["name"]).strip() if p["honorific"] in ("Dr.", "Rev.", "Officer.") \
            else p["name"]
        p["display"] = p["display"].replace("Officer.", "Officer")
        p["episodes"].sort()
        p["aliases"] = [{"form": f, "how": sorted(s)} for f, s in sorted(p["aliases"].items())]
        p["ambiguous"] = (not split(p["name"])[0] and not p["name"].startswith("The ")
                          and sum(1 for h in heads if key(split(h)[1]) == key(p["name"])) > 1)
    return by_episode, people


# ------------------------------------------------------------------- main
def load_inputs() -> dict:
    meta = json.loads(META.read_text())
    eps = {}
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        m = meta.get(d["slug"], {})
        eps[d["slug"]] = {"title": d.get("title", ""), "description": m.get("description", ""),
                          "transcript": d.get("transcript", "")}
    return eps


def apply(by_episode: dict, people: dict) -> int:
    changed = 0
    for f in sorted(EPISODES.glob("*.json")):
        d = json.loads(f.read_text())
        ids = by_episode.get(d["slug"], [])
        sp = [{"id": i, "name": people[i]["display"]} for i in ids]
        disp = " & ".join(s["name"] for s in sp) or "Unknown"
        dirty = d.get("speakers") != sp or d.get("speaker") != disp
        d["speakers"], d["speaker"] = sp, disp
        for p in d.get("pearls", []):
            if p.get("speaker") != disp:
                p["speaker"] = disp
                dirty = True
        if dirty:
            f.write_text(json.dumps(d, indent=2, ensure_ascii=False))
            changed += 1
    return changed


SELFTEST = [
    ("Dr. Jon Calvert on Adnexal Masses 1/22/26", "", ["Jon Calvert"]),
    ("Kelly Dunn, MD on Benzodiazepines", "", ["Kelly Dunn"]),
    ("Dr. Dunn speaks on Depression", "", ["Dunn"]),
    ("Drs Jones & Shannon on Addiction Med Part 1", "", ["Jones", "Shannon"]),
    ("Drs. Greuel and White on Business of Medicine", "", ["Greuel", "White"]),
    ("Pressors - Engstrom", "Dr. Engstrom lectures on pressors.", ["Engstrom"]),
    ("Rheumatology with Dr. Lucashu", "", ["Lucashu"]),
    ("Dr. Lee's Morning Report", "", ["Lee"]),
    ("L’Dogg’s Morning Report", "Dr. Lee presents on the morning report", ["Lee"]),
    ("Tuberculosis - Kyle Jones, DO", "", ["Kyle Jones"]),
    ("Radiation Oncology Primer for Primary Care - Gabriel S. Vidal, MD, FACP", "", ["Gabriel S Vidal"]),
    ("Medical Education as Missions, Part 1 - Mark Crouch MD in PNG", "", ["Mark Crouch"]),
    ("Moral Injury", "Dr. Jim Ritchie presents on moral injury", ["Jim Ritchie"]),
    ("Trauma Informed Care", "Sara Gadd speaks on Human Trafficking", ["Sara Gadd"]),
    ("Security Training", "Security Training with Officer Latif", ["Latif"]),
    ("Dr. Haney- CHF", "", ["Haney"]),
    ("Dr. Place Outpatient Lecture", "", ["Place"]),
    ("Dr. Sutton Lecture #3: Trust and Uncertainty 1/17/26", "", ["Sutton"]),
    ("Dianne Hughes on WIR", "", ["Dianne Hughes"]),
    ("The Condies on Surgery", "", ["The Condies"]),
    ("Dr. Gerald C. Miller on Syphilis Testing", "", ["Gerald C Miller"]),
    ("FMIS August MM", "Dr. Youmans presents followed by Dr. Wheeler and Dr. Jones",
     ["Youmans", "Wheeler", "Jones"]),
    ("AMBOSS Resident Training", "AMBOSS Resident Training", []),
]


def selftest() -> int:
    bad = 0
    for title, desc, want in SELFTEST:
        got = [n for n, _ in parse_names(title, True)]
        if not got or (desc and desc.lower() != title.lower()):
            got += [n for n, _ in parse_names(desc, True) if n not in got]
        ok = got == want
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {title[:50]:50s} -> {got}" + ("" if ok else f"  want {want}"))
    # merging
    eps = {"a": {"title": "Dr. Calvert on X", "description": "", "transcript": ""},
           "b": {"title": "Dr. Jon Calvert on Y", "description": "", "transcript": ""},
           "c": {"title": "Dr. Hermann on Z", "description": "", "transcript": ""},
           "d": {"title": "Dr. Herrmann on Z", "description": "", "transcript": ""},
           "e": {"title": "Dr. Herrmann on W", "description": "", "transcript": ""},
           "f": {"title": "Dr. Miller on Urology", "description": "", "transcript": ""},
           "g": {"title": "Dr. Gerald Miller on Syphilis", "description": "", "transcript": ""},
           "h": {"title": "Dr. Nichols on AKI", "description": "",
                 "transcript": "Welcome. I'm Doctor. Amanda Nichols, and I'm excited."}}
    by, people = resolve(eps, {})
    checks = [(by["a"] == by["b"], "Calvert joins Jon Calvert"),
              (by["c"] == by["d"] == by["e"], "Hermann/Herrmann merge"),
              (by["f"] != by["g"], "common surname Miller is NOT merged"),
              (people[by["h"][0]]["name"] == "Amanda Nichols", "self-intro supplies first name")]
    for ok, what in checks:
        bad += not ok
        print(f"  {'PASS' if ok else 'FAIL'}  {what}")
    print("selftest:", "FAILED" if bad else "ok")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()

    overrides = json.loads(OVERRIDES.read_text()) if OVERRIDES.exists() else {}
    by_episode, people = resolve(load_inputs(), overrides)
    plist = sorted(people.values(), key=lambda p: (-len(p["episodes"]), p["name"]))
    REGISTRY.write_text(json.dumps({
        "_comment": "GENERATED by scripts/speakers.py. Edit data/speaker_overrides.json instead.",
        "people": plist,
        "unattributed": sorted(s for s, ids in by_episode.items() if not ids),
    }, indent=2, ensure_ascii=False))
    changed = apply(by_episode, people)

    n_unk = sum(1 for ids in by_episode.values() if not ids)
    print(f"  {len(people)} people across {len(by_episode)} lectures; "
          f"{n_unk} unattributed; {changed} episode files updated")
    if a.report:
        for p in plist:
            al = "; ".join(f"{x['form']} ({','.join(x['how'])})" for x in p["aliases"])
            print(f"  {len(p['episodes']):3d}  {p['display']:28s}{' [ambiguous]' if p['ambiguous'] else ''}  {al}")
        print("  unattributed:", ", ".join(s for s, ids in by_episode.items() if not ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
