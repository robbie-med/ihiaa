"""Build the concept map: nodes, branches, co-occurrence edges, domain colours.

Layout is computed HERE, at build time, not in the browser. A force simulation over
400+ nodes churns CPU on every page load and settles differently each time; solving it
once and shipping coordinates makes the map instant, identical for every visitor, and
diffable between builds.

The simulation is a plain seeded force-directed layout -- repulsion between all nodes,
springs along edges, mild gravity. No RNG that varies run to run: initial positions are
placed deterministically on a spiral, so the same corpus always yields the same map.

Two kinds of link:
  * BRANCH  -- parent/child from the taxonomy ("Gestational Hypertension" under
               "Hypertension"). Drawn stronger and shorter; these are the tree.
  * EDGE    -- co-occurrence, two concepts taught in the same lecture. Weighted by how
               many lectures they share, and only kept above a threshold so the map
               does not turn into a hairball.

Only concepts appearing in >= MIN_EPISODES lectures are included. Below that the corpus
is a dust cloud of single-mention topics (76% of all labels) that add no structure.
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SITE = BASE / "site" / "data.json"
TAX = BASE / "data" / "taxonomy.json"
OUT = BASE / "site" / "graph.json"

MIN_EPISODES = 3        # concepts below this are noise, not structure
MIN_SHARED = 2          # an edge needs this many shared lectures
W, H = 1600, 1100
ITERS = 320

DOMAIN_COLORS = {
    "Cardiovascular": "#e11d48", "Infectious": "#16a34a", "Psychiatry": "#8b5cf6",
    "Endocrine": "#f59e0b", "Musculoskeletal": "#0891b2", "Obstetrics": "#ec4899",
    "Gynecology": "#db2777", "Neurology": "#6366f1", "Gastrointestinal": "#ca8a04",
    "Pulmonary": "#0ea5e9", "Renal": "#14b8a6", "Renal/Urologic": "#14b8a6",
    "Dermatology": "#f97316", "Hematology/Oncology": "#7c3aed",
    "Pediatrics": "#22c55e", "Geriatrics": "#64748b", "Preventive": "#84cc16",
    "Procedures": "#06b6d4", "Global Health": "#a16207", "Practice": "#94a3b8",
    # domains introduced by the MeSH tree
    "Pharmacology": "#f43f5e", "Diagnostics": "#38bdf8", "Health Care": "#94a3b8",
    "Signs/Symptoms": "#a3a3a3", "Physiology": "#65a30d", "Anatomy": "#a8a29e",
    "Organisms": "#4d7c0f", "Toxicology": "#b45309", "Dental/ENT": "#c026d3",
    "Ophthalmology": "#0d9488", "Immunology": "#2563eb",
    "General": "#6b7280",
}


def layout(nodes: list, links: list) -> None:
    """Seeded force-directed layout. Deterministic: spiral init, no randomness."""
    n = len(nodes)
    idx = {nd["id"]: i for i, nd in enumerate(nodes)}
    # golden-angle spiral, so the start state is spread but reproducible
    ga = math.pi * (3 - math.sqrt(5))
    px = [0.0] * n
    py = [0.0] * n
    for i in range(n):
        r = 380 * math.sqrt((i + 0.5) / n)
        px[i] = W / 2 + r * math.cos(i * ga)
        py[i] = H / 2 + r * math.sin(i * ga)

    adj = defaultdict(list)
    for l in links:
        a, b = idx[l["s"]], idx[l["t"]]
        adj[a].append((b, l["w"]))
        adj[b].append((a, l["w"]))

    k = math.sqrt((W * H) / max(n, 1)) * 0.6
    for it in range(ITERS):
        cool = 1.0 - it / ITERS
        dx = [0.0] * n
        dy = [0.0] * n
        # repulsion (O(n^2); n is a few hundred, so this is fine)
        for i in range(n):
            for j in range(i + 1, n):
                ux, uy = px[i] - px[j], py[i] - py[j]
                d2 = ux * ux + uy * uy + 0.01
                if d2 > 640000:            # ignore far pairs, big speedup
                    continue
                f = (k * k) / d2
                fx, fy = ux * f, uy * f
                dx[i] += fx; dy[i] += fy
                dx[j] -= fx; dy[j] -= fy
        # springs
        for i in range(n):
            for j, w in adj[i]:
                ux, uy = px[j] - px[i], py[j] - py[i]
                d = math.sqrt(ux * ux + uy * uy) + 0.01
                f = (d * d) / (k * 14) * min(w, 4)
                dx[i] += ux / d * f
                dy[i] += uy / d * f
        # gravity + integrate
        for i in range(n):
            dx[i] += (W / 2 - px[i]) * 0.006
            dy[i] += (H / 2 - py[i]) * 0.006
            d = math.sqrt(dx[i] ** 2 + dy[i] ** 2) + 1e-9
            step = min(d, 16 * cool)
            px[i] += dx[i] / d * step
            py[i] += dy[i] / d * step
            px[i] = max(30, min(W - 30, px[i]))
            py[i] = max(30, min(H - 30, py[i]))

    for i, nd in enumerate(nodes):
        nd["x"] = round(px[i], 1)
        nd["y"] = round(py[i], 1)


def main() -> int:
    if not (SITE.exists() and TAX.exists()):
        print("run build_site.py and build_taxonomy.py first")
        return 1
    data = json.loads(SITE.read_text())
    tax = json.loads(TAX.read_text())["nodes"]

    keep = {k: v for k, v in tax.items() if v["n"] >= MIN_EPISODES}
    print(f"  {len(keep)} concepts in >={MIN_EPISODES} lectures")

    # MeSH gives a real parent chain; prefer it over the suffix heuristic. A branch
    # is drawn only when the parent concept is itself in the kept set.
    by_ui = {}
    for k, v in keep.items():
        if v.get("mesh_ui"):
            by_ui.setdefault(v["mesh_ui"], k)
    heading_to_key = {}
    for k, v in keep.items():
        heading_to_key.setdefault(v["label"].lower(), k)

    nodes = []
    for k, v in sorted(keep.items()):
        par = None
        mp = (v.get("mesh_parent") or "").lower()
        if mp and mp in heading_to_key and heading_to_key[mp] != k:
            par = heading_to_key[mp]
        elif v["parent"] in keep:
            par = v["parent"]
        nodes.append({"id": k, "label": v["label"], "slug": v["slug"],
                      "domain": v["domain"],
                      "color": DOMAIN_COLORS.get(v["domain"], "#6b7280"),
                      "mesh": v.get("mesh_ui"), "n": v["n"], "parent": par})

    # branches: parent/child inside the kept set
    links = [{"s": nd["parent"], "t": nd["id"], "w": 4, "kind": "branch"}
             for nd in nodes if nd["parent"]]
    seen = {(l["s"], l["t"]) for l in links} | {(l["t"], l["s"]) for l in links}

    # edges: concepts sharing lectures
    ep2t = defaultdict(list)
    for k, v in keep.items():
        for e in v["episodes"]:
            ep2t[e].append(k)
    pair = Counter()
    for ts in ep2t.values():
        ts = sorted(set(ts))
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                pair[(ts[i], ts[j])] += 1
    for (a, b), c in pair.items():
        if c >= MIN_SHARED and (a, b) not in seen:
            links.append({"s": a, "t": b, "w": c, "kind": "edge"})

    print(f"  {sum(1 for l in links if l['kind']=='branch')} branches, "
          f"{sum(1 for l in links if l['kind']=='edge')} co-occurrence edges")
    print("  computing layout ...", flush=True)
    layout(nodes, links)

    dom = Counter(nd["domain"] for nd in nodes)
    OUT.write_text(json.dumps(
        {"nodes": nodes, "links": links, "w": W, "h": H,
         "colors": DOMAIN_COLORS,
         "domains": sorted(dom.items(), key=lambda kv: -kv[1]),
         "min_episodes": MIN_EPISODES},
        separators=(",", ":")))
    print(f"  wrote {OUT.name} ({OUT.stat().st_size/1024:.0f} KB)")
    for d, c in sorted(dom.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {d:22s} {c:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
