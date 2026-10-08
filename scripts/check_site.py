"""Sanity-check the built site before it is published.

Catches the failures that would otherwise only show up as a blank page in a
browser: unparseable JSON, an episode listed in data.json with no ep/<slug>.json
behind it, an empty corpus, or a syntax error in the inline app script.

    python3 scripts/check_site.py

Exits non-zero on any problem. Runs in CI and right before every deploy.
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SITE = BASE / "site"


def main() -> int:
    errors = []

    def load(name: str):
        try:
            return json.loads((SITE / name).read_text())
        except (OSError, ValueError) as e:
            errors.append(f"{name}: {e}")
            return None

    data = load("data.json")
    load("graph.json")

    if data:
        eps = data.get("episodes", [])
        if not eps:
            errors.append("data.json: no episodes")
        if data.get("stats", {}).get("n_episodes") != len(eps):
            errors.append("data.json: stats.n_episodes disagrees with episode list")
        missing = [e["slug"] for e in eps if not (SITE / "ep" / f"{e['slug']}.json").exists()]
        if missing:
            errors.append(f"ep/: {len(missing)} episode file(s) missing, e.g. {missing[:3]}")
        print(f"  data.json: {len(eps)} episodes, {len(data.get('pearls', []))} pearls")

    html = (SITE / "index.html").read_text()
    js = "\n".join(re.findall(r"<script(?![^>]*\bsrc=)[^>]*>(.*?)</script>", html, re.S))
    if not js.strip():
        errors.append("index.html: no inline script found")
    elif shutil.which("node"):
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as fh:
            fh.write(js)
        r = subprocess.run(["node", "--check", fh.name], capture_output=True, text=True)
        Path(fh.name).unlink(missing_ok=True)
        if r.returncode:
            errors.append("index.html: inline script has a syntax error\n" + r.stderr[-800:])
        else:
            print("  index.html: inline script parses")
    else:
        print("  index.html: node not installed, script syntax not checked")

    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    print("site check:", "FAILED" if errors else "ok")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
