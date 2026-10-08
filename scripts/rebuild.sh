#!/usr/bin/env bash
# Rebuild every derived artifact from data/episodes/*.json, in dependency order.
# No API keys needed and no money spent; safe to run any time, locally or in CI.
#
#   build_site      topics -> site/data.json (the grader reads these as concepts)
#   score_pearls    deterministic grades written back into data/episodes
#   build_site      again, so the new grades decide which pearls are shown
#   build_taxonomy  topic hierarchy (needs data/ref/mesh.bin)
#   build_graph     concept map -> site/graph.json
#
# Pearl clustering (scripts/cluster_pearls.py) is separate: it calls a paid API.
# Re-run build_site.py after it to fold fresh clusters into data.json.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 scripts/build_site.py
python3 scripts/score_pearls.py --apply
python3 scripts/build_site.py
python3 scripts/build_taxonomy.py
python3 scripts/build_graph.py
