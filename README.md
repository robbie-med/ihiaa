# HPI Lecture Corpus

Searchable, cross-linked clinical knowledge base built from **The HPI Lecture Podcast**
(Hope Partnerships International) — 257 residency lectures, 228 hours, back to 2021.

Lectures are public and the speaking physicians consent to publication.
[Original feed ↗](https://hopepartners.podbean.com/)

> **Unverified teaching material.** Transcripts are machine-generated and machine-corrected.
> Every pearl links to the exact second it was said. Verify doses and numbers at the source
> before acting on them.

## What it does

| | |
|---|---|
| **Transcribe** | Deepgram `nova-3-medical`, word-level timestamps + speaker diarization |
| **Repair** | LLM pass fixing ASR drug-name errors; raw text always retained |
| **Enrich** | Abstract, key points, topics, and **pearls** — atomic actionable teaching points |
| **Anchor** | Every pearl carries the verbatim span + timestamp it came from |
| **Cluster** | Near-identical pearls grouped; contradictions surfaced, never auto-resolved |
| **Search** | Keyword over pearls/topics/abstracts instantly; full transcripts on demand |
| **Translate** | ko / fr / de for UI, abstracts and pearls (built, currently disabled) |

## Layout

```
scripts/
  fetch_feed.py      feed -> episodes_meta.json (dedup by <guid>)
  ingest.py          download -> ffmpeg audio -> pipeline (deletes video after)
  backfill.py        parallel backfill of the whole archive
  pipeline.py        transcribe -> correct -> enrich -> embed (all resumable)
  stt_deepgram.py    speech-to-text provider
  ppq.py             chat + embeddings provider
  cluster_pearls.py  pearl similarity clustering + conflict judging
  translate.py       tier-1 translation (off by default)
  build_site.py      emit the static site
  rebuild.sh         every derived artifact in order (free, no keys)
  check_site.py      sanity-check site/ before publishing
data/
  episodes/*.json    source of truth, one file per lecture
  audio/*.m4a        24kbps mono, gitignored (video is never kept)
site/                static output: index.html + data.json + ep/*.json
```

## Run

```bash
python3 scripts/fetch_feed.py            # discover new episodes
python3 scripts/backfill.py --workers 5  # process everything outstanding
./scripts/rebuild.sh                     # site, grades, taxonomy, concept map
python3 scripts/cluster_pearls.py        # group pearls, find contradictions (paid)
python3 scripts/build_site.py            # fold clusters into the site
python3 -m http.server 3907 --bind 127.0.0.1   # then open /site/
```

Keys live in `.ppq_key` and `.deepgram_key` (both gitignored), or the
`PPQ_API_KEY` / `DEEPGRAM_API_KEY` environment variables.

## Automation

Three workflows, all on GitHub's runners. **Nothing runs on a local machine.**

| Workflow | When | What |
|---|---|---|
| `ingest.yml` | daily 09:17 UTC, or by hand | poll feed, process new episodes, rebuild, commit, then call deploy |
| `deploy.yml` | after every ingest, on pushes that touch `site/`, or by hand | publish `site/` to Cloudflare Pages and verify the live site serves it |
| `ci.yml` | every push | lint, grader self-test, check that derived data is reproducible |

Repo secrets:

- `DEEPGRAM_API_KEY`, `PPQ_API_KEY`: transcription and LLM calls
- `CLOUDFLARE_API_TOKEN`: an **API Token** (not the Global API Key) with
  the *Cloudflare Pages: Edit* permission
- `CLOUDFLARE_ACCOUNT_ID`

Repo variables (optional): `PAGES_PROJECT` (default `ihiaa`), and
`RUN_TRANSLATE=1` to re-enable translation.

The Pages project is direct-upload, not connected to this repo, so **a push alone
deploys nothing**. If the site looks stale, open the latest *Deploy site* run: its
first failing step says what is wrong. To publish by hand, run *Deploy site* from
the Actions tab.

An episode whose source can never be transcribed (e.g. a silent audio track) is
marked `"status": "unavailable"` in its JSON and is skipped from then on.

## Design principles

- **JSON in git is the source of truth.** The site and any database are derived
  artifacts, rebuildable from `data/episodes/*.json` plus audio.
- **No unanchored clinical claims.** A pearl whose source span cannot be located is
  dropped rather than shown with a guessed timestamp.
