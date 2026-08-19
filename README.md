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
data/
  episodes/*.json    source of truth, one file per lecture
  audio/*.m4a        24kbps mono, gitignored (video is never kept)
site/                static output: index.html + data.json + ep/*.json
```

## Run

```bash
python3 scripts/fetch_feed.py            # discover new episodes
python3 scripts/backfill.py --workers 5  # process everything outstanding
python3 scripts/cluster_pearls.py        # group pearls, find contradictions
python3 scripts/build_site.py            # rebuild the site
python3 -m http.server 3907 --bind 127.0.0.1   # then open /site/
```

Keys live in `.ppq_key` and `.deepgram_key` (both gitignored), or the
`PPQ_API_KEY` / `DEEPGRAM_API_KEY` environment variables.

## Automation

`.github/workflows/ingest.yml` polls the feed weekly and processes new episodes on
GitHub's runners — **nothing runs on a local machine**. Set repo secrets
`DEEPGRAM_API_KEY` and `PPQ_API_KEY`. Set the repo variable `RUN_TRANSLATE=1` to
re-enable translation.

## Design principles

- **JSON in git is the source of truth.** The site and any database are derived
  artifacts, rebuildable from `data/episodes/*.json` plus audio.
- **No unanchored clinical claims.** A pearl whose source span cannot be located is
  dropped rather than shown with a guessed timestamp.
