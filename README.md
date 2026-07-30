# digest

Daily/weekly news digest bot: collects articles from GDELT GKG, filters them
by topic+country relevance, scores and summarizes them with an LLM, picks a
diverse set with MMR, and delivers the result as a `.docx` over Telegram.

Built for a narrow editorial focus (science/tech/higher-education coverage
across Turkey, Central Asia and the South Caucasus) — see `config.py` for the
actual topic/country filters, which are the main thing to change if you want
to repurpose this for a different beat.

## Pipeline

1. **`daily_collector.py`** (cron, every 6h) — scans 15-minute GDELT GKG dumps,
   keeps rows where a target theme and a target country appear near each
   other in the same article (`gkg_filter.py`, symbolic-offset matching, not
   just "both mentioned somewhere"). Fetches and extracts article text
   (`trafilatura`), embeds it (`intfloat/multilingual-e5-small`), and writes
   it to SQLite. `seen_store.py` journals every URL's final verdict
   (accepted/rejected/short/non_event/prefiltered) so the next run never
   re-fetches or re-judges the same URL. `prefiltered` is deliberately kept
   apart from `rejected` and stored *without* its embedding: that verdict came
   from the local classifier below, and feeding it back as training data would
   let the model confirm its own decisions run after run.
2. **LLM relevance/topic check** — via `model_rotation.py`, which rotates
   across OpenRouter's free-tier models, then falls back to Groq, then
   Google (Gemini), round-robin, with per-provider cooldown on exhaustion.
3. **`sunday_processor_mmr.py`** (cron, weekly) — builds the digest from the
   week's accepted articles. First the week's articles are collapsed into
   **stories**: near-duplicates (cosine ≥ `STORY_COSINE`) become one candidate
   carrying all its reprints. Then MMR (Maximal Marginal Relevance) selects a
   diverse, non-redundant top-N per region quota, and the LLM re-tells each
   winner **in English** — from *all* its reprints at once (see step 4).

   Story importance blends four factors, none of which costs an API call
   (`config.IMPORTANCE_W_*`; a weight of 0 switches its factor off):

   - **lexrank** over the cosine-similarity graph — thematic centrality.
   - **coverage** — how many *distinct outlets* filed the story. LexRank alone
     only measures how dense a neighbourhood is, so a dozen near-identical
     local pieces ("village pupil aces the national exam") form the same tight
     clique as a genuine national story; that is exactly how a note about one
     school in Kadıköy scored 0.966. Counting outlets separates them cleanly —
     over the week of 27.07 the national stories (student-amnesty bill, the YKS
     guide, the lunar programme) ran in 13–29 outlets and the school note in
     one. Syndication networks that republish one wire item under six domains
     are collapsed to a single outlet by `importance.publisher()`, so a content
     farm cannot buy coverage.
   - **entity** — how many *distinct prominent* actors the story names:
     persons/organizations GDELT already extracted (`V1Persons`/
     `V1Organizations`), prominent meaning they recur in at least
     `ENTITY_MIN_DF` articles of the week. No external dictionary of political
     names is downloaded — prominence is derived from our own corpus, so it is
     multilingual and self-updating. The factor stays off by itself while no
     actor clears the threshold.
   - **topic** — the local classifier's probability (see `prefilter.py` below),
     reused here as a continuous "is this our subject at all" measure. Without
     it the thin CA/SC buckets floated chess politics, party-building reports
     and a murder case to the top: those were let through by the LLM judge, and
     coverage only amplified them. The factor stays off while no artifact is
     trained.
4. **Retelling a story, not an article** — the prompt carries up to
   `RETELL_MAX_VERSIONS` reprints of the same story under a shared character
   budget (`RETELL_CHARS_*`), and every URL that went into the prompt is
   printed under the summary in the `.docx`. Different newsrooms keep different
   details — figures, names, quotes — and previously all but one reprint were
   simply discarded (49 of them in the run of 26.07). The caps are the point:
   past them the free-tier models start truncating JSON and inventing links
   between the versions.
5. **Translation to Russian** — the LLM writes English on purpose; the final
   Russian text comes from a local Marian model (`opus-mt-tc-big-en-zle` via
   CTranslate2 int8, `translate_ru.py`), run *after* the dedup pass so
   duplicate detection compares stable English rather than two diverging
   translations. Proper names go through a shared glossary (`glossary.py`):
   `keep` terms are masked before translation and restored in Latin script,
   `ru` terms that leaked through untranslated are replaced with their
   accepted Russian form. This replaced asking the LLM to transliterate names
   itself, which it did inconsistently from call to call.
6. **`word_generator.py`** renders the selection to a `.docx`.
7. **`telegram_sender.py`** delivers it to the configured chat.
8. **`telegram_bot_listener.py`** — long-polling bot exposing `/rundigest`
   (manual trigger) and `/resetlock` (clears a stuck run lock).

`prefilter.py` / `train_prefilter.py` are a local classifier distilled from the
LLM's own past verdicts. It does two jobs off one training run: it screens out
confident junk before an LLM call is spent on it (measured on 19 450 labels:
47% of the junk cut at 97% recall, LLM volume down to 63% of what it was), and
it supplies the *topic* factor of story importance above. `train_prefilter.py`
runs weekly from cron and refuses to overwrite the artifact while the data is
thin (`PREFILTER_MIN_LABELS`, `PREFILTER_MIN_MINORITY`) or the measured gain
does not pay for the stage (`PREFILTER_MIN_GAIN`). While no artifact exists,
both users of it stay switched off.

`cleanup_old_articles.py` and `week_swap_run.sh` handle retention and
zero-downtime rebuilds of the database.

`embedder.py` talks to onnxruntime directly instead of going through
`sentence-transformers`. The CPU this runs on has no AVX, and every path that
reaches `torch` dies by `SIGILL` — see the same note in `gdelt_rss`'s README.

## Shared with `gdelt_rss`

Two modules are not copies but symlinks into the sibling project, which owns
them:

| link | target |
| --- | --- |
| `glossary.py` | `../gdelt_rss/glossary.py` |
| `translate_ru.py` | `../gdelt_rss/translate_ru.py` |

So **clone both repos side by side** — `git clone …/gdelt_rss` next to this
one — or the links dangle and step 4 of the pipeline fails. Both modules are
env-configurable and hold no project-specific state; the machine-level
resources they need live outside either repo:

- `/opt/translate/ct2/tcbig-en-ru` — the CTranslate2-converted model
  (`CT2_DIR`).
- `/opt/translate/models` — Hugging Face cache for the tokenizer
  (`TRANSLATE_HF_HOME`).
- `/opt/translate/glossary*.tsv` — the proper-name dictionary, tab-separated
  `source⇥replacement⇥mode` where mode is `keep` or `ru` (`GLOSSARY_DIR`).

## Setup

1. `python -m venv venv && venv/bin/pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and fill in:
   - `OPENROUTER_API_KEY` — https://openrouter.ai/keys (free `:free` models
     work without billing).
   - `GROQ_API_KEY` (optional fallback) — https://console.groq.com/keys
   - `GOOGLE_API_KEY` (optional fallback) — https://aistudio.google.com/apikey
   - `TELEGRAM_BOT_TOKEN` — via `@BotFather`.
   - `TELEGRAM_CHAT_ID` — chat/channel to broadcast the weekly digest to.
   - `ALLOWED_TG_USER_IDS` (optional) — comma-separated user IDs allowed to
     run `/rundigest` manually; leave empty to allow everyone.
3. Edit `config.py`: `QUERIES_GKG` (your own topic/country filters).
4. `venv/bin/python init_db.py`
5. Wire up cron: `daily_collector.py` every 6h, `sunday_processor_mmr.py`
   weekly, `cleanup_old_articles.py` daily, `telegram_bot_listener.py` as a
   long-running service.

## Notes

- `PROXIES` in `config.py` is `None` by default (direct connection); set it
  if GDELT/OpenRouter need to go through a SOCKS proxy from your host.
- No hardcoded model list: OpenRouter's `:free` slugs rotate over time, so
  the pool is fetched live and filtered rather than pinned in code.
