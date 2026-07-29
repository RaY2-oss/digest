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
   (accepted/rejected/short/non_event) so the next run never re-fetches or
   re-judges the same URL.
2. **LLM relevance/topic check** — via `model_rotation.py`, which rotates
   across OpenRouter's free-tier models, then falls back to Groq, then
   Google (Gemini), round-robin, with per-provider cooldown on exhaustion.
3. **`sunday_processor_mmr.py`** (cron, weekly) — builds the digest from the
   week's accepted articles: LLM re-tells/summarizes each one **in English**,
   then MMR
   (Maximal Marginal Relevance) selects a diverse, non-redundant top-N per
   region quota. Article importance blends LexRank over the cosine-similarity
   graph with a *political weight* (`entities.py`): LexRank only measures how
   dense an article's neighbourhood is, so a dozen near-identical local pieces
   ("village pupil aces the national exam") form the same tight clique as a
   genuine national story. The second factor counts how many *distinct
   prominent* actors the article names — persons/organizations GDELT already
   extracted (`V1Persons`/`V1Organizations`), prominent meaning they recur in
   at least `ENTITY_MIN_DF` articles of the week's corpus. No external
   dictionary of political names is downloaded: prominence is derived from our
   own corpus, so it is multilingual and self-updating. Tunable via
   `config.ENTITY_*`; `ENTITY_WEIGHT = 0` restores plain LexRank, and the
   factor stays off by itself while no actor clears the threshold.
4. **Translation to Russian** — the LLM writes English on purpose; the final
   Russian text comes from a local Marian model (`opus-mt-tc-big-en-zle` via
   CTranslate2 int8, `translate_ru.py`), run *after* the dedup pass so
   duplicate detection compares stable English rather than two diverging
   translations. Proper names go through a shared glossary (`glossary.py`):
   `keep` terms are masked before translation and restored in Latin script,
   `ru` terms that leaked through untranslated are replaced with their
   accepted Russian form. This replaced asking the LLM to transliterate names
   itself, which it did inconsistently from call to call.
5. **`word_generator.py`** renders the selection to a `.docx`.
6. **`telegram_sender.py`** delivers it to the configured chat.
7. **`telegram_bot_listener.py`** — long-polling bot exposing `/rundigest`
   (manual trigger) and `/resetlock` (clears a stuck run lock).

`prefilter.py` / `train_prefilter.py` add an optional local classifier,
distilled from the LLM's own past verdicts, that screens out confident junk
before spending an LLM call on it — kicks in automatically once enough
labeled history has accumulated (`PREFILTER_MIN_LABELS`).

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
