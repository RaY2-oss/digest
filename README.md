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
   week's accepted articles: LLM re-tells/summarizes each one, then MMR
   (Maximal Marginal Relevance) selects a diverse, non-redundant top-N per
   region quota.
4. **`word_generator.py`** renders the selection to a `.docx`.
5. **`telegram_sender.py`** delivers it to the configured chat.
6. **`telegram_bot_listener.py`** — long-polling bot exposing `/rundigest`
   (manual trigger) and `/resetlock` (clears a stuck run lock).

`prefilter.py` / `train_prefilter.py` add an optional local classifier,
distilled from the LLM's own past verdicts, that screens out confident junk
before spending an LLM call on it — kicks in automatically once enough
labeled history has accumulated (`PREFILTER_MIN_LABELS`).

`cleanup_old_articles.py` and `week_swap_run.sh` handle retention and
zero-downtime rebuilds of the database.

## Setup

1. `python -m venv venv && venv/bin/pip install -r requirements.txt`
   (no `requirements.txt` yet — see packages imported across the `.py`
   files: `requests`, `pandas`, `numpy`, `scikit-learn`, `sentence-transformers`,
   `trafilatura`, `htmldate`, `langdetect`, `python-docx`, `joblib`).
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
