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

   Pages that returned no usable text get one more chance at the end of the
   run — `short` is final on the first attempt, so it is now or never. The
   chance is a ladder, cheapest rung first. Measured on 100 such URLs
   (2026-08-08), counting only genuine articles:

   | rung | rescued | cost |
   |---|---|---|
   | plain `requests` retry | 26 | one request |
   | + second `trafilatura` pass with `favor_recall` | 36 | none, same HTML |
   | `articleBody` from JSON-LD | 11 (**+0**) | — *not wired in* |
   | `/amp` URL variants | 29 (**+2**) | 3 requests each — *not wired in* |
   | Web Archive snapshot (`rescue_from_archive`) | 39 (**+24**) | ~2 s |
   | camoufox (`rescue_with_browser`) | 32 | ~15 s |
   | **all wired rungs together** | **62** | |

   So the recall retry is free and lives inside `extract_page` itself; the
   archive (`archive_fetch.py`, shared with gdelt_rss) runs before the browser
   and takes roughly twice as many pages for a fifth of the time; the browser
   is last and spends its `config.CAMOUFOX_MAX` budget only on what the
   archive did not have. Rationale for each rejected rung — and for querying
   the availability API rather than CDX — is in `archive_fetch.py`'s header.

   `ANTIBOT_PATTERNS` in `extract_page` keeps anti-bot interstitials out of
   the corpus — they are 300–500 characters of prose, i.e. comfortably above
   `MIN_TEXT_LENGTH`, and shields answer a browser more readily than a script.
2. **LLM relevance/topic check** — via `model_rotation.py`, which rotates
   across OpenRouter's free-tier models, then falls back to Groq, then
   Google (Gemini), round-robin, with per-provider cooldown on exhaustion.
3. **`sunday_processor_mmr.py`** (cron, weekly) — builds the digest from the
   week's accepted articles. First the week's articles are collapsed into
   **stories**: near-duplicates (cosine ≥ `STORY_COSINE`) become one candidate
   carrying all its reprints. Then MMR (Maximal Marginal Relevance) selects a
   diverse, non-redundant top-N per region quota, and the LLM re-tells each
   winner **in Russian** — from *all* its reprints at once (see steps 4–5).

   Story importance blends five factors, none of which costs an API call
   (`config.IMPORTANCE_W_*`; a weight of 0 switches its factor off). One outlet
   gets one vote in every one of them: `importance.publisher()` collapses both
   domains and wire-syndication networks, `lexrank` zeroes intra-domain edges,
   `entity` takes the max over versions and `scale` averages per publisher
   first. Without that last step a paper that ran the same piece twice voted
   twice — over the week of 09.08 that shifted the scale of 26 stories out of
   725, by 0.062 on average and up to 0.225, which at weight 0.40 is worth a
   place or two in the selection.

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
   - **scale** — how far the event reaches, graded by the LLM judge that
     already reads the article (`"TR3"` — bucket plus one digit, no extra
     call): one institution, a city, a country, beyond it. Averaged over the
     story's versions rather than maxed, so a single outlet calling its own
     story national cannot carry it alone.
   - **topic** — the local classifier's probability (see `prefilter.py` below),
     reused here as a continuous "is this our subject at all" measure. Without
     it the thin CA/SC buckets floated chess politics, party-building reports
     and a murder case to the top: those were let through by the LLM judge, and
     coverage only amplified them. The factor stays off while no artifact is
     trained.
4. **Retelling a story, not an article** — the prompt carries up to
   `RETELL_MAX_VERSIONS` reprints of the same story under a shared character
   budget (`RETELL_CHARS_*`), and every URL that went into the prompt is
   printed under the summary in the `.docx`, all of them on one `URL:` line
   separated by `; ` — one line each took more room under a story than the
   summary itself. Different newsrooms keep different
   details — figures, names, quotes — and previously all but one reprint were
   simply discarded (49 of them in the run of 26.07). The caps are the point:
   past them the free-tier models start truncating JSON and inventing links
   between the versions. What goes into that budget is a *selection*, not the
   first N characters of each version (`retell_select.py`): reprints share
   their lead verbatim, so truncation spent the budget printing one lead four
   times while the distinguishing details sat below the cut. Selection scores
   sentences by tf-idf centrality over the story's own sentences, goes round
   robin so no outlet takes a second sentence before every outlet has taken a
   first, and rejects a sentence whose 4-word shingles are already 28% covered.
   A fragment with no sentence-final punctuation is not a sentence — that is
   what a quota table row and a subhead look like. Measured on 244
   multi-version stories: 313 distinct words and 515 distinct trigrams per
   prompt against 265 and 434 for truncation, in a 22% *shorter* prompt, and
   3.25 outlets represented instead of 2.88.

   The same call also names the **date of the event** (`event_date.py`). It is
   nowhere else: GKG carries the dump timestamp and the page markup the
   publication time, so a summit held on 29 July under a story first filed on
   2 August was dated 02.08. Only the article text says otherwise, so something
   has to *read* it — and a hallucinated date in a heading looks exactly as
   convincing as a real one. Hence a guard rather than trust: the date must
   fall in `[today − 7 days, first publication]` (later than publication is a
   deadline, not an event) and must be spelled out **literally** in the text
   the model was given, in any of the nine languages whose month names
   `dateparser`'s locales supply. Anything else falls back to the publication
   date, and every printed date is clamped to the one-week window. Doing the
   reading with a regex instead was measured and dropped: on the week of
   09.08 it found a date for 22–35% of stories and changed 4–9% of them, about
   half of those wrongly — «her geçen gün» ("with every passing day") became
   yesterday and «1 Ağustos itibarıyla» ("as of 1 August") became the event
   day. Form does not separate those from real dates; meaning does.
5. **Russian straight out of the model** — the summary is written in Russian
   in the same call, and the old leg through a local Marian model was dropped
   (07.08.2026): two legs meant the translator's own breakage on top of the
   model's. What the leg used to guarantee is now a prompt rule plus a guard
   that measures it. Proper names ride into the prompt as a shared glossary
   (`glossary.py`, symlinked from `gdelt_rss`) and the guards live in
   `_validate_summary_fast`: mixed scripts inside a word, a third alphabet, a
   non-Russian Cyrillic language, a summary cut off mid-word — each one is
   there because that exact thing reached a reader once.

   The newest of them (`translit_guard.py`) catches a foreign *common* noun
   spelled out in Cyrillic instead of being translated — «В Турции увеличен
   контэнджан стипендиальной программы» (`kontenjan` is Turkish for a quota of
   places). Nothing else notices it: the JSON is valid, the text is Russian,
   the Cyrillic share is fine. The signal is a conjunction, and both halves are
   needed: the word does not occur in Russian text at all (`wordfreq`) **and**
   it matches a lowercase Latin word of the source article. The first half
   alone flags 18% of shipped items (`межсекторального`, `астронавтическом` —
   ordinary Russian compounds missing from frequency lists), the second alone
   would flag «университет» next to Turkish `üniversite`. Together, on 487
   shipped items: two flagged, both the real thing.
6. **`word_generator.py`** renders the selection to a `.docx`.
7. **`telegram_sender.py`** delivers it to the configured chat.
8. **`telegram_bot_listener.py`** — long-polling bot exposing `/rundigest`
   (manual trigger), `/digests` and `/last` (past digests, newest first),
   `/status` and `/resetlock` (clears a stuck run lock). A run takes minutes,
   so `/rundigest` reports progress by **editing its own status message**
   rather than posting a new one per stage: the processor prints
   `DIGEST_STAGE=<text>` to stdout (same convention as the existing
   `DIGEST_ARTICLES=n/m`) and the bot streams that from a `Popen` pipe, at most
   one edit per `EDIT_EVERY` seconds because Telegram rate-limits a chat to
   roughly one message a second. Every timestamp shown to a reader is Moscow
   time even though the host runs on UTC — several runs can land on one date,
   and without the clock they are indistinguishable.

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
5. Wire up cron: `daily_collector.py` every 6h, and `sunday_processor_mmr.py`
   chained onto the last Sunday collection in one line — `0 18 * * 0 collector;
   processor` on a UTC host. Chained rather than given an hour of its own: a
   collection takes 20–25 minutes and sometimes longer, so any guessed hour
   either cuts the corpus mid-collection or waits for nothing. The digest goes
   out around 21:30 Moscow with the freshest week the database has. Also
   `cleanup_old_articles.py` daily and `telegram_bot_listener.py` as a
   long-running service.

## Notes

- `PROXIES` in `config.py` is `None` by default (direct connection); set it
  if GDELT/OpenRouter need to go through a SOCKS proxy from your host.
- No hardcoded model list: OpenRouter's `:free` slugs rotate over time, so
  the pool is fetched live and filtered rather than pinned in code.
