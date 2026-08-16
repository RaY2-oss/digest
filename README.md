# digest

Daily/weekly news digest bot: collects articles from GDELT GKG, filters them
by topic+country relevance, scores and summarizes them with an LLM, picks a
diverse set per region quota, and delivers the result as a `.docx` over
Telegram.

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
   carrying all its reprints. Then the top-N of each region quota is taken by
   importance, and the LLM re-tells each winner **in Russian** — from *all* its
   reprints at once (see steps 4–5).

   **MMR is gone as of 16.08.2026** — the module keeps its name only because
   cron, the bot and six tests import it. The diversity term existed to keep
   two retellings of one event out of an issue, and that job is already done
   twice downstream: `_is_semantic_duplicate` on the article embedding (0.90)
   in `_process_slot`, and `SUMMARY_DUP_COSINE` on the finished retelling.
   Measured on the frozen basket (`bench.py mmr`, removed with the branch):
   after those guards, λ=0.5 and λ=1.0 differ by two items out of twenty, mean
   importance 0.7615 against 0.7622, and **both leave zero pairs above cosine
   0.90**. λ was not buying diversity — the guards had already delivered it —
   it was only reshuffling neighbours inside the noise band. Raising λ to 0.8
   would have been the same as deleting the branch: at 0.8 the pick already
   equals plain importance ranking exactly.

   Story importance blends six factors, none of which costs an API call
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
   - **novelty** — the only factor that looks *outside* the week's window
     (`issue_archive.py`). The other five all measure one family, "is this our
     subject and how many wrote about it", so a follow-up is indistinguishable
     from a first report: 14% of items repeat an item of an earlier issue at
     cosine ≥ 0.90. Past issues are parsed out of `output/*.docx`, embedded
     once and cached; a story's novelty is 1 below `NOVELTY_FLOOR` and falls
     linearly to 0 at an exact match.

     It is a tail penalty, not a slope, and that was the measurement's main
     lesson. Plain `1 − cosine` moved every story at once and changed no
     selection at all up to weight 0.10 (`bench.py sweep`). And the floor
     cannot be set at the distribution tail either: one repeated event had
     *eight* versions spread over cosine 0.888–0.963, so a floor of 0.90
     demoted seven of them and let the eighth — the same story in Armenian —
     take the freed slot. The floor has to sit below the weakest member of a
     repeat cluster or the factor shuffles versions instead of removing the
     repeat. At 0.88 (54 of 1172 articles) it swaps one item of twenty:
     out goes the Firebird AI-factory story already printed on 09.08, in comes
     Georgia's free-textbook programme. That single swap is the honest size of
     the effect on the one basket that still exists.
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

## Relevance bench (`bench.py`)

Ranking changes are compared against a frozen set, not against memory.

```
venv/bin/python bench.py build   # freeze the set (rewrites bench/)
venv/bin/python bench.py run     # metrics -> bench/BASELINE.md
venv/bin/python bench.py labels  # 198 hard cases for the owner to grade
```

The set is **one basket, six issues, no 26-week replay**. `cleanup_old_articles.py`
drops articles older than seven days, so only the current basket ever exists;
the verdict journal keeps four weeks but stores no text or scale. `output/` holds
30 `.docx` but only 6 are issues — `sunday_processor_mmr` is a Sunday cron job
and the rest are test runs, which share six days of window out of seven and
therefore repeat each other by construction. See `_weekly()`.

Baseline of 16.08.2026, no LLM calls:

| | TR | CA | SC |
| --- | --- | --- | --- |
| articles → stories | 879 → 364 | 157 → 129 | 134 → 91 |
| mean importance of the quota | 0.9119 | 0.6750 | 0.6784 |
| gap between last taken and first dropped | 0.0078 | 0.0156 | 0.0053 |
| stories sharing the top scale | 101 | 40 | 31 |
Issue-wide: `scale_top_share` 0.373, mean importance 0.7589 (0.8034 without
quotas, so quotas cost 0.0445), `repeat_share` 0.140 — that many items of an
issue repeat an item of an earlier one at cosine ≥ 0.90.

Adding the **novelty** factor (weight 0.20) moves `stale` in SC from 0.868 to
0.857 by swapping one item. Mean importance is *not* comparable across
different weight sets — `blend` renormalizes, so adding any factor rescales the
number.

### Soft quotas: measured and rejected

Per-bucket importance is **not** on a common scale — `topic` is a percentile
rank within the bucket and `coverage` is normalized by the bucket's own
maximum, which is 31 outlets in TR and 3 in CA. Any comparison of "0.93 in TR"
with "0.72 in CA" is a comparison of two different scales. `bench.py quotas`
therefore scores all 584 stories of the week at once:

| policy | mean importance | composition | worst item |
| --- | --- | --- | --- |
| hard 7/7/6 (current) | 0.7736 | TR 7 · CA 7 · SC 6 | 0.6535 |
| soft 4–10 / 3–9 / 3–8 | 0.7967 | TR 10 · CA 6 · SC 4 | 0.6645 |
| soft + floor 0.70 | 0.8270 | TR 10 · CA 3 · SC 3 — **16 items, not 20** | 0.6885 |
| no quota at all | 0.8266 | TR 19 · CA 0 · SC 1 | 0.7448 |

The soft quota does raise the number, and the number is the wrong thing to
raise. Look at what the three extra Turkish slots buy: two more items about the
student amnesty, which already holds three slots, one of them with a
mojibake title (`ÃÄrenci affÄ±nda...`). What they cost: Kazakhstan's NEET
share falling 6.7 points, an Azerbaijan–Ukraine health agreement, and
Georgia's national free-textbook programme. Rejected — kept at hard 7/7/6.

The floor variant silently shortens the issue to 16 items, which is a different
decision from "raise relevance" and belongs to the owner, not to a weight.

`ndcg@7` stays empty until `bench/labels.tsv` exists. It is deliberately not
faked: every automatic stand-in for relevance here is built out of the same
factors being judged and would only measure itself.

### Language labels are cosmetic

`articles.language` is written by `detect_text_language` and read by exactly
one thing: a `lang=xx: N` line in the collector log. No prompt, glossary,
translation, prefilter or importance factor touches it — `load_week_articles`
does not even select the column.

That was worth checking, because 46 Armenian (`1in.am`, `panorama.am`) and
Georgian (`sputnik-georgia.com`) pages carry `language='et'`. It is not a bug
in our code: `langdetect` ships 55 profiles and neither `hy` nor `ka` is among
them, so it has to put those scripts somewhere. Left as is — a label nobody
reads cannot corrupt anything. If it ever gains a reader, get a detector that
knows those scripts first (both have their own alphabet, a Unicode range check
is enough) and only then rely on it.

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
