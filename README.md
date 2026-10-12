# doomscroll

> reverse-engineering the loop. why you stay, what keeps you watching, and how to keep you hooked longer.

this is an eda-first project on short-video behaviour. the goal is simple: understand the algorithm loop from real watch, click, like, follow and share behaviour, then use that understanding to build a recommender that keeps people on the platform as long as possible. recommender comes next. right now this repo is all about the data, the behaviour, and the graphs.

more eda graphs to come. lets cook.

---

## tldr analytics strip

- raw: ~19.66m events — ~14.61m feed + ~5.06m search — 25,877 users — ~6.89m items — 19.1 days — 2023-05-22 to 2023-06-10
- clean: ~19.62m events kept — 99.75% of raw — only exact duplicates dropped — 48,257 rows
- modelled: ~13.07m events after k-core — 66.6% events kept — 25,865 users kept — ~100% — ~1.30m items kept — 18.9%
- density: user-item matrix ~0.03656% dense — median 223 events per user — median 5 events per item — 38 categories
- engagement: ~48.18% events have any engagement — feed ~84.5% of volume — search ~15.5% of volume
- splits: train ~2.73m rows — val ~1.51m rows — test ~1.34m rows — temporal only, no leakage — 43 model features
- signal: best single feature `u_pos_rate` — auc 0.683 for any engagement — `u_rate_like` — auc 0.832 for strong engagement
- loop: peak 19-21h local — median 8 events per session — 4.2 min median session — longer sessions engage way more
- risk: ~44-47% cold items per split — test cold users 3.30% — label mix stable across splits

---

## visual tour — lead with graphs

every claim below points at a figure in `eda/figures/`. start here, read text later.

### 1. the engagement funnel is brutally steep

![engagement funnel by source](eda/figures/03_action_rates.png)

- feed: click ~53.79% — like ~3.00% — follow ~0.41% — forward ~0.31%
- search: click ~12.95% — like / follow / forward are not logged here, they read as 0, not as true zero intent
- given a click: like ~4.0% — follow ~0.55% — forward ~0.42%
- takeaway for the loop: clicks are cheap, deeper actions are rare. treating everything as binary positive throws away the whole hierarchy. that is why we tier intent instead.

![how actions co-occur](eda/figures/03_action_cooccurrence.png)

- conditional matrix `p(column | row)` + phi correlation between click, like, follow, forward
- likes rarely travel with follows/forwards. deep actions are distinct flavours of hooked, not just more of the same.
- takeaway: graded relevance beats binary labels. a follow/forward should outrank a lone click.

### 2. intent tiers — the currency of hooked

![intent tier distribution](eda/figures/02_tier_distribution.png)

- tier 0 seen only: ~51.82% — ~6.77m — mean score 0.0
- tier 1 click: ~43.24% — ~5.65m — mean score 1.0
- tier 2 search-click / long view: ~2.00% — ~261k — mean score 3.0
- tier 3 like: ~2.34% — ~305k — mean score ~3.75
- tier 4 follow / forward: ~0.59% — ~77k — mean score ~6.42
- weights used: click 1.0 — long view 2.0 — like 3.0 — comment 3.0 — forward 4.0 — follow 5.0 — hate -3.0 — search click bonus +2.0 with min tier 2
- takeaway: only ~5% of events carry strong intent. the recommender has to hunt for those.

![intent mix by source](eda/figures/03_tier_by_source.png)

- feed vs search stacked tier mix. search skews higher intent per impression even though its raw click rate is lower.
- takeaway: a click that comes from search is explicit intent. we bump it.

### 3. users — few addicts, long tail of casuals

![user activity inequality](eda/figures/03_user_activity.png)

- events per user log10 looks almost normal-ish but wide. active days per user peaks at 3-4 days, long tail to 20.
- lorenz gini ~0.63. top 1% users make ~12.0% of events. top 10% make ~47.9%.
- takeaway: personalisation matters. global averages lie. heavy users drive the loop, light users churn if you bore them.

### 4. items — extreme popularity bias

![item popularity and bias](eda/figures/03_item_popularity.png)

- rank-frequency log-log is straight-line long tail. lorenz gini ~0.55. top 1% items take ~20.3% of all exposure.
- exposure vs engagement hexbin for items with 20+ events: corr(log exposure, engagement) ~0.128 — weak positive. popular is not the same as loved.
- takeaway: a pure popularity baseline will look ok on average but will miss niche hooks and trap the loop in the head. need intent-aware + novelty.

### 5. categories — where attention actually goes

![top categories and exposure vs engagement](eda/figures/03_category_overview.png)

- 38 first-level categories. cat 39 dominates at ~1.35m events. then 9, 0, 3, 12, 20, 6 follow.
- bubble plot: x exposure share, y engagement rate, size users, colour mean intent. big cats are not always high-engagement cats. small cats can hook hard.
- takeaway: do not just push the biggest cat. push the cat where this user over-engages.

![category x action heatmap](eda/figures/03_category_action_heatmap.png)

- rows categories, cols click/like/follow/forward, cell value rate in %, colour z-score within action.
- some cats are like-cats, some are follow-cats. behaviour taste is category-dependent.
- takeaway: user x category interest vectors need separate counts per action, not one blended score.

![how concentrated are user interests](eda/figures/03_interest_concentration.png)

- interest entropy per user is high, centred ~4 bits. median distinct cats engaged ~20. median share in #1 cat only ~0.21.
- users are broad, not one-trick. they graze.
- takeaway: the loop wins by rotating across a portfolio of interests, not spamming the top cat. diversity + repeat balance matters.

### 6. time — the loop has a daily rhythm

![temporal patterns across day and week](eda/figures/03_temporal_patterns.png)

- events per day: feed spikes early, then settles ~0.5-0.6m per day. engagement rate dashed line climbs over time from ~40% to ~59%.
- weekday x hour heatmap: evenings explode, 19-21h local is prime doomscroll time. weekends spread more daytime volume.
- volume vs engagement by hour: volume peaks 20-21h. engagement rate peaks midday ~12-13h at ~55% and again ~18-19h, collapses 3-4h to ~23%.
- intent mix across day: late night 0-5h is mostly tier 0 scroll-past. daytime has fatter tier 1-3 bands.
- weekend vs weekday: weekend ~4.01m events at 48.80% engagement, weekday ~9.06m at 47.90% — small but real lift.
- takeaway: context features — hour sin/cos, weekend flag — are free signal for when to push light vs deep hooks.

### 7. sessions — the unit of hooked

![session structure](eda/figures/03_sessions.png)

- session = run with <30 min gap, on a 20k-user sample. gaps are lower-bounded by what got logged, so treat as approx.
- ~430,875 sessions — median 8 events per session — median 4.2 min
- engagement by session length: 1 event 19.1% — 2-3 events 28.2% — 4-10 events 36.8% — 11-30 events 45.6% — 31-100 events 51.6% — 100+ events 49.9%
- takeaway: the longer you keep them, the more they engage, up to a point. session length is both label and lever. early-session ranking decides if there is a late session at all.

### 8. watch time carries intent beyond clicks

![watch behaviour](eda/figures/03_watch_behaviour.png)

- watch ratio = play time / duration, clipped 0-5. median ~0.218. huge spike near 0, second bump at ~1.0 = full watches, thin tail to 3+.
- by tier: tier 0 median near 0, tier 1+ medians ~0.7-0.9. watching separates seen-only from engaged.
- duration decile: shorter videos get completed more, longer videos get engaged with more. engagement rate rises with duration decile while mean watch ratio falls.
- corr(play ratio, score) ~0.331
- takeaway: keep `u_play_ratio` and `i_play_ratio`. watch is a dense signal where likes are sparse.

### 9. repeat, novelty, recency — freshness wins

![repeat novelty recency](eda/figures/03_repeat_novelty_recency.png)

- times a user saw same video: 1x dominates ~9.5m pairs. only ~2.6% pairs ever repeat.
- engagement by prior views of same video: 0x 52.04% — 1x 15.22% — 2x 10.57% — 3+ 4.09%. repeats kill.
- share of events in a category new to the user: ~28% day 0, ~16% day 1-2, slides to ~1% by day 10+. median ~2.4%. novelty burns fast.
- engagement vs days since last event in same category dips then holds — recency is non-linear, hence decay features.
- takeaway: penalise seen items hard, decay old interest with ~3-day half-life, keep exploring new cats gently.

### 10. search predicts feed — explicit intent leaks

![search vs feed](eda/figures/03_search_vs_feed.png)

- ~15.5% events are search. ~97.4% users have at least 1 search. search peaks 3-5h, feed peaks 20-21h — different missions, different hours.
- feed engagement split by history in same category:
  - no past feed, no past search: 49.31% — mean tier 0.56
  - no past feed, searched: 55.63% — mean tier 0.66
  - past feed, no search: 54.74% — mean tier 0.62
  - past feed + searched: 64.30% — mean tier 0.75
- takeaway: search history lifts feed engagement ~+6-10pp. the +2.0 search bonus and `u_search_share` / `ctx_is_search` features are justified.

### 11. do the engineered features actually work

![single feature auc](eda/figures/03_feature_viability.png)

- single-feature roc-auc on train for any engagement vs strong engagement. 0.5 = chance.
- top: `u_pos_rate` 0.683 / 0.626 — `u_rate_click` 0.682 / 0.557 — `u_mean_score` 0.661 / 0.784 — `u_play_ratio` 0.627 / 0.557 — `ctx_is_search` 0.407 / 0.437 (negative = search impressions engage less per view but mean more when they do) — `uc_lift` 0.567 / 0.558 — `u_rate_like` 0.556 / 0.832 (monster for strong)
- by group: u strongest and widest spread, then uc, i, ctx, ui, cold weakest alone but critical for fallback.
- takeaway: user habit + user x category lift + item quality all carry. no single feature solves it. need the full 43.

![engagement by feature decile](eda/figures/03_feature_deciles.png)

- `uc_decay_share`, `uc_lift`, `u_pos_rate`, `i_pos_rate` all rise monotonically with decile for `label_pos`. `ui_prior_n` flat then drops — again, repeats hurt.
- takeaway: intent-weighted, time-decayed interest is monotone and model-ready.

![user interest vectors](eda/figures/03_interest_representation.png)

- left: 25 random users x top-12 cats, decayed intent share heatmap — every user looks different.
- right: pca on 20k users, coloured by dominant cat — clusters separate. first 10 pcs explain ~0.141, 0.106, 0.080, 0.067, 0.059, 0.053, 0.048, 0.043, 0.035, 0.031.
- takeaway: `uc_decay_share` vectors are a real taste fingerprint. use them for retrieval + ranking.

### 12. how hard is the test — cold start and drift

![cold start and drift](eda/figures/03_coldstart_drift.png)

- train: 2.73m rows — pos 50.76% — strong 3.06% — search 12.27% — cold user 0.14% — cold item 47.44% — cold user-cat 4.89%
- val: 1.51m rows — pos 48.17% — strong 2.82% — search 11.24% — cold user 0.62% — cold item 46.29% — cold user-cat 4.11%
- test: 1.34m rows — pos 51.38% — strong 3.05% — search 8.01% — cold user 3.30% — cold item 44.79% — cold user-cat 13.20%
- label tier mix stacked bars are stable across splits. no big drift.
- takeaway: user cold start is easy, item cold start is brutal (~45% unseen items). the model must handle unseen videos with category + popularity priors, not id memorisation.

![cohort retention](eda/figures/03_cohorts.png)

- retention heatmap by first-active day x days since + daily active users bar.
- short 19-day window, so cohorts are truncated, but daily actives hold steady. no cliff.
- takeaway: 19 days is enough to learn habits, too short to learn long-term churn. optimise session extension first.

### 13. pipeline hygiene

![raw volume and prevalence](eda/figures/01_raw_volume_and_prevalence.png)

- raw events per day stacked by source + behaviour prevalence by source. feed logs click/like/follow/forward, search logs click only.
- takeaway: never compare like rates across sources. a missing log is 0, not unknown — check `schema.json → observed_by_source`.

![cleaning summary](eda/figures/01_cleaning_summary.png)

- rows removed per step: only exact duplicates, 48,257 rows. plus flag binarisation nan→0, counts→1, play/duration sanitised to `play_ratio` 0-5.
- takeaway: 99.75% kept. cleaning is conservative by design.

![feature correlation](eda/figures/02_feature_correlation.png)

- spearman across sampled train features. user activity block correlates together, uc block together, item block together — expected. no single perfect collinearity that breaks training, but scale + decorrelate in the model pipe, fit on train only.

---

## dataset — what we feed the loop

- source: kuaisar v2 — kuaishou short-video logs — feed + search — china time, utc+8 offset applied for hour features
- raw files under `data/raw/`, git-ignored:
  - `kuaisar_v2/rec_inter.csv` — ~1156.2 mb — feed interactions — user, item, duration, playing time, timestamp, forward, like, follow, search, click
  - `kuaisar_v2/src_inter.csv` — ~553.4 mb — search impressions — user, session, keyword, item, click count
  - `kuaisar_v2/item_features.csv` — ~2008.5 mb — 6.95m rows — 37 raw cats — we use first-level category id
  - `kuaisar_v2/user_features.csv` — ~0.4 mb — 25,877 rows — onehot feats + active levels
  - `kuaisar_v2/social_network.csv` — ~0.0 mb — not used in eda
- canonical event table after notebook 01: one row per logged interaction — `user, item, ts, source(rec/search), click, like, follow, forward, duration, play_time, play_ratio, query`
- time span: 2023-05-22 06:50 to 2023-06-10 08:50 — ~19.1 days — ~1.38m distinct timestamps
- audit notes worth knowing:
  - click counts >1 in ~30,553 rows — 0.155% — binarised to 1
  - strong action without click ~159,881 rows — 0.813% — kept as is, no implied-click fix, because a like can happen without a detail click in feed
  - duration <=0 ~529k rows — 2.691% — set to nan, ratio becomes nan
  - events on items missing from item table ~230k — 1.172% — category becomes unknown path, not dropped
  - no null users/items/timestamps, no impossible timestamps, no bot purge — extreme user check only, 3 users >24,913 events, kept

---

## what was done — three notebooks, zero leakage

### 01 — load, validate and clean — `eda/01_load_and_clean.ipynb`

- locate files by regex on names, inspect 5-row heads per role, resolve raw → canonical columns with an override table you review once
- load with memory-friendly dtypes — int32 users/items, category source, datetime ts — ~1436 mb in memory for ~19.66m rows
- profile dtypes, nulls, nunique, min/max — plot raw volume per day + prevalence
- audit duplicates, nulls, timestamp sanity, flag domains, play/duration sanity, orphan users/items
- clean: drop null user/item/ts — drop impossible timestamps — binarise flags — drop exact duplicates — sanitise play/duration to `play_ratio` clipped 0-5 — sort by ts — save
- writes to `data/interim/`: `events.parquet`, `items.parquet`, `users.parquet`, `schema.json`, `cleaning_report.csv`
- figures: `01_raw_volume_and_prevalence.png`, `01_cleaning_summary.png`

### 02 — preprocess and intent-aware features — `eda/02_preprocess_and_features.ipynb`

team config in one place, edit here and re-run:

- intent tier: click 1 — long view 2 — like 3 — comment 3 — forward 4 — follow 4
- intent weight: click 1.0 — long view 2.0 — like 3.0 — comment 3.0 — forward 4.0 — follow 5.0 — hate -3.0
- search click min tier 2 + bonus 2.0 — half-life 3.0 days — lookback 7.0 days — splits 0.50 / 0.70 / 0.85 — k-core 5 / 3 — local offset 8h — seed 42

steps:

- k-core filter users with <5 events and items with <3 events, iterated to convergence. retention table in tldr above.
- attach category via item map, build per-event `tier`, `score`, `neg`, `pos`, `is_search`
- time axis: `t` days since start, `day` floor, `hour` + `dow` in local time, cut points c1/c2/c3 as fractions of span
- splits are rolling as-of, history only:
  - train history [c1-lookback, c1) — labels [c1, c2) — ~3.87 label days
  - val history [c2-lookback, c2) — labels [c2, c3) — ~2.91 label days
  - test history [c3-lookback, c3) — labels [c3, end] — ~2.91 label days
- feature builder from history window only, leakage asserts included:
  - `u_` user habit: counts, distinct items/days/cats, mean score, pos rate, search share, last-24h count, play ratio, recency, per-action rates
  - `uc_` user x category interest: counts, distinct items, pos sum, score sum, decayed score with 0.5^((asof-t)/half-life), exposure share, decay share, lift = decay share / exposure share, repeat = n - nunique, recency
  - `i_` item quality: exposures, users, mean score, pos rate, median duration, mean play ratio
  - `ui_` user x item memory: prior count, prior max tier, prior summed score
  - `ctx_` context: hour, dow, is search, hour sin/cos, weekend flag, cat id
  - `cold_` fallback flags: cold user, cold item, cold user-cat
- labels per impression in label window: `label_tier` 0-4 — `label_pos` tier>0 — `label_strong` tier>=3-ish high intent
- integrity checks: no nan in features — no label leakage — labels strictly after history — pos present in every split — 43 features listed in `preprocess_config.json`
- writes to `data/processed/`: `events_enriched.parquet`, `features_train.parquet`, `features_val.parquet`, `features_test.parquet`, `user_interest.parquet`, `feature_dictionary.csv`, `id_maps_users.parquet`, `id_maps_items.parquet`, `preprocess_config.json`
- figures: `02_tier_distribution.png`, `02_feature_correlation.png`

### 03 — eda — `eda/03_eda.ipynb`

- 15 sections answering: how big — how rare — how unequal — where attention goes — when the loop peaks — how sessions work — does watch matter — does repeat/recency matter — does search transfer — do cohorts stick — how cold is cold — do features beat chance — can interest vectors separate users
- every figure saved to `eda/figures/03_*.png` — see visual tour above for the read-out
- sampling for heavy plots: 1m-event sample + 20k-user sample, seed 42 — full tables for kpis

---

## feature dictionary — the 43 signals the recommender will get

groups: `u_` who you are — `uc_` what you love right now — `i_` how good the video is — `ui_` have you seen it — `ctx_` when/where — `cold_` fallback — `cat_` which shelf.

| feature | group | what it means |
|---|---|---|
| `u_n_events` | u | events in history window |
| `u_n_items` | u | distinct videos seen |
| `u_active_days` | u | distinct active days |
| `u_mean_score` | u | mean intent score per event |
| `u_pos_rate` | u | share of events with any engagement |
| `u_search_share` | u | share of events from search |
| `u_recent24` | u | events in last 24h before cutoff |
| `u_n_cats` | u | distinct categories seen |
| `u_recency` | u | days since last event |
| `u_play_ratio` | u | mean watch ratio |
| `u_rate_click` | u | click rate in history |
| `u_rate_like` | u | like rate in history |
| `u_rate_follow` | u | follow rate in history |
| `u_rate_forward` | u | forward rate in history |
| `uc_n` | uc | user events in this category |
| `uc_items` | uc | distinct videos in this category |
| `uc_pos` | uc | engaged events in this category |
| `uc_score` | uc | summed intent scores in category |
| `uc_decay` | uc | time-decayed intent in category, 3-day half-life |
| `uc_exposure_share` | uc | share of user exposure in this category |
| `uc_decay_share` | uc | share of decayed intent in this category |
| `uc_lift` | uc | intent share / exposure share, >1 means over-engages |
| `uc_repeat` | uc | repeat interactions inside category |
| `uc_recency` | uc | days since last event in this category |
| `i_n` | i | item exposures in history |
| `i_users` | i | distinct users who saw item |
| `i_mean_score` | i | item mean intent score |
| `i_pos_rate` | i | item engagement rate |
| `i_duration` | i | video duration median |
| `i_play_ratio` | i | item mean watch ratio |
| `ui_prior_n` | ui | times this user saw this video before |
| `ui_prior_max_tier` | ui | highest prior tier on this video |
| `ui_prior_score` | ui | summed prior intent on this video |
| `cold_user` | cold | 1 if user unseen in history |
| `cold_item` | cold | 1 if item unseen in history |
| `cold_uc` | cold | 1 if user never saw this category |
| `ctx_hour` | ctx | local hour 0-23 |
| `ctx_dow` | ctx | weekday 0 = mon |
| `ctx_is_search` | ctx | 1 if this impression is search |
| `ctx_hour_sin` | ctx | cyclic hour sin |
| `ctx_hour_cos` | ctx | cyclic hour cos |
| `ctx_is_weekend` | ctx | 1 if sat/sun |
| `cat_id` | cat | integer category code |

full descriptions + correlations in `data/processed/feature_dictionary.csv` and `02_feature_correlation.png`. scaling, encoding of `cat_id`, and any standardisation belong in the model pipe, fit on train only.

---

## so what — what the loop wants

- predict depth, not just click. tier 0 vs 1 vs 3 vs 4 behave differently in watch, repeat, and search lift. use `label_tier` with graded gains, keep `label_pos` only as an equal-weight baseline.
- personalise by habit + current interest. `u_pos_rate`, `u_mean_score`, `uc_lift`, `uc_decay_share` are the most monotone signals. global popularity alone will bore light users and overfeed heavy users.
- time it. evenings for volume, midday + early evening for engagement rate, late night for lean-back. hour + weekend are cheap wins.
- protect novelty. repeats convert ~3-10x worse. downrank seen items hard, decay with 3-day half-life, keep 7-day lookback so scales stay comparable.
- borrow search intent. if they searched it and engaged feed in that cat before, that cat is hot. boost it.
- watch is the dense label. likes are sparse at ~3%, watch ratio correlates ~0.331 with intent and separates tier 0 from the rest. use it.
- plan for cold items. ~45% of label rows are unseen videos. back off to category interest + item priors + popularity, plus explore.
- evaluate like the product: rank each user’s label-window impressions, score ndcg@k with tier gains. beat popularity and equal-weight baselines before calling it a win.

---

## run it

```text
data/raw/          <- put kuaisar v2 csvs here, stays git-ignored
data/interim/      <- written by 01
data/processed/    <- written by 02
eda/figures/       <- all pngs, committed
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyarrow jupyterlab
jupyter lab
# run in order:
# eda/01_load_and_clean.ipynb
# eda/02_preprocess_and_features.ipynb
# eda/03_eda.ipynb
```

notes:

- run 01 and 02 before 03. 03 reads `data/processed/`.
- raw stays in `data/raw/` and is git-ignored. nothing in 01 writes outside `data/` and `eda/figures/`.
- behaviour a source does not log stays 0. check `schema.json → observed_by_source` before comparing search vs feed on likes/follows/forwards.
- scaling and encoding live in the model pipe, fit on train only. no peeking at val/test.

---

## repo map

```text
doomscroll/
  eda/
    01_load_and_clean.ipynb        <- raw → canonical events
    02_preprocess_and_features.ipynb <- intent tiers, splits, 43 features
    03_eda.ipynb                   <- all the graphs above
    figures/                       <- 22 pngs, embedded in this readme
  recommender-engine/              <- next up, ranking + eval, currently brewing
  data/
    raw/                           <- ignored, put kuaisar v2 here
    interim/                       <- ignored, events.parquet + schema.json
    processed/                     <- ignored, features_{train,val,test}.parquet
```

---

## roadmap — more eda graphs to come, then the recommender

eda still cooking:

- per-category retention curves — which cats hook vs burn out
- creator/author concentration — do few authors own the loop
- session-start vs session-end ranking — what opens vs what extends
- dwell + skip modelling — separating fast skips from slow burns
- late-night vs prime-time taste shift — same user, different cravings
- search query → feed category transfer matrix — explicit → implicit mapping

recommender-engine next:

- baselines to beat: global popularity — personalised popularity — equal-weight interactions
- intent-aware ranker using `label_tier` and intent weights, time-decayed user x category vectors
- cold-start path: category priors + item quality + explore
- eval: per-user ranking of label-window impressions — ndcg@k with graded tier gains — plus coverage, diversity, novelty
- loop metric we care about: longer sessions, more return visits, more deep actions per hour — keep them hooked, kindly

---

## caveats

- 19 days is short. habits yes, churn no. session extension first, long-term retention later.
- logs are interactions the dataset kept, not every pixel scrolled. gaps and sessions are approx.
- search only logs clicks. do not read its 0 likes as dislike.
- k-core keeps 66.6% events and 18.9% items. tail items are cut for modelling, but the product still has to serve them.
- single-feature auc is sanity, not a model. interactions between features will matter more.

---

## license + crew

- license: mit — see `license`
- data: kuaisar v2, for research. raw data not committed.
- workflow: eda notebooks are the source of truth. figures are artefacts, not decorations.
- status: eda live, recommender brewing. more graphs loading. lets keep them watching.

