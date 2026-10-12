# doomscroll

> behaviour-aware analysis for short-video recommendation. modelling interaction intent, session dynamics, and retention drivers on the kuaisar v2 dataset.

this repository contains exploratory data analysis supporting a video recommendation system. the objective is to model the platform feedback loop — how feed and search interactions predict continued engagement — and to define features and evaluation protocols that maximise session duration and return usage. this phase covers data validation, preprocessing, feature engineering, and eda. the ranking model is developed separately in `recommender-engine/` following completion of this analysis.

additional eda outputs are planned and will be appended as subsequent figures and sections.

---

## summary statistics

- raw input: ~19.66m events — ~14.61m feed + ~5.06m search — 25,877 users — ~6.89m items — 19.1 days — 2023-05-22 to 2023-06-10
- cleaned: ~19.62m events retained — 99.75% of raw — 48,257 exact duplicate rows removed
- modelled: ~13.07m events after k-core filtering — 66.6% events retained — 25,865 users retained — ~100% — ~1.30m items retained — 18.9%
- sparsity: user-item matrix density ~0.03656% — median 223 events per user — median 5 events per item — 38 categories
- engagement: ~48.18% of events include at least one engagement signal — feed ~84.5% of volume — search ~15.5% of volume
- splits: train ~2.73m rows — validation ~1.51m rows — test ~1.34m rows — strictly temporal, no leakage — 43 model features
- predictive signal: strongest univariate predictor for any engagement is `u_pos_rate` — auc 0.683 — strongest for strong engagement is `u_rate_like` — auc 0.832
- temporal concentration: peak volume 19-21h local time — median session 8 events over 4.2 min — engagement rate increases with session length
- evaluation risk: ~44-47% cold items per split — test cold users 3.30% — label tier distribution stable across splits

---

## analytical results

each subsection references committed figures in `eda/figures/`. interpretations summarise the implication for feature design and ranking.

### 1. engagement funnel and action distribution

![engagement probability by source](eda/figures/03_action_rates.png)

- feed: click ~53.79% — like ~3.00% — follow ~0.41% — forward ~0.31%
- search: click ~12.95% — like, follow, and forward are not logged for this source and are encoded as 0, which denotes absence of logging rather than confirmed negative behaviour
- conditional on click: like ~4.0% — follow ~0.55% — forward ~0.42%
- interpretation: interaction frequency is inversely related to intent depth. binary positive labelling discards this hierarchy and motivates a graded intent representation.

![action co-occurrence and correlation](eda/figures/03_action_cooccurrence.png)

- left panel: conditional probability matrix `p(column | row)`. right panel: phi correlation between click, like, follow, and forward.
- likes exhibit low co-occurrence with follows and forwards, indicating distinct behavioural intents rather than scaled variants of a single signal.
- interpretation: graded relevance is required. follow and forward events should receive higher gain than isolated clicks during ranking evaluation.

### 2. intent tier definition and distribution

![intent tier distribution](eda/figures/02_tier_distribution.png)

- tier 0, exposure without engagement: ~51.82% — ~6.77m events — mean score 0.0
- tier 1, click: ~43.24% — ~5.65m events — mean score 1.0
- tier 2, search-click and long view: ~2.00% — ~261k events — mean score 3.0
- tier 3, like: ~2.34% — ~305k events — mean score ~3.75
- tier 4, follow and forward: ~0.59% — ~77k events — mean score ~6.42
- configured weights: click 1.0 — long view 2.0 — like 3.0 — comment 3.0 — forward 4.0 — follow 5.0 — hate -3.0 — search click bonus +2.0 with minimum tier 2
- interpretation: approximately 5% of events carry high-intent signal. the ranking objective must prioritise retrieval of this minority class.

![intent tier composition by source](eda/figures/03_tier_by_source.png)

- stacked tier composition for feed versus search. search yields higher mean intent per impression despite a lower raw click rate.
- interpretation: search-originated clicks encode explicit intent and justify the applied tier promotion and score bonus.

### 3. user activity distribution

![user activity distribution and inequality](eda/figures/03_user_activity.png)

- events per user in log10 scale show a broad unimodal distribution. active days per user peak at 3-4 days with a tail extending to 20 days.
- lorenz curve gini ~0.63. the top 1% of users account for ~12.0% of events. the top 10% account for ~47.9%.
- interpretation: activity is highly skewed. global estimators are insufficient. user-level normalisation and personalised priors are required.

### 4. item popularity and exposure bias

![item popularity, exposure inequality, and engagement](eda/figures/03_item_popularity.png)

- rank-frequency plot in log-log scale follows an approximately linear long-tail form. exposure gini ~0.55. the top 1% of items receive ~20.3% of total exposure.
- exposure versus engagement rate for items with 20 or more events: correlation between log exposure and engagement rate ~0.128, indicating a weak positive association.
- interpretation: exposure and engagement are not interchangeable. a popularity-only baseline provides coverage but underperforms on niche relevance and requires correction for novelty and intent.

### 5. category-level engagement analysis

![category exposure and engagement comparison](eda/figures/03_category_overview.png)

- 38 first-level categories. category 39 is largest at ~1.35m events, followed by categories 9, 0, 3, 12, 20, and 6.
- bubble plot axes: x exposure share, y engagement rate, marker size proportional to user count, colour mapped to mean intent score. high-exposure categories do not uniformly yield high engagement rates.
- interpretation: allocation should follow user-specific category affinity rather than global category volume.

![category by action rate matrix](eda/figures/03_category_action_heatmap.png)

- matrix of categories by click, like, follow, and forward rates. cell annotations show rate in percent. colour encodes within-action z-score.
- categories exhibit action-specific affinities. preference structure varies by behaviour type within the same category.
- interpretation: user by category representations should retain per-action granularity in addition to aggregated scores.

![user interest concentration](eda/figures/03_interest_concentration.png)

- per-user interest entropy centres near 4 bits. median distinct engaged categories is 20. median share of engagement in the top category is ~0.21.
- interpretation: users maintain broad interest portfolios. ranking should balance exploitation of dominant categories with diversified exposure.

### 6. temporal engagement patterns

![temporal volume and engagement patterns](eda/figures/03_temporal_patterns.png)

- daily volume: feed volume peaks early in the observation window and stabilises at ~0.5-0.6m events per day. the aggregate engagement rate increases from ~40% to ~59% across the period.
- weekday by hour volume matrix: maximum density occurs at 19-21h local time. weekend volume is more evenly distributed across daytime hours.
- hourly breakdown: volume peaks at 20-21h. engagement rate peaks at ~12-13h near 55% with a secondary peak at ~18-19h, and declines to ~23% at 03-04h.
- tier composition by hour: overnight hours 00-05h are dominated by tier 0 exposures. daytime hours contain proportionally larger tier 1-3 bands.
- weekend aggregate: ~4.01m events at 48.80% engagement. weekday aggregate: ~9.06m events at 47.90%.
- interpretation: hour and weekend indicators provide low-cost contextual signal for time-dependent ranking adjustments.

### 7. session structure

![session gap, length, and frequency distributions](eda/figures/03_sessions.png)

- session definition: consecutive events from the same user separated by less than 30 min, computed on a 20k-user sample. gap measurements are lower bounds constrained by logged interactions.
- total ~430,875 sessions — median 8 events per session — median duration 4.2 min.
- engagement rate by session length: 1 event 19.1% — 2-3 events 28.2% — 4-10 events 36.8% — 11-30 events 45.6% — 31-100 events 51.6% — 100+ events 49.9%.
- interpretation: engagement rate increases monotonically with session length up to extended sessions. accurate ranking of initial impressions is determinative for session continuation.

### 8. watch-time behaviour

![watch ratio distribution and duration effects](eda/figures/03_watch_behaviour.png)

- watch ratio defined as play time divided by duration, clipped to the interval 0-5. median ~0.218. the distribution shows a primary mode near 0, a secondary mode near 1.0 corresponding to complete views, and a sparse tail above 2.0.
- stratification by tier: tier 0 median near 0. tier 1 and above exhibit medians in the range ~0.7-0.9.
- duration decile analysis: shorter videos yield higher completion ratios while longer videos yield higher engagement rates. mean watch ratio decreases with duration decile as engagement rate increases.
- correlation between play ratio and intent score is ~0.331.
- interpretation: watch ratio provides a dense continuous signal in regimes where discrete actions are sparse. both user-level and item-level aggregates are retained as features.

### 9. repeat consumption, novelty, and recency

![repeat frequency, category novelty, and recency effects](eda/figures/03_repeat_novelty_recency.png)

- repeat distribution: single-view user-item pairs dominate at ~9.5m pairs. approximately 2.6% of pairs contain a repeat view.
- engagement rate by prior view count for the same video: 0 priors 52.04% — 1 prior 15.22% — 2 priors 10.57% — 3 or more priors 4.09%.
- proportion of events in categories previously unseen by the user: ~28% on day 0, ~16% on days 1-2, declining to ~1% after day 10. median ~2.4%.
- engagement rate by days since last interaction in the same category is non-monotonic, supporting a parameterised decay rather than a linear recency term.
- interpretation: previously consumed items should be substantially downweighted. category interest should apply exponential time decay with an approximate 3-day half-life over a 7-day lookback window.

### 10. search-to-feed intent transfer

![search and feed temporal and behavioural comparison](eda/figures/03_search_vs_feed.png)

- search constitutes ~15.5% of events. approximately 97.4% of users record at least one search event. search volume peaks at 03-05h while feed volume peaks at 20-21h, indicating distinct usage contexts.
- feed engagement rate conditioned on same-category history:
  - no prior feed engagement, no prior search: 49.31% — mean tier 0.56
  - no prior feed engagement, prior search: 55.63% — mean tier 0.66
  - prior feed engagement, no prior search: 54.74% — mean tier 0.62
  - prior feed engagement and prior search: 64.30% — mean tier 0.75
- interpretation: prior search in a category yields a consistent 6-10 percentage point lift in feed engagement. this supports retention of search-derived features and the configured search bonus.

### 11. feature predictive viability

![univariate predictive performance by feature](eda/figures/03_feature_viability.png)

- univariate roc-auc on train rows for any engagement versus strong engagement. a value of 0.5 denotes chance performance.
- selected results: `u_pos_rate` 0.683 / 0.626 — `u_rate_click` 0.682 / 0.557 — `u_mean_score` 0.661 / 0.784 — `u_play_ratio` 0.627 / 0.557 — `ctx_is_search` 0.407 / 0.437, where values below 0.5 indicate an inverse association — `uc_lift` 0.567 / 0.558 — `u_rate_like` 0.556 / 0.832.
- grouped signal strength measured by absolute deviation from 0.5 is highest for user features, followed by user-category and item groups. context, user-item, and cold-start indicators are weaker univariately but retained for conditional and fallback utility.
- interpretation: no single feature is sufficient. the retained 43-feature set combines habitual, contextual, affinity-based, and quality-based signals.

![engagement rate by feature decile](eda/figures/03_feature_deciles.png)

- `uc_decay_share`, `uc_lift`, `u_pos_rate`, and `i_pos_rate` exhibit monotonically increasing engagement rates across deciles for `label_pos`. `ui_prior_n` is approximately flat before declining, consistent with repeat fatigue.
- interpretation: time-decayed and intent-weighted affinity features display monotone response curves suitable for gradient-based and tree-based rankers.

![user interest vector structure](eda/figures/03_interest_representation.png)

- left panel: decayed intent share across the top 12 categories for 25 sampled users. right panel: two-component pca projection of 20k user interest vectors coloured by dominant category.
- the first 10 principal components explain approximately 0.141, 0.106, 0.080, 0.067, 0.059, 0.053, 0.048, 0.043, 0.035, and 0.031 of variance respectively.
- interpretation: decayed category-share vectors form discriminative user representations applicable to both retrieval and ranking stages.

### 12. cold-start prevalence and temporal stability

![cold-start rates and label distribution by split](eda/figures/03_coldstart_drift.png)

- train: 2.73m rows — positive rate 50.76% — strong rate 3.06% — search share 12.27% — cold user 0.14% — cold item 47.44% — cold user-category 4.89%
- validation: 1.51m rows — positive rate 48.17% — strong rate 2.82% — search share 11.24% — cold user 0.62% — cold item 46.29% — cold user-category 4.11%
- test: 1.34m rows — positive rate 51.38% — strong rate 3.05% — search share 8.01% — cold user 3.30% — cold item 44.79% — cold user-category 13.20%
- label tier composition is stable across splits, indicating limited target drift.
- interpretation: user cold start is minimal while item cold start affects approximately 45% of label rows. the model requires category-level priors and content-based fallback rather than reliance on item identity memorisation.

![cohort retention and active user counts](eda/figures/03_cohorts.png)

- retention matrix by first-active day and elapsed days, alongside daily active user counts.
- the 19-day observation window truncates long-horizon cohorts. daily active counts remain stable without abrupt discontinuities.
- interpretation: the window is sufficient for short-term habit estimation but insufficient for long-term churn modelling. optimisation should prioritise session extension and short-term return.

### 13. data validation and preprocessing diagnostics

![raw volume and behaviour prevalence](eda/figures/01_raw_volume_and_prevalence.png)

- daily raw event counts by source and per-action prevalence by source. feed logs click, like, follow, and forward. search logs click only.
- interpretation: cross-source comparisons of like, follow, and forward rates are invalid without conditioning on logging coverage. absent logs are encoded as 0 and documented in `schema.json` under `observed_by_source`.

![cleaning operation summary](eda/figures/01_cleaning_summary.png)

- rows removed per cleaning operation. the sole row-reducing step is exact duplicate removal at 48,257 rows. remaining steps perform flag binarisation from nan to 0 and counts to 1, and sanitisation of play time and duration into `play_ratio` bounded to 0-5.
- interpretation: cleaning is conservative at 99.75% retention and preserves distributional structure for downstream modelling.

![feature correlation matrix](eda/figures/02_feature_correlation.png)

- spearman correlation across sampled train features. expected block structure is present within user activity, user-category affinity, and item quality groups. no degenerate perfect collinearity is observed.
- interpretation: feature scaling and any decorrelation procedures should be implemented in the modelling pipeline and fitted exclusively on train data.

---

## dataset specification

- source: kuaisar v2 — kuaishou short-video interaction logs — feed and search surfaces — timestamps standardised with a utc+8 offset for local hour derivation
- raw files under `data/raw/`, excluded from version control:
  - `kuaisar_v2/rec_inter.csv` — ~1156.2 mb — feed interactions — user, item, duration, playing time, timestamp, forward, like, follow, search, click
  - `kuaisar_v2/src_inter.csv` — ~553.4 mb — search impressions — user, session, keyword, item, click count
  - `kuaisar_v2/item_features.csv` — ~2008.5 mb — 6.95m rows — 37 raw categories — first-level category id is used as the canonical category
  - `kuaisar_v2/user_features.csv` — ~0.4 mb — 25,877 rows — onehot attributes and activity levels
  - `kuaisar_v2/social_network.csv` — ~0.0 mb — excluded from this analysis
- canonical event schema after notebook 01: one row per logged interaction — `user, item, ts, source(rec/search), click, like, follow, forward, duration, play_time, play_ratio, query`
- temporal coverage: 2023-05-22 06:50 to 2023-06-10 08:50 — ~19.1 days — ~1.38m distinct timestamps
- data quality audit:
  - click counts exceeding 1 in ~30,553 rows — 0.155% — binarised to 1
  - high-intent action without click in ~159,881 rows — 0.813% — retained without implied-click correction, as feed likes may occur without a detail-page click
  - non-positive duration in ~529k rows — 2.691% — set to nan with derived ratio set to nan
  - events referencing items absent from the item table in ~230k rows — 1.172% — retained with category handled through the unknown path
  - zero null users, items, or timestamps. zero out-of-range timestamps. no bot-threshold filtering applied — 3 users exceed 24,913 events and are retained with distributional review

---

## methodology

### 01 — data loading, validation, and cleaning — `eda/01_load_and_clean.ipynb`

- file discovery by filename pattern matching, per-file 5-row schema inspection, and explicit raw-to-canonical column resolution with a configurable override mapping
- memory-efficient loading with reduced integer dtypes, categorical source encoding, and datetime parsing — approximately 1436 mb resident for ~19.66m rows
- profiling of dtypes, null rates, cardinality, and numeric ranges. visualisation of daily volume and per-source prevalence
- audit of exact duplicates, null keys, timestamp validity, flag domains, play-time and duration validity, and orphan references to user and item tables
- cleaning operations: removal of null user, item, or timestamp rows — removal of out-of-range timestamps — binarisation of behaviour flags — removal of exact duplicates — sanitisation of play time and duration into `play_ratio` clipped to 0-5 — chronological sorting — serialisation
- outputs to `data/interim/`: `events.parquet`, `items.parquet`, `users.parquet`, `schema.json`, `cleaning_report.csv`
- figures: `01_raw_volume_and_prevalence.png`, `01_cleaning_summary.png`

### 02 — preprocessing and intent-aware feature engineering — `eda/02_preprocess_and_features.ipynb`

centralised configuration, modified in a single block prior to re-execution:

- intent tier mapping: click 1 — long view 2 — like 3 — comment 3 — forward 4 — follow 4
- intent weight mapping: click 1.0 — long view 2.0 — like 3.0 — comment 3.0 — forward 4.0 — follow 5.0 — hate -3.0
- search click minimum tier 2 with bonus 2.0 — recency half-life 3.0 days — history lookback 7.0 days — split fractions 0.50 / 0.70 / 0.85 — k-core thresholds 5 / 3 — local offset 8h — seed 42

procedure:

- iterative k-core filtering of users with fewer than 5 events and items with fewer than 3 events until convergence
- category assignment through the item mapping. per-event derivation of `tier`, `score`, `neg`, `pos`, and `is_search`
- temporal axis construction: continuous time `t` in days since origin, integer `day`, local `hour` and `dow`, with cut points c1, c2, and c3 defined as fractions of the total span
- rolling as-of splits computed exclusively from history windows:
  - train history [c1-lookback, c1) — labels [c1, c2) — ~3.87 label days
  - validation history [c2-lookback, c2) — labels [c2, c3) — ~2.91 label days
  - test history [c3-lookback, c3) — labels [c3, end] — ~2.91 label days
- leakage-controlled feature construction from history only, with programmatic assertions:
  - `u_` user behaviour: event counts, distinct item, day, and category counts, mean intent score, positive rate, search share, trailing 24h count, mean watch ratio, recency, and per-action rates
  - `uc_` user by category affinity: event counts, distinct items, engaged counts, summed scores, exponentially decayed scores with base 0.5 and the configured half-life, exposure share, decayed intent share, lift defined as decay share divided by exposure share, repeat count defined as events minus distinct items, and recency
  - `i_` item statistics: exposure count, unique users, mean intent score, positive rate, median duration, and mean watch ratio
  - `ui_` user by item memory: prior exposure count, prior maximum tier, and prior summed intent
  - `ctx_` context: hour, weekday, search indicator, cyclic hour encodings, weekend indicator, and category id
  - `cold_` fallback indicators: unseen user, unseen item, and unseen user-category combinations
- label definition per impression in the label window: `label_tier` in the range 0-4 — `label_pos` indicating tier greater than 0 — `label_strong` indicating high-intent tiers
- integrity validation: absence of nan values in features — exclusion of label columns from the feature set — strict temporal ordering of labels after history — positive class presence in all splits — 43 features recorded in `preprocess_config.json`
- outputs to `data/processed/`: `events_enriched.parquet`, `features_train.parquet`, `features_val.parquet`, `features_test.parquet`, `user_interest.parquet`, `feature_dictionary.csv`, `id_maps_users.parquet`, `id_maps_items.parquet`, `preprocess_config.json`
- figures: `02_tier_distribution.png`, `02_feature_correlation.png`

### 03 — exploratory data analysis — `eda/03_eda.ipynb`

- fifteen analytical sections covering volume, action rarity, inequality, category structure, temporal structure, session structure, watch behaviour, repeat and recency effects, search transfer, cohort retention, cold-start prevalence, univariate feature viability, and interest representation geometry
- all outputs serialised to `eda/figures/03_*.png` and reviewed in the analytical results section above
- computationally intensive visualisations use a 1m-event sample and a 20k-user subsample with seed 42. tabulated metrics use full data

---

## feature specification — 43 model inputs

prefix groups: `u_` user behaviour — `uc_` user by category affinity — `i_` item statistics — `ui_` user by item memory — `ctx_` request context — `cold_` cold-start indicators — `cat_` category identifier.

| feature | group | definition |
|---|---|---|
| `u_n_events` | u | events in history window |
| `u_n_items` | u | distinct videos observed |
| `u_active_days` | u | distinct active days |
| `u_mean_score` | u | mean intent score per event |
| `u_pos_rate` | u | proportion of events with any engagement |
| `u_search_share` | u | proportion of events originating from search |
| `u_recent24` | u | events in trailing 24h before cutoff |
| `u_n_cats` | u | distinct categories observed |
| `u_recency` | u | days since most recent event |
| `u_play_ratio` | u | mean watch ratio |
| `u_rate_click` | u | historical click rate |
| `u_rate_like` | u | historical like rate |
| `u_rate_follow` | u | historical follow rate |
| `u_rate_forward` | u | historical forward rate |
| `uc_n` | uc | user events in this category |
| `uc_items` | uc | distinct videos in this category |
| `uc_pos` | uc | engaged events in this category |
| `uc_score` | uc | summed intent scores in this category |
| `uc_decay` | uc | exponentially decayed intent in this category, 3-day half-life |
| `uc_exposure_share` | uc | proportion of user exposure allocated to this category |
| `uc_decay_share` | uc | proportion of decayed intent allocated to this category |
| `uc_lift` | uc | decay share divided by exposure share, values above 1 indicate over-engagement |
| `uc_repeat` | uc | repeat interactions within this category |
| `uc_recency` | uc | days since most recent event in this category |
| `i_n` | i | item exposure count in history |
| `i_users` | i | distinct users exposed to item |
| `i_mean_score` | i | item mean intent score |
| `i_pos_rate` | i | item engagement rate |
| `i_duration` | i | item median duration |
| `i_play_ratio` | i | item mean watch ratio |
| `ui_prior_n` | ui | prior exposures of this user to this video |
| `ui_prior_max_tier` | ui | maximum prior intent tier on this video |
| `ui_prior_score` | ui | summed prior intent on this video |
| `cold_user` | cold | indicator for user absent from history |
| `cold_item` | cold | indicator for item absent from history |
| `cold_uc` | cold | indicator for user-category pair absent from history |
| `ctx_hour` | ctx | local hour in range 0-23 |
| `ctx_dow` | ctx | weekday with 0 denoting mon |
| `ctx_is_search` | ctx | indicator for search-originated impression |
| `ctx_hour_sin` | ctx | cyclic sine encoding of hour |
| `ctx_hour_cos` | ctx | cyclic cosine encoding of hour |
| `ctx_is_weekend` | ctx | indicator for sat or sun |
| `cat_id` | cat | integer category code |

complete definitions and univariate correlations are recorded in `data/processed/feature_dictionary.csv` and `02_feature_correlation.png`. normalisation, standardisation, and encoding of `cat_id` are deferred to the modelling pipeline and must be fitted on train data only.

---

## implications for recommender design

- optimise for graded intent rather than binary clicks. tier 0 through tier 4 differ systematically in watch behaviour, repeat sensitivity, and search lift. use `label_tier` with graded gains. retain `label_pos` solely as an equal-weight baseline.
- condition ranking on habitual behaviour and current affinity. `u_pos_rate`, `u_mean_score`, `uc_lift`, and `uc_decay_share` show the most consistent monotone response. global popularity alone is insufficient for heterogeneous activity levels.
- incorporate temporal context. evening hours drive volume while midday and early evening drive engagement efficiency. hour encodings and weekend indicators are retained for this purpose.
- enforce novelty control. repeat exposures convert at substantially reduced rates. apply penalties for previously seen items, exponential interest decay, and a fixed 7-day lookback to preserve cross-split comparability.
- propagate search intent into feed ranking. same-category search history produces measurable lift in feed engagement and tier. search-derived features are therefore retained.
- utilise watch ratio as a dense auxiliary signal. discrete actions are sparse near 3% for likes, while watch ratio correlates materially with intent and separates non-engaged exposures from engaged ones.
- design an explicit cold-item path. with approximately 45% unseen items per split, fallback logic based on category affinity, item quality priors, and controlled exploration is required.
- evaluate with per-user ranking of label-window impressions using ndcg at k with tier-derived gains. candidate systems must outperform global popularity, personalised popularity, and equal-weight interaction baselines.

---

## reproduction

```text
data/raw/          <- kuaisar v2 csv inputs, excluded from version control
data/interim/      <- outputs of notebook 01
data/processed/    <- outputs of notebook 02
eda/figures/       <- committed visualisation artefacts
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn pyarrow jupyterlab
jupyter lab
# execute strictly in order:
# eda/01_load_and_clean.ipynb
# eda/02_preprocess_and_features.ipynb
# eda/03_eda.ipynb
```

operational notes:

- notebook 03 depends on processed artefacts from notebooks 01 and 02.
- raw data remains in `data/raw/` and is git-ignored. notebook 01 writes exclusively to `data/` and `eda/figures/`.
- behaviour types not logged by a given source are encoded as 0. consult `schema.json` under `observed_by_source` before cross-source comparison of like, follow, and forward metrics.
- all scaling and encoding must be fitted on train data only, with no use of validation or test statistics.

---

## repository structure

```text
doomscroll/
  eda/
    01_load_and_clean.ipynb          <- raw inputs to canonical event table
    02_preprocess_and_features.ipynb <- intent tiers, temporal splits, 43 features
    03_eda.ipynb                     <- behavioural analysis and visualisations
    figures/                         <- 22 committed png files referenced above
  recommender-engine/                <- ranking models and evaluation, following eda
  data/
    raw/                             <- ignored, kuaisar v2 source files
    interim/                         <- ignored, events.parquet and schema.json
    processed/                       <- ignored, features_{train,val,test}.parquet
```

---

## roadmap

planned eda extensions:

- per-category retention curves distinguishing sustained interest from transient exposure
- creator and author concentration analysis for supply-side skew
- session-phase analysis comparing opening-ranked items against continuation-ranked items
- dwell-time and skip-behaviour decomposition for fast versus sustained disengagement
- diurnal preference shift between late-night and prime-time consumption
- search-query to feed-category transfer matrix for explicit-to-implicit intent mapping

subsequent recommender-engine development:

- baseline implementations: global popularity — personalised popularity — equal-weight interaction ranking
- intent-aware ranker trained on `label_tier` with configured intent weights and decayed user by category vectors
- cold-start handling through category priors, item quality estimates, and exploration policy
- evaluation by per-user ranking of label-window impressions — ndcg at k with graded tier gains — supplemented by coverage, diversity, and novelty diagnostics
- primary product metrics: extended session length, repeat visitation rate, and high-intent actions per active hour

---

## limitations and caveats

- the 19-day window supports short-term habit estimation but not long-term churn analysis. initial optimisation targets session extension.
- logged interactions represent retained events rather than exhaustive scroll telemetry. derived gaps and session boundaries are approximate.
- search impressions log click events only. zero values for other actions in this source must not be interpreted as negative feedback.
- k-core filtering retains 66.6% of events and 18.9% of items. tail items excluded from modelling remain in the serving distribution and require fallback handling.
- univariate auc values validate feature viability only. multivariate interactions are expected to contribute additional predictive value in the ranking model.

---

## license and attribution

- license: mit — see `license`
- data: kuaisar v2, provided for research use. raw data is not committed to this repository.
- provenance: eda notebooks constitute the source of truth. figures in `eda/figures/` are derived artefacts.
- status: eda phase complete with extensions planned. recommender-engine implementation follows.
