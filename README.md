# Dota 2 Draft Outcome Predictor

Predicts the probability of Radiant winning a pro Dota 2 match based on the draft,
team history, and a custom-built Elo rating system.

## Setup

```bash
pip install -r requirements.txt
```

Get a free OpenDota API key at https://www.opendota.com/api-keys and add it to
`1_collect_data.py` → `CONFIG["api_key"]`. Not required, but raises rate limits.

---

## Run Order

Run scripts in order. Each script depends on the output of the previous one.

```bash
python 1_collect_data.py       # Fetch pro match data from OpenDota API (~3 hrs for 10k matches)
python 2_build_elo.py          # Compute Elo ratings for each team (seconds)
python 3_build_features.py     # Build the feature matrix (seconds)
python 4_train_model.py        # Train all models (1–10 minutes depending on tuning)
python 5_predict.py            # Predict a single match (instant)
python 6_export_elo_csv.py     # Export Elo ratings to CSV (optional)
python 7_explain_model.py      # Generate feature importance charts (optional)
```

---

## Scripts

### 1_collect_data.py
Fetches pro match data from the OpenDota API.
- Stores one JSON file per match under `data/raw/matches/`
- Also fetches team names (`teams.json`) and hero constants (`hero_constants.json`)
- Logs progress to `data/collect_data.log`

**Key config:**

| Setting | Default | Notes |
|---|---|---|
| `target_match_count` | 10,000 | More data = better model |
| `request_delay` | 1.05s | Keeps under 60 req/min (free tier) |
| `api_key` | `""` | Optional — raises rate limit cap |

---

### 2_build_elo.py
Builds a chronological Elo rating system for all teams.
- Processes matches in time order
- Outputs `data/processed/matches_with_elo.json` with pre-match Elo attached to each match

**Key config:**

| Setting | Default | Notes |
|---|---|---|
| `k_factor` | 32 | Higher = faster Elo shifts |
| `initial_elo` | 1500 | Starting rating for new teams |

---

### 3_build_features.py
Transforms enriched match data into an ML-ready feature matrix.
- Filters by patch, date, and optionally Elo reliability
- Builds all feature groups (see Features section below)
- Outputs `data/processed/features.csv` and supporting lookup JSONs

**Key config:**

| Setting | Default | Notes |
|---|---|---|
| `last_n_patches` | 3 | `None` = use all patches |
| `min_date` | `2023-01-01` | Drop older matches |
| `include_draft_order` | `True` | Adds 21 draft-order columns |
| `include_bans` | `True` | Adds ban vector columns |
| `include_synergy` | `True` | Adds hero pair synergy scores |
| `include_counters` | `True` | Adds hero matchup counter scores |
| `team_encoding_min_matches` | 5 | Min matches for a real team encoding |
| `synergy_min_matches` | 3 | Min observations before trusting pair win rate |
| `counter_min_matches` | 3 | Min observations before trusting matchup win rate |

---

### 4_train_model.py
Trains and evaluates all models. Saves models to `models/` and results to `data/processed/evaluation_results.json`. Appends a row to `data/output/training_runs.csv` after each run.

**Key config:**

| Setting | Default | Notes |
|---|---|---|
| `test_size` | 0.2 | 80/20 temporal train/test split |
| `recency_halflife_days` | 180 | Exponential decay weight; `None` = uniform |
| `tune_hyperparams` | `False` | Enable to run Optuna tuning (slow, ~10 min) |
| `tune_n_trials` | 50 | Number of Optuna trials per model |
| `train_logistic` | `True` | |
| `train_lightgbm` | `True` | |
| `train_catboost` | `True` | Requires `pip install catboost` |
| `train_ensemble` | `True` | Stacks all trained base models |

**Hyperparameter tuning workflow:**
1. Set `tune_hyperparams: True` and run the script
2. Copy the logged best params into `lgbm_params` / `catboost_params` in CONFIG
3. Set `tune_hyperparams: False` — tuned params are now used permanently

---

### 5_predict.py
Predicts the outcome of a single match given teams and draft.
- Uses the best available model (configurable via `default_model`)
- Accepts hero names or IDs

**Key config:**

| Setting | Default | Notes |
|---|---|---|
| `default_model` | `lightgbm` | Which model to load for prediction |

---

### 6_export_elo_csv.py
Exports the final Elo leaderboard to `data/output/elo_ratings.csv`. Optional utility script.

---

### 7_explain_model.py
Generates feature importance charts and a model comparison plot. Outputs to `data/output/`.

Charts produced:
- `lgbm_feature_importances.png` — LightGBM split-count importances
- `lr_feature_importances.png` — Logistic Regression signed coefficients
- `catboost_feature_importances.png` — CatBoost PredictionValuesChange importances
- `ensemble_weights.png` — Stacking meta-learner base model weights
- `model_comparison.png` — Side-by-side Accuracy / AUC / Log Loss for all models

---

## Features

### Hero Vectors
| Feature | Description |
|---|---|
| `r_pick_0..N` | Binary — 1 if Radiant picked that hero |
| `d_pick_0..N` | Binary — 1 if Dire picked that hero |
| `ban_0..N` | Binary — 1 if hero was banned (combined both sides) |

### Elo
| Feature | Description |
|---|---|
| `radiant_elo` | Radiant team's Elo rating before this match |
| `dire_elo` | Dire team's Elo rating before this match |
| `elo_diff` | `radiant_elo - dire_elo` |

### Team Encoding
| Feature | Description |
|---|---|
| `radiant_team_enc` | Target-encoded historical Radiant win rate for this team |
| `dire_team_enc` | Target-encoded historical Dire win rate for this team |
| `team_enc_diff` | Difference between the two |

Leave-one-out encoding is used to prevent leakage. Teams with fewer than 5 appearances fall back to the global mean.

### Draft Order
| Feature | Description |
|---|---|
| `pick_order_0..9_hero` | Hero picked at each draft slot (normalised to [0,1]) |
| `pick_order_0..9_team` | Side that picked at each slot (0=Radiant, 1=Dire) |
| `radiant_has_last_pick` | 1.0 if Radiant made the final pick (counter-pick advantage) |

### Team Form (chronological — no leakage)
| Feature | Description |
|---|---|
| `radiant_recent_form` | Radiant win rate over last 10 matches |
| `dire_recent_form` | Dire win rate over last 10 matches |
| `recent_form_diff` | `radiant_recent_form - dire_recent_form` |

### Hero Experience (chronological — no leakage)
| Feature | Description |
|---|---|
| `radiant_hero_exp_mean` | Avg times Radiant's heroes have been picked by this team before |
| `radiant_hero_exp_min` | Minimum across Radiant's 5 heroes (weakest link) |
| `dire_hero_exp_mean` | Same for Dire |
| `dire_hero_exp_min` | Same for Dire |

### Cheese Score (global stat)
| Feature | Description |
|---|---|
| `radiant_cheese_score` | Sum of `win_rate × (1 − pick_rate)` for Radiant's picks |
| `dire_cheese_score` | Same for Dire |
| `cheese_score_diff` | Difference |

Rewards heroes with a high win rate but low pick rate (unconventional/surprise picks).

### Hero Synergy (chronological — no leakage)
| Feature | Description |
|---|---|
| `radiant_synergy_score` | Sum of historical win rates for all 10 same-team hero pairs (C(5,2)) |
| `dire_synergy_score` | Same for Dire |
| `synergy_diff` | `radiant - dire` |

Falls back to 0.5 if a pair has been seen fewer than 3 times.

### Hero Counter Matchups (chronological — no leakage)
| Feature | Description |
|---|---|
| `radiant_counter_score` | Avg historical Radiant win rate across all 25 cross-team hero matchups (5×5) |
| `counter_diff` | `radiant_counter_score - 0.5` |

Falls back to 0.5 if a matchup has been seen fewer than 3 times.

### Other
| Feature | Description |
|---|---|
| `patch` | Numeric patch ID of the match |

---

## Models

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline — StandardScaler + L2 penalty. Fast and interpretable. |
| **LightGBM** | Gradient boosted trees. Strong on tabular data. Optuna-tuned. |
| **CatBoost** | Gradient boosted trees. Handles categorical-style features well. Optuna-tuned. |
| **Stacking Ensemble** | Logistic Regression meta-learner trained on out-of-fold predictions from all 3 base models. |

### Training Enhancements
- **Recency weighting** — exponential decay by match age. Matches 180 days old get weight 0.5; older matches get progressively lower weight.
- **Temporal train/test split** — matches are split 80/20 in time order. No random shuffling, preventing data leakage.
- **Optuna hyperparameter tuning** — Bayesian search over 50 trials, maximising ROC-AUC. Run once, best params baked into CONFIG.

### Current Best Results (most recent run)

| Model | Accuracy | AUC | Log Loss |
|---|---|---|---|
| Logistic Regression | 0.6143 | 0.6677 | 0.6614 |
| LightGBM | 0.6076 | 0.6683 | 0.6502 |
| CatBoost | 0.6267 | 0.6781 | 0.6442 |
| Ensemble | 0.6314 | 0.6801 | 0.6439 |

---

## Folder Structure

```
Dota 2/
├── data/
│   ├── raw/                         # Raw data from OpenDota (not committed)
│   │   ├── matches/                 # One JSON per match
│   │   ├── teams.json
│   │   └── hero_constants.json
│   ├── processed/                   # Generated by scripts 2–3
│   │   ├── matches_with_elo.json
│   │   ├── features.csv
│   │   ├── feature_names.json
│   │   ├── hero_index.json
│   │   ├── team_encoder.json
│   │   ├── elo_ratings.json
│   │   ├── evaluation_results.json
│   │   ├── team_form.json
│   │   ├── team_hero_exp.json
│   │   ├── hero_cheese_ratings.json
│   │   ├── hero_synergy_rates.json
│   │   └── hero_counter_rates.json
│   └── output/                      # Charts and logs
│       ├── training_runs.csv
│       ├── model_comparison.png
│       ├── lgbm_feature_importances.png/.csv
│       ├── lr_feature_importances.png/.csv
│       ├── catboost_feature_importances.png/.csv
│       └── ensemble_weights.png/.csv
├── models/                          # Saved model files
│   ├── logistic_regression.pkl
│   ├── lightgbm_model.pkl
│   ├── catboost_model.pkl
│   ├── ensemble_model.pkl
│   └── ensemble_base_models.json
├── 1_collect_data.py
├── 2_build_elo.py
├── 3_build_features.py
├── 4_train_model.py
├── 5_predict.py
├── 6_export_elo_csv.py
├── 7_explain_model.py
└── requirements.txt
```

---

## Using predict() in your own code

```python
from predict import predict, load_resources, load_model

resources = load_resources()
model = load_model()

result = predict(
    radiant_team="Team Spirit",
    dire_team="OG",
    radiant_picks=["Anti-Mage", "Invoker", "Rubick", "Tidehunter", "Gyrocopter"],
    dire_picks=["Pudge", "Crystal Maiden", "Juggernaut", "Faceless Void", "Lion"],
    bans=["Techies", "Tinker", "Leshrac", "Broodmother",
          "Mirana", "Phantom Assassin", "Storm Spirit", "Nature's Prophet"],
    resources=resources,
    model=model,
)

print(f"Prediction: {result['prediction']} wins ({result['radiant_win_probability']:.1%} Radiant)")
```
