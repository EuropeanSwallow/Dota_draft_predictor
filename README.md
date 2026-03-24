# Dota 2 Draft Outcome Predictor

Predicts the probability of Radiant winning a pro Dota 2 match based on the draft,
team names, and a custom-built Elo rating system.

## Setup

```bash
cd "Dota 2 model"
pip install -r requirements.txt
```

Get a free OpenDota API key at https://www.opendota.com/api-keys and add it to
`1_collect_data.py` → `CONFIG["api_key"]`. Not required, but raises rate limits.

## Run Order

```bash
python 1_collect_data.py     # ~3 hours for 10k matches at free tier
python 2_build_elo.py        # seconds
python 3_build_features.py   # seconds
python 4_train_model.py      # 1-5 minutes
python 5_predict.py          # instant
```

## Folder Structure

```
Dota 2 model/
├── data/
│   ├── raw/
│   │   ├── matches/          # one JSON per match
│   │   ├── teams.json
│   │   └── hero_constants.json
│   └── processed/
│       ├── matches_with_elo.json
│       ├── features.csv
│       ├── feature_names.json
│       ├── hero_index.json
│       ├── team_encoder.json
│       ├── elo_ratings.json
│       └── evaluation_results.json
├── models/
│   ├── logistic_regression.pkl
│   └── lightgbm_model.pkl
├── 1_collect_data.py
├── 2_build_elo.py
├── 3_build_features.py
├── 4_train_model.py
├── 5_predict.py
└── requirements.txt
```

## Expected Accuracy

Draft-only models for pro Dota typically achieve **55–62% accuracy**.
The draft alone doesn't determine outcomes — player execution matters heavily.
Adding team Elo pushes accuracy up toward the higher end of that range.

## Configuration

Each script has a `CONFIG` block at the top. Key settings:

| Script | Setting | Default | Notes |
|--------|---------|---------|-------|
| 1 | `target_match_count` | 10,000 | More data = better model |
| 1 | `request_delay` | 1.05s | Keeps under 60 req/min |
| 2 | `k_factor` | 32 | Higher = faster Elo shifts |
| 3 | `last_n_patches` | 3 | `None` = use all data |
| 4 | `default_model` | lightgbm | Best accuracy |

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
